//! Text rules of the conversation: what takes the floor from the bot, what
//! is spoken, and where a streamed answer can be spoken already.

/// Utterances made only of these show the speaker is listening; they do not
/// take the floor from the bot.
const BACKCHANNELS: &[&str] = &[
    "原来如此",
    "明白了",
    "知道了",
    "没问题",
    "有道理",
    "这样啊",
    "然后呢",
    "gotit",
    "okay",
    "haha",
    "hehe",
    "mhmm",
    "right",
    "sure",
    "isee",
    "cool",
    "nice",
    "yeah",
    "好的",
    "是的",
    "对的",
    "好吧",
    "行吧",
    "明白",
    "可以",
    "没错",
    "好嘞",
    "yes",
    "yep",
    "yup",
    "huh",
    "hmm",
    "mhm",
    "lol",
    "ok",
    "uh",
    "um",
    "mm",
    "嗯",
    "啊",
    "哦",
    "噢",
    "哎",
    "诶",
    "唉",
    "呃",
    "额",
    "对",
    "好",
    "是",
    "行",
    "哈",
    "嘿",
    "呵",
    "嗷",
    "哼",
    "呀",
    "啦",
    "嘛",
    "呢",
    "咳",
];
/// One character left after backchannels that still takes the floor.
const FLOOR_WORDS: &[char] = &['停', '不', '别', '等', '喂'];
/// A clause ends at these, whatever its length.
const CLAUSE_END: &[char] = &['。', '！', '？', '!', '?', '；', ';', '…', '\n'];
/// A clause may end at these once it is long enough.
const CLAUSE_PAUSE: &[char] = &['，', ',', '、', '：', ':'];
/// The first clause is spoken as soon as it is this long (at a pause), later
/// ones once they are this long: speech starts early and then flows.
const FIRST_CLAUSE_CHARS: usize = 2;
const CLAUSE_CHARS: usize = 12;
/// Spoken text is cut to this (at a sentence end).
const MAX_SPOKEN_CHARS: usize = 400;

/// Whether an utterance heard over the bot's speech means to stop it. In a
/// group (`names`), only one naming the bot does; one to one, anything but
/// backchannels ("嗯", "对对", "好的", "okay", laughter).
pub fn takes_floor(text: &str, names: Option<&[String]>) -> bool {
    let lowered = text.to_lowercase();
    if let Some(names) = names {
        return names
            .iter()
            .any(|name| !name.is_empty() && lowered.contains(&name.to_lowercase()));
    }
    let question = text.trim_end().ends_with(['?', '？']);
    let mut rest: String = lowered.chars().filter(|c| c.is_alphanumeric()).collect();
    while !rest.is_empty() {
        let Some(word) = BACKCHANNELS.iter().find(|word| rest.starts_with(**word)) else {
            // Two characters or more ("别说了"), a question ("可以吗") or a
            // one-word command ("停"); one character else is a particle.
            return rest.chars().count() >= 2
                || rest.contains('吗')
                || question
                || rest.chars().all(|c| FLOOR_WORDS.contains(&c));
        };
        rest.drain(..word.len());
    }
    false
}

/// Plain spoken text: no Markdown, links or emoji.
pub fn speakable(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_code = false;
    for word in text.split_inclusive(char::is_whitespace) {
        let fences = word.matches("```").count();
        if fences > 0 {
            in_code ^= fences % 2 == 1;
            continue;
        }
        if in_code || word.trim().starts_with("http://") || word.trim().starts_with("https://") {
            continue;
        }
        out.push_str(word);
    }
    let mut spoken: String = out
        .chars()
        .filter(|c| !matches!(c, '*' | '_' | '`' | '#' | '>' | '|' | '~'))
        .filter(|c| !is_emoji(*c))
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if spoken.chars().count() > MAX_SPOKEN_CHARS {
        let cut: String = spoken.chars().take(MAX_SPOKEN_CHARS).collect();
        let end = cut
            .rfind(['。', '！', '？', '.', '!', '?'])
            .map(|i| i + cut[i..].chars().next().map_or(1, char::len_utf8))
            .unwrap_or(cut.len());
        spoken = cut[..end].to_string();
    }
    spoken
}

fn is_emoji(c: char) -> bool {
    matches!(c as u32,
        0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0x2B00..=0x2BFF | 0xFE0F | 0x200D)
}

/// Cuts streamed text into clauses to speak as soon as each is whole.
#[derive(Debug, Default)]
pub struct ClauseSplitter {
    text: String,
    spoken_any: bool,
}

impl ClauseSplitter {
    pub fn feed(&mut self, delta: &str) -> Vec<String> {
        self.text.push_str(delta);
        let mut clauses = Vec::new();
        while let Some(cut) = self.cut() {
            let rest = self.text.split_off(cut);
            let clause = std::mem::replace(&mut self.text, rest);
            self.spoken_any = true;
            if !clause.trim().is_empty() {
                clauses.push(clause.trim().to_string());
            }
        }
        clauses
    }

    pub fn flush(&mut self) -> Option<String> {
        let rest = std::mem::take(&mut self.text);
        let rest = rest.trim();
        (!rest.is_empty()).then(|| rest.to_string())
    }

    fn cut(&self) -> Option<usize> {
        let minimum = if self.spoken_any {
            CLAUSE_CHARS
        } else {
            FIRST_CLAUSE_CHARS
        };
        for (index, c) in self.text.char_indices() {
            let end = index + c.len_utf8();
            if CLAUSE_END.contains(&c) {
                return Some(end);
            }
            if CLAUSE_PAUSE.contains(&c) && self.text[..end].trim().chars().count() >= minimum {
                return Some(end);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backchannels_do_not_take_the_floor() {
        for text in ["嗯。", "对对对", "好的好的", "OK", "哈哈哈", "明白了"] {
            assert!(!takes_floor(text, None), "{text}");
        }
        for text in ["等一下", "停", "可以吗", "真的？", "你说什么", "不对"] {
            assert!(takes_floor(text, None), "{text}");
        }
    }

    #[test]
    fn in_a_group_only_the_name_takes_the_floor() {
        let names = vec!["小乐".to_string(), "Lele".to_string()];
        assert!(takes_floor("小乐，停一下", Some(&names)));
        assert!(takes_floor("hey lele stop", Some(&names)));
        assert!(!takes_floor("等一下，我去倒杯水", Some(&names)));
    }

    #[test]
    fn spoken_text_has_no_markup_links_or_emoji() {
        assert_eq!(
            speakable("**好的**，看这里 https://x.y/z 😄 ```code``` 就行"),
            "好的，看这里 就行"
        );
    }

    #[test]
    fn clauses_start_short_then_flow() {
        let mut splitter = ClauseSplitter::default();
        let mut clauses = Vec::new();
        for delta in [
            "好的，",
            "我给你讲个笑话：",
            "为什么熊不喜欢上网",
            "？因为",
            "会被熊到。",
            "哈哈",
        ] {
            clauses.extend(splitter.feed(delta));
        }
        clauses.extend(splitter.flush());
        assert_eq!(
            clauses,
            [
                "好的，",
                "我给你讲个笑话：为什么熊不喜欢上网？",
                "因为会被熊到。",
                "哈哈"
            ]
        );
    }
}
