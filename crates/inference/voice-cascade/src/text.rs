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
/// A sentence ends at these (`.` only before whitespace, see `period_ends`).
const SENTENCE_END: &[char] = &['。', '！', '？', '!', '?', '；', ';', '…', '\n'];
/// Closing marks that stay with the sentence before them ("好吗？”").
const CLOSERS: &[char] = &['”', '’', '"', '\'', '）', ')', '】', ']', '」', '』', '》'];
/// Pauses inside a sentence: the first chunk may end at one, a later one only
/// when its sentence outgrows `MAX_CHUNK_UNITS`.
const PAUSES: &[char] = &['，', ',', '、', '：', ':'];
/// The first chunk is spoken as soon as it is this long at a pause, so
/// speech starts early ("好的，").
const FIRST_CHUNK_UNITS: usize = 2;
/// Later chunks are whole sentences, merged until twice as long as the chunk
/// before (it plays while the next is synthesized) and at most this long.
const MERGE_UNITS: usize = 60;
/// A sentence longer than this is cut at its last pause (or word) before.
const MAX_CHUNK_UNITS: usize = 120;
/// Words before a `.` that do not end a sentence ("Mr. Smith").
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "no", "fig", "approx",
];
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

/// Spoken length of `text`: one unit per CJK (or other non-ASCII) letter and
/// per ASCII word or number; punctuation and spaces count nothing.
fn units(text: &str) -> usize {
    let mut count = 0;
    let mut in_word = false;
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            count += usize::from(!in_word);
            in_word = true;
        } else {
            in_word = false;
            count += usize::from(c.is_alphanumeric());
        }
    }
    count
}

/// Whether the `.` at byte `index` ends a sentence, given the character
/// after it: it must be followed by whitespace and not close a number
/// ("3. ", a list item) or an abbreviation ("Mr. ", "J. ").
fn period_ends(text: &str, index: usize, next: char) -> bool {
    if !next.is_whitespace() {
        return false;
    }
    let before = &text[..index];
    if before.ends_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    let word = before
        .rsplit(|c: char| c.is_whitespace())
        .next()
        .unwrap_or_default()
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    let single_letter = word.chars().count() == 1 && word.chars().all(|c| c.is_ascii_alphabetic());
    !single_letter && !ABBREVIATIONS.contains(&word.as_str())
}

/// Cuts streamed text into chunks to speak as soon as each is whole.
///
/// The TTS reads each chunk on its own, so a chunk is whole sentences:
/// cutting at commas breaks the prosody and pauses mid-sentence. Only the
/// first chunk may end at a pause, to start speaking early; then each chunk
/// merges sentences until it is twice as long as the one before, which is
/// playing while it is synthesized. A sentence end counts once the next
/// character shows it is one (`？”`, `……`, `3.5`, `Mr. Smith`).
#[derive(Debug, Default)]
pub struct ClauseSplitter {
    text: String,
    /// Units of the last chunk handed out; 0 before the first.
    last_units: usize,
}

impl ClauseSplitter {
    pub fn feed(&mut self, delta: &str) -> Vec<String> {
        self.text.push_str(delta);
        let mut clauses = Vec::new();
        while let Some(cut) = self.cut() {
            let rest = self.text.split_off(cut);
            let clause = std::mem::replace(&mut self.text, rest);
            self.last_units = units(&clause).max(1);
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
        let first = self.last_units == 0;
        let target = if first {
            0
        } else {
            (2 * self.last_units).min(MERGE_UNITS)
        };
        let chars: Vec<(usize, char)> = self.text.char_indices().collect();
        let mut last_end = None;
        let mut i = 0;
        while i < chars.len() {
            let (index, c) = chars[i];
            let end = index + c.len_utf8();
            if first && PAUSES.contains(&c) && units(&self.text[..end]) >= FIRST_CHUNK_UNITS {
                return Some(end);
            }
            let ends = SENTENCE_END.contains(&c)
                || (c == '.'
                    && chars
                        .get(i + 1)
                        .is_some_and(|&(_, next)| period_ends(&self.text, index, next)));
            if !ends {
                i += 1;
                continue;
            }
            // The sentence takes the end marks and closers after it; it is
            // over once something else follows.
            let mut j = i + 1;
            while j < chars.len()
                && (SENTENCE_END.contains(&chars[j].1)
                    || CLOSERS.contains(&chars[j].1)
                    || chars[j].1 == '.')
            {
                j += 1;
            }
            if j == chars.len() {
                break;
            }
            let end = chars[j].0;
            let length = units(&self.text[..end]);
            if length > MAX_CHUNK_UNITS && last_end.is_some() {
                return last_end;
            }
            if length >= target {
                return Some(end);
            }
            last_end = Some(end);
            i = j;
        }
        if units(&self.text) > MAX_CHUNK_UNITS {
            return last_end.or_else(|| self.cut_long_sentence());
        }
        None
    }

    /// Where to cut a sentence over `MAX_CHUNK_UNITS`: after its last pause
    /// within the limit, else after the last whole word.
    fn cut_long_sentence(&self) -> Option<usize> {
        let (mut pause, mut word, mut count, mut in_word) = (None, None, 0, false);
        for (index, c) in self.text.char_indices() {
            if c.is_ascii_alphanumeric() {
                if !in_word {
                    if count == MAX_CHUNK_UNITS {
                        break;
                    }
                    word = Some(index);
                    count += 1;
                }
                in_word = true;
                continue;
            }
            in_word = false;
            if c.is_alphanumeric() {
                if count == MAX_CHUNK_UNITS {
                    break;
                }
                word = Some(index);
                count += 1;
            } else if PAUSES.contains(&c) {
                pause = Some(index + c.len_utf8());
            }
        }
        pause.or(word).filter(|&cut| cut > 0)
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

    fn split(deltas: &[&str]) -> Vec<String> {
        let mut splitter = ClauseSplitter::default();
        let mut clauses = Vec::new();
        for delta in deltas {
            clauses.extend(splitter.feed(delta));
        }
        clauses.extend(splitter.flush());
        clauses
    }

    #[test]
    fn clauses_start_short_then_flow() {
        assert_eq!(
            split(&[
                "好的，",
                "我给你讲个笑话：",
                "为什么熊不喜欢上网",
                "？因为",
                "会被熊到。",
                "哈哈"
            ]),
            [
                "好的，",
                "我给你讲个笑话：为什么熊不喜欢上网？",
                "因为会被熊到。哈哈"
            ]
        );
    }

    #[test]
    fn later_chunks_end_only_at_sentence_ends() {
        assert_eq!(
            split(&["好的。今天天气不错，我们去公园散步吧，顺便买点水果。回来再说。"]),
            [
                "好的。",
                "今天天气不错，我们去公园散步吧，顺便买点水果。",
                "回来再说。"
            ]
        );
    }

    #[test]
    fn short_sentences_merge_as_chunks_grow() {
        assert_eq!(
            split(&["嗯。", "对。", "你说得对。", "我们明天见。", "再见。"]),
            ["嗯。", "对。你说得对。", "我们明天见。再见。"]
        );
    }

    #[test]
    fn a_sentence_end_waits_for_the_next_character() {
        let mut splitter = ClauseSplitter::default();
        assert!(splitter.feed("真的吗？").is_empty());
        assert!(splitter.feed("”").is_empty());
        assert_eq!(splitter.feed("他笑了"), ["真的吗？”"]);
        assert!(splitter.feed("……").is_empty());
        assert_eq!(splitter.flush().as_deref(), Some("他笑了……"));
    }

    #[test]
    fn periods_end_english_sentences_but_not_numbers_or_abbreviations() {
        assert_eq!(
            split(&["Sure. Mr. Smith paid 3.5 dollars. ", "Then he left. Bye"]),
            ["Sure.", "Mr. Smith paid 3.5 dollars.", "Then he left. Bye"]
        );
        assert_eq!(
            split(&["1. 先洗手，2. 再吃饭。"]),
            ["1. 先洗手，", "2. 再吃饭。"]
        );
    }

    #[test]
    fn long_sentences_are_cut_at_their_last_pause() {
        let long = format!(
            "{}，{}，{}。",
            "很".repeat(70),
            "长".repeat(40),
            "句".repeat(30)
        );
        let clauses = split(&["好。", &long]);
        let expected_head = format!("{}，{}，", "很".repeat(70), "长".repeat(40));
        assert_eq!(
            clauses,
            [
                "好。".to_string(),
                expected_head,
                format!("{}。", "句".repeat(30))
            ]
        );
    }

    #[test]
    fn streaming_splits_like_the_whole_text() {
        let text =
            "好的，我来说说。首先，Rust 很快。其次，它很安全！最后……就这些。Mr. Lee said so. 谢谢";
        let whole = split(&[text]);
        let pieces: Vec<String> = text.chars().map(String::from).collect();
        let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
        assert_eq!(split(&pieces), whole);
        assert_eq!(whole.concat().replace(' ', ""), text.replace(' ', ""));
    }

    #[test]
    fn units_count_cjk_letters_and_ascii_words() {
        assert_eq!(units("你好，Rust 2024 world！"), 5);
    }
}
