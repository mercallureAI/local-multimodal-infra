//! The talker's prompt, position by position, as the official `generate`
//! builds it (`Qwen3TTSForConditionalGeneration.generate`, voice clone).
//!
//! Every position is the sum of a text part (a projected text token, or
//! nothing), a codec part (codebook codes, or nothing) and an optional raw
//! embedding (the speaker x-vector). The talker graph embeds them itself, so
//! a prompt is only ids.

use crate::artifacts::SpecialTokens;
use local_error::{InfraError, Result};

pub const GROUPS: usize = 16;
/// One codec frame: a code per codebook (`-1` = absent in prompts).
pub type Codes = [i64; GROUPS];

const NONE: Codes = [-1; GROUPS];

fn first(code: i64) -> Codes {
    let mut codes = NONE;
    codes[0] = code;
    codes
}

#[derive(Debug, Clone, PartialEq)]
pub struct Prompt {
    pub text_ids: Vec<i64>,
    pub codec_ids: Vec<Codes>,
    /// The position that carries the speaker x-vector.
    pub speaker_position: Option<usize>,
    /// Text fed one token per generated frame (then `tts_pad`).
    pub trailing: Vec<i64>,
}

impl Prompt {
    pub fn len(&self) -> usize {
        self.text_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text_ids.is_empty()
    }

    fn push(&mut self, text: i64, codec: Codes) {
        self.text_ids.push(text);
        self.codec_ids.push(codec);
    }
}

/// In-context voice cloning: the reference transcript and its codes.
#[derive(Debug, Clone, Copy)]
pub struct Reference<'a> {
    /// Tokens of `<|im_start|>assistant\n{ref_text}<|im_end|>\n`.
    pub ids: &'a [u32],
    pub codes: &'a [Codes],
}

/// `ids`: tokens of `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`.
pub fn build_prompt(
    tokens: &SpecialTokens,
    ids: &[u32],
    language: Option<i64>,
    speaker: bool,
    reference: Option<Reference<'_>>,
) -> Result<Prompt> {
    // 3 role tokens, the text, then `<|im_end|>\n<|im_start|>assistant\n`.
    if ids.len() < 9 {
        return Err(InfraError::BadRequest(
            "Qwen3-TTS text is empty".to_string(),
        ));
    }
    let ids = ids.iter().map(|id| *id as i64).collect::<Vec<_>>();
    let (role, body) = (&ids[..3], &ids[3..ids.len() - 5]);
    let mut prompt = Prompt {
        text_ids: Vec::new(),
        codec_ids: Vec::new(),
        speaker_position: None,
        trailing: Vec::new(),
    };
    for id in role {
        prompt.push(*id, NONE);
    }
    // Codec control prefix (thinking tags, language), the speaker, then pad
    // and bos; under each but the last: tts_pad, and tts_bos last.
    let mut codec_input: Vec<Option<i64>> = match language {
        Some(language) => vec![
            Some(tokens.codec_think),
            Some(tokens.codec_think_bos),
            Some(language),
            Some(tokens.codec_think_eos),
        ],
        None => vec![
            Some(tokens.codec_nothink),
            Some(tokens.codec_think_bos),
            Some(tokens.codec_think_eos),
        ],
    };
    if speaker {
        codec_input.push(None);
    }
    codec_input.push(Some(tokens.codec_pad));
    codec_input.push(Some(tokens.codec_bos));
    let prefix = codec_input.len() - 1;
    for (index, code) in codec_input[..prefix].iter().enumerate() {
        let text = if index == prefix - 1 {
            tokens.tts_bos
        } else {
            tokens.tts_pad
        };
        match code {
            Some(code) => prompt.push(text, first(*code)),
            None => {
                prompt.speaker_position = Some(prompt.len());
                prompt.push(text, NONE);
            }
        }
    }
    match reference {
        Some(reference) => {
            if reference.ids.len() < 6 || reference.codes.is_empty() {
                return Err(InfraError::BadRequest(
                    "Qwen3-TTS reference text and audio must not be empty".to_string(),
                ));
            }
            let ref_body = &reference.ids[3..reference.ids.len() - 2];
            let mut text = ref_body.iter().map(|id| *id as i64).collect::<Vec<_>>();
            text.extend_from_slice(body);
            text.push(tokens.tts_eos);
            let mut codec = vec![first(tokens.codec_bos)];
            codec.extend_from_slice(reference.codes);
            if text.len() > codec.len() {
                prompt.trailing = text.split_off(codec.len());
            } else {
                text.resize(codec.len(), tokens.tts_pad);
            }
            for (text, codes) in text.into_iter().zip(codec) {
                prompt.push(text, codes);
            }
        }
        None => {
            prompt.push(body[0], first(tokens.codec_bos));
            prompt.trailing = body[1..].to_vec();
            prompt.trailing.push(tokens.tts_eos);
        }
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens() -> SpecialTokens {
        SpecialTokens {
            tts_bos: 151672,
            tts_eos: 151673,
            tts_pad: 151671,
            codec_bos: 2149,
            codec_eos: 2150,
            codec_pad: 2148,
            codec_think: 2154,
            codec_nothink: 2155,
            codec_think_bos: 2156,
            codec_think_eos: 2157,
        }
    }

    // `<|im_start|>assistant\n你好，我是你的语音助手，今天有什么可以帮你的吗？<|im_end|>\n<|im_start|>assistant\n`
    const IDS: [u32; 22] = [
        151644, 77091, 198, 108386, 3837, 104198, 103929, 105761, 110498, 3837, 100644, 104139,
        73670, 99663, 103929, 101037, 11319, 151645, 198, 151644, 77091, 198,
    ];

    #[test]
    fn x_vector_prompt_matches_the_official_layout() {
        let t = tokens();
        let prompt = build_prompt(&t, &IDS, Some(2055), true, None).unwrap();
        // role(3) + think, think_bos, lang, think_eos, speaker, pad + first text with bos
        assert_eq!(prompt.len(), 10);
        assert_eq!(&prompt.text_ids[..3], &[151644, 77091, 198]);
        assert_eq!(
            &prompt.text_ids[3..],
            &[t.tts_pad, t.tts_pad, t.tts_pad, t.tts_pad, t.tts_pad, t.tts_bos, 108386]
        );
        let g0 = prompt.codec_ids.iter().map(|c| c[0]).collect::<Vec<_>>();
        assert_eq!(g0, vec![-1, -1, -1, 2154, 2156, 2055, 2157, -1, 2148, 2149]);
        assert_eq!(prompt.speaker_position, Some(7));
        assert_eq!(prompt.trailing.len(), IDS.len() - 3 - 5 - 1 + 1);
        assert_eq!(*prompt.trailing.last().unwrap(), t.tts_eos);
    }

    #[test]
    fn icl_prompt_pads_text_under_the_reference_codes() {
        let t = tokens();
        let ref_ids = [151644, 77091, 198, 1, 2, 3, 151645, 198];
        let codes = vec![[7i64; GROUPS]; 20];
        let reference = Reference {
            ids: &ref_ids,
            codes: &codes,
        };
        let prompt = build_prompt(&t, &IDS, None, true, Some(reference)).unwrap();
        // role(3) + nothink, think_bos, think_eos, speaker, pad (tts_bos) + bos + 20 codes
        assert_eq!(prompt.len(), 3 + 5 + 21);
        assert_eq!(prompt.text_ids[8..11], [1, 2, 3]);
        assert_eq!(prompt.codec_ids[8][0], t.codec_bos);
        assert_eq!(prompt.codec_ids[9], [7; GROUPS]);
        // 3 ref + 14 text + eos = 18 < 21: padded, nothing trails.
        assert_eq!(prompt.text_ids[8 + 17], t.tts_eos);
        assert_eq!(*prompt.text_ids.last().unwrap(), t.tts_pad);
        assert!(prompt.trailing.is_empty());
    }
}
