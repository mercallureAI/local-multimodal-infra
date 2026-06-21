use lcoal_adapter_index_tts::{dump_text_frontend, SentencePieceTokenizer};
use serde::Serialize;
use serde_json::Value;
use std::{env, io::Read, path::PathBuf};

#[derive(Debug, Serialize)]
struct Output<T> {
    texts: Vec<T>,
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("indextts_text_frontend: {err}");
            std::process::ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let mut bpe_model = None::<PathBuf>;
    let mut input_json = None::<String>;
    let mut texts = Vec::<String>::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bpe-model" => {
                bpe_model = Some(PathBuf::from(
                    args.next().ok_or("--bpe-model requires a path")?,
                ));
            }
            "--input-json" => {
                input_json = Some(args.next().unwrap_or_else(|| "-".to_string()));
            }
            "--text" => {
                texts.push(args.next().ok_or("--text requires a value")?);
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => texts.push(other.to_string()),
        }
    }

    if let Some(raw) = input_json {
        let raw = if raw == "-" {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| format!("read stdin: {e}"))?;
            buf
        } else if raw.trim_start().starts_with(['{', '[', '"']) {
            raw
        } else {
            std::fs::read_to_string(&raw).map_err(|e| format!("read {raw}: {e}"))?
        };
        texts.extend(texts_from_json(&raw)?);
    }
    if texts.is_empty() {
        return Err("provide text arguments, --text, or --input-json".to_string());
    }

    let tokenizer = if let Some(path) = bpe_model {
        Some(SentencePieceTokenizer::load(&path).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let dumps = texts
        .iter()
        .map(|text| dump_text_frontend(text, tokenizer.as_ref()).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&Output { texts: dumps }).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn texts_from_json(raw: &str) -> Result<Vec<String>, String> {
    let payload: Value = serde_json::from_str(raw.trim_start_matches('\u{feff}'))
        .map_err(|e| format!("parse JSON input: {e}"))?;
    match payload {
        Value::String(text) => Ok(vec![text]),
        Value::Array(items) => items
            .into_iter()
            .map(|item| match item {
                Value::String(text) => Ok(text),
                _ => Err("JSON input list must contain only strings".to_string()),
            })
            .collect(),
        Value::Object(mut object) => {
            if let Some(Value::String(text)) = object.remove("text") {
                Ok(vec![text])
            } else if let Some(Value::Array(items)) = object.remove("texts") {
                items
                    .into_iter()
                    .map(|item| match item {
                        Value::String(text) => Ok(text),
                        _ => Err("JSON input texts list must contain only strings".to_string()),
                    })
                    .collect()
            } else {
                Err("JSON object must contain text or texts".to_string())
            }
        }
        _ => Err("JSON input must be a string, list, or object with text/texts".to_string()),
    }
}

fn print_help() {
    println!(
        "Usage: indextts_text_frontend [--bpe-model PATH] [--text TEXT ...] [--input-json JSON_OR_PATH|-] [TEXT ...]\n\nOutputs JSON with input, normalized, tokenized, tokens, and optional SentencePiece pieces/token_ids."
    );
}
