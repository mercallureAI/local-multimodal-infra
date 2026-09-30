//! Times each frontend stage: `cargo run --release --example probe -- <asset dir> [text...]`.
use local_backend_ort::{OrtBackend, ProviderSelection};
use local_zh_tts_frontend::{Options, PinyinDict, ZhFrontend};
use std::{path::Path, time::Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("asset dir");
    let dir = Path::new(&dir);
    let t = Instant::now();
    let dict = PinyinDict::load(&dir.join("pinyin")).expect("pinyin");
    eprintln!("pinyin dict {:?}: {:?}", dict, t.elapsed());
    let t = Instant::now();
    let backend = OrtBackend::new(ProviderSelection::from_strings(&["cpu".to_string()]));
    let mut frontend = ZhFrontend::load(dir, &backend, Options::paddlespeech()).expect("frontend");
    eprintln!("frontend load: {:?}", t.elapsed());
    let texts: Vec<String> = args.collect();
    let texts = if texts.is_empty() {
        vec!["我们一起去银行办业务，他在银行工作了三年。".to_string()]
    } else {
        texts
    };
    for text in texts {
        let t = Instant::now();
        let readings = frontend.readings(&text).expect("readings");
        eprintln!("{:?} {text}\n  {readings:?}", t.elapsed());
    }
}
