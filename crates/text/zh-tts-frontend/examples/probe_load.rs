use std::time::Instant;
fn main() {
    let dir = std::env::args().nth(1).expect("asset dir");
    let t = Instant::now();
    let _j = jieba_rs::Jieba::new();
    eprintln!("jieba: {:?}", t.elapsed());
    let t = Instant::now();
    let cc =
        ferrous_opencc::OpenCC::from_config(ferrous_opencc::config::BuiltinConfig::S2tw).unwrap();
    eprintln!(
        "opencc: {:?} -> {}",
        t.elapsed(),
        cc.convert("我们一起去银行")
    );
    let t = Instant::now();
    let backend =
        local_backend_ort::OrtBackend::new(local_backend_ort::ProviderSelection::from_strings(&[
            "cpu".to_string(),
        ]));
    let s = backend
        .load_session(std::path::Path::new(&dir).join("g2pw/g2pw_int8.onnx"))
        .unwrap();
    eprintln!(
        "ort g2pw: {:?} inputs={:?}",
        t.elapsed(),
        s.inputs()
            .iter()
            .map(|i| i.name.clone())
            .collect::<Vec<_>>()
    );
}
