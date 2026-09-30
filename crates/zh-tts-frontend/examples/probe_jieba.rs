fn main() {
    let jieba = jieba_rs::Jieba::new();
    for text in std::env::args().skip(1) {
        let tags: Vec<String> = jieba
            .tag(&text, true)
            .iter()
            .map(|t| format!("{}/{}", t.word, t.tag))
            .collect();
        println!("{}", tags.join(" "));
    }
}
