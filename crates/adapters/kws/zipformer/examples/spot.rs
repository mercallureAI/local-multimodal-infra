//! Spots wake words in WAV files (16 kHz mono), as a live stream would get
//! them (100 ms at a time), and prints what it heard, when, and how fast:
//!
//!     cargo run --release -p local-adapter-kws-zipformer --example spot -- \
//!         <model dir> <words: "M3,小M" or @keywords.txt> <wav>...
//!
//! `KWS_LEVEL=0` turns the input leveling off; `KWS_TUNE=<boost>,<threshold>,
//! <short threshold>` sets the keywords' (words only, not keywords.txt);
//! `KWS_PATHS=<n>` the paths the search keeps.

use local_adapter_kws_zipformer::KeywordSpotter;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, words, wavs @ ..] = args.as_slice() else {
        return Err("usage: spot <model dir> <words|@keywords.txt> <wav>...".into());
    };
    let mut spotter = KeywordSpotter::load(std::path::Path::new(dir))?;
    let unread = match words.strip_prefix('@') {
        Some(file) => {
            let lines: Vec<String> = std::fs::read_to_string(file)?
                .lines()
                .map(str::to_string)
                .collect();
            spotter.set_keyword_lines(&lines)
        }
        None => spotter.set_keywords(&words.split(',').map(str::to_string).collect::<Vec<_>>()),
    };
    if !unread.is_empty() {
        eprintln!("not used: {unread:?}");
    }
    let (mut audio_s, mut busy_s) = (0.0, 0.0);
    for wav in wavs {
        let mut reader = hound::WavReader::open(wav)?;
        let rate = reader.spec().sample_rate as f64;
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<_, _>>()?;
        let mut spotter = KeywordSpotter::load(std::path::Path::new(dir))?;
        if std::env::var("KWS_LEVEL").is_ok_and(|v| v == "0") {
            spotter.set_leveling(false);
        }
        if let Some(paths) = std::env::var("KWS_PATHS").ok().and_then(|v| v.parse().ok()) {
            spotter.set_paths(paths);
        }
        if let Some(tb) = std::env::var("KWS_TB").ok().and_then(|v| v.parse().ok()) {
            spotter.set_trailing_blanks(tb);
        }
        if let Ok(tune) = std::env::var("KWS_TUNE") {
            let v: Vec<f32> = tune
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            if let [boost, threshold, short] = v[..] {
                spotter.tune(boost, threshold, short);
            }
        }
        match words.strip_prefix('@') {
            Some(file) => {
                let lines: Vec<String> = std::fs::read_to_string(file)?
                    .lines()
                    .map(str::to_string)
                    .collect();
                spotter.set_keyword_lines(&lines);
            }
            None => {
                spotter.set_keywords(&words.split(',').map(str::to_string).collect::<Vec<_>>());
            }
        }
        let started = Instant::now();
        let mut heard = Vec::new();
        let tail = vec![0.0f32; (0.66 * rate) as usize];
        for chunk in samples.chunks(1600).chain(std::iter::once(tail.as_slice())) {
            for d in spotter.accept(chunk)? {
                heard.push(format!(
                    "{} {:.2}-{:.2}s p={:.2}",
                    d.keyword,
                    d.start as f64 / rate,
                    d.end as f64 / rate,
                    d.score
                ));
            }
        }
        busy_s += started.elapsed().as_secs_f64();
        audio_s += samples.len() as f64 / rate;
        println!(
            "{} {:.2}s {heard:?}",
            std::path::Path::new(wav)
                .file_name()
                .unwrap()
                .to_string_lossy(),
            samples.len() as f64 / rate
        );
    }
    println!("RTF {:.4}", busy_s / audio_s.max(1e-9));
    Ok(())
}
