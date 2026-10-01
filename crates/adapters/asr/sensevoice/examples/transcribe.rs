//! Transcribes WAV files, one `file<TAB>text` line each:
//! `cargo run --release -p local-adapter-sensevoice-asr --example transcribe -- <model dir> <wav>...`
use local_adapter_sensevoice_asr::SenseVoiceAsrAdapter;
use local_core::{
    AdapterKind, ArtifactKind, BackendKind, FileRef, InferenceOutput, ModelArtifact, ModelSpec,
    ResourceRequirement, RuntimePolicy,
};
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().expect("model dir"));
    let spec = ModelSpec {
        id: "sensevoice-small-onnx".to_string(),
        name: "SenseVoiceSmall ONNX".to_string(),
        enabled: true,
        task_kinds: Vec::new(),
        adapter: AdapterKind::SenseVoiceAsr,
        backend: BackendKind::Ort,
        artifacts: vec![ModelArtifact {
            kind: ArtifactKind::Local,
            path: model_dir,
            source_path: None,
            sha256: None,
            url: None,
            repo_id: None,
            revision: None,
            files: Vec::new(),
            allow_patterns: Vec::new(),
            metadata: Default::default(),
        }],
        runtime: RuntimePolicy {
            provider_order: vec!["cpu".to_string()],
            ..Default::default()
        },
        resources: ResourceRequirement::default(),
        load_policy: Default::default(),
        metadata: Default::default(),
    };
    let mut adapter = SenseVoiceAsrAdapter::load(&spec).expect("load SenseVoice");
    for wav in args {
        let output = adapter
            .transcribe(&FileRef::local(PathBuf::from(&wav)))
            .expect("transcribe");
        let InferenceOutput::AsrTranscription { text, .. } = output else {
            panic!("unexpected output");
        };
        println!("{wav}\t{text}");
    }
}
