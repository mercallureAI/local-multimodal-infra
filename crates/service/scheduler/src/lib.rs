use chrono::{Duration, Utc};
use local_core::{ModelSpec, NodeStatus};

#[derive(Debug, Clone, Copy, Default)]
pub struct Scheduler;

const DEFAULT_HEARTBEAT_TTL_SECS: i64 = 60;

impl Scheduler {
    pub fn select_worker<'a>(
        &self,
        model: &ModelSpec,
        nodes: &'a [NodeStatus],
    ) -> Option<&'a NodeStatus> {
        nodes
            .iter()
            .filter(|node| {
                Utc::now().signed_duration_since(node.last_heartbeat_at)
                    <= Duration::seconds(DEFAULT_HEARTBEAT_TTL_SECS)
            })
            .filter(|node| {
                node.registration
                    .supported_backends
                    .contains(&model.backend)
            })
            .filter(|node| {
                node.registration
                    .supported_adapters
                    .contains(&model.adapter)
            })
            .max_by_key(|node| score_node(model, node))
    }
}

fn score_node(model: &ModelSpec, node: &NodeStatus) -> i64 {
    let mut score = 0_i64;
    if node.loaded_models.iter().any(|m| m == &model.id) {
        score += 1_000;
    }
    if node.registration.resources.total_ram_mb >= model.resources.min_ram_mb {
        score += 200;
    }
    if node.registration.resources.devices.has_cuda && model.resources.min_vram_mb > 0 {
        score += 100;
    }
    score -= (node.queued_jobs as i64) * 20;
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use local_core::*;

    #[test]
    fn prefers_warm_model() {
        let spec = ModelSpec {
            id: "m".into(),
            name: "M".into(),
            enabled: true,
            task_kinds: vec![TaskKind::ObjectDetect],
            adapter: AdapterKind::Yolo,
            backend: BackendKind::Ort,
            artifacts: vec![],
            runtime: RuntimePolicy::default(),
            resources: ResourceRequirement::default(),
            load_policy: LoadPolicy::default(),
            metadata: Default::default(),
        };
        let resources = ResourceSnapshot {
            cpu_cores: 1,
            total_ram_mb: 1000,
            used_ram_mb: 1,
            devices: DeviceSpec::default(),
            captured_at: Utc::now(),
        };
        let node = |id: &str, loaded: Vec<String>| NodeStatus {
            registration: WorkerRegistration {
                node_id: id.into(),
                base_url: "http://x".into(),
                registration_token: None,
                supported_backends: vec![BackendKind::Ort],
                supported_adapters: vec![AdapterKind::Yolo],
                resources: resources.clone(),
            },
            last_heartbeat_at: Utc::now(),
            loaded_models: loaded,
            queued_jobs: 0,
        };
        let nodes = vec![node("cold", vec![]), node("warm", vec!["m".into()])];
        assert_eq!(
            Scheduler
                .select_worker(&spec, &nodes)
                .unwrap()
                .registration
                .node_id,
            "warm"
        );
    }

    #[test]
    fn ignores_stale_workers() {
        let spec = ModelSpec {
            id: "m".into(),
            name: "M".into(),
            enabled: true,
            task_kinds: vec![TaskKind::ObjectDetect],
            adapter: AdapterKind::Yolo,
            backend: BackendKind::Ort,
            artifacts: vec![],
            runtime: RuntimePolicy::default(),
            resources: ResourceRequirement::default(),
            load_policy: LoadPolicy::default(),
            metadata: Default::default(),
        };
        let resources = ResourceSnapshot {
            cpu_cores: 1,
            total_ram_mb: 1000,
            used_ram_mb: 1,
            devices: DeviceSpec::default(),
            captured_at: Utc::now(),
        };
        let node = |id: &str, stale: bool| NodeStatus {
            registration: WorkerRegistration {
                node_id: id.into(),
                base_url: "http://x".into(),
                registration_token: None,
                supported_backends: vec![BackendKind::Ort],
                supported_adapters: vec![AdapterKind::Yolo],
                resources: resources.clone(),
            },
            last_heartbeat_at: if stale {
                Utc::now() - chrono::Duration::seconds(120)
            } else {
                Utc::now()
            },
            loaded_models: vec!["m".into()],
            queued_jobs: 0,
        };
        let nodes = vec![node("stale", true), node("fresh", false)];
        assert_eq!(
            Scheduler
                .select_worker(&spec, &nodes)
                .unwrap()
                .registration
                .node_id,
            "fresh"
        );
    }
}
