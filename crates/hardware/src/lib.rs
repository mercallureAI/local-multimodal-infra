use chrono::Utc;
use local_core::{DeviceSpec, ResourceSnapshot};
use sysinfo::System;

pub fn snapshot() -> ResourceSnapshot {
    let mut system = System::new_all();
    system.refresh_memory();
    let total_ram_mb = system.total_memory() / 1024 / 1024;
    let used_ram_mb = system.used_memory() / 1024 / 1024;

    // CPU/RAM is implemented with sysinfo. CUDA/DML discovery is intentionally
    // conservative for MVP: CUDA provider selection is configured in backend-ort,
    // while full NVML probing is a follow-up.
    ResourceSnapshot {
        cpu_cores: system.cpus().len(),
        total_ram_mb,
        used_ram_mb,
        devices: DeviceSpec {
            has_cuda: false,
            cuda_devices: Vec::new(),
            has_dml: cfg!(target_os = "windows"),
        },
        captured_at: Utc::now(),
    }
}
