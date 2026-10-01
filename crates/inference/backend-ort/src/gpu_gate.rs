//! One process-wide gate around ORT calls that touch the GPU.
//!
//! ORT captures CUDA graphs in global capture mode: while one thread captures,
//! a synchronizing CUDA call on any other thread (a stream sync, an
//! allocation) fails with "operation not permitted when stream is capturing"
//! and invalidates the capture. Calls in this crate that allocate, copy, free
//! or run on the device hold the gate shared (binding tensors already on the
//! session's device and releasing to ORT's arena make no CUDA call and do
//! not); the runs that may capture hold it exclusively. Captures
//! happen once per graph, so the gate is uncontended afterwards.

use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

static GATE: RwLock<()> = RwLock::new(());

pub(crate) fn gpu_shared() -> RwLockReadGuard<'static, ()> {
    GATE.read().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn gpu_exclusive() -> RwLockWriteGuard<'static, ()> {
    GATE.write().unwrap_or_else(PoisonError::into_inner)
}

/// A binding's last field: `Drop` puts a shared guard in it, so the fields
/// before it (device buffers, the session they keep alive) are released under
/// the gate. Only ever filled during drop, on the dropping thread.
#[derive(Default)]
pub(crate) struct DropGate(Option<RwLockReadGuard<'static, ()>>);

impl Clone for DropGate {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl DropGate {
    pub(crate) fn hold(&mut self) {
        self.0 = Some(gpu_shared());
    }
}

impl std::fmt::Debug for DropGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DropGate")
    }
}

// SAFETY: the guard is only set inside `Drop::drop` of the owning binding and
// released when that binding's fields drop, on the same thread; the struct is
// never shared or sent while it holds one.
unsafe impl Send for DropGate {}
unsafe impl Sync for DropGate {}

/// Held for one run.
pub(crate) enum GpuGate {
    #[allow(dead_code)]
    Shared(RwLockReadGuard<'static, ()>),
    #[allow(dead_code)]
    Exclusive(RwLockWriteGuard<'static, ()>),
}
