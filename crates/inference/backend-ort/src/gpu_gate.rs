//! One process-wide gate around ORT calls that touch the GPU.
//!
//! ORT captures CUDA graphs in global capture mode: while one thread captures,
//! a synchronizing CUDA call on any other thread (a stream sync, an
//! allocation) fails with "operation not permitted when stream is capturing"
//! and invalidates the capture. Every device-touching call in this crate holds
//! the gate shared; the runs that may capture hold it exclusively. Captures
//! happen once per graph, so the gate is uncontended afterwards.

use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

static GATE: RwLock<()> = RwLock::new(());

pub(crate) fn gpu_shared() -> RwLockReadGuard<'static, ()> {
    GATE.read().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn gpu_exclusive() -> RwLockWriteGuard<'static, ()> {
    GATE.write().unwrap_or_else(PoisonError::into_inner)
}

/// Held for one run.
pub(crate) enum GpuGate {
    #[allow(dead_code)]
    Shared(RwLockReadGuard<'static, ()>),
    #[allow(dead_code)]
    Exclusive(RwLockWriteGuard<'static, ()>),
}
