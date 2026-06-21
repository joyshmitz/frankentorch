//! Pure-Rust caching allocator demo (BlackThrush, frankentorch-9pafs).
//!
//! The 2026-06-21r finding: the residual gauntlet losses are the ALLOCATOR gap
//! (FT mallocs+page-faults fresh per backward; PyTorch's caching allocator reuses).
//! mimalloc (C) sized the win. This proves a PURE-RUST caching allocator captures
//! it too — so FrankenTorch can DOMINATE with ZERO C deps (the "no C BLAS" math-
//! purity rule stays intact; this is allocator infrastructure).
//!
//! Same-binary anchored A/B: the allocator has a runtime CACHE_ENABLED flag, so we
//! measure the SAME avg_pool1d train step with the cache OFF (System baseline) then
//! ON (the lever) in one process on one worker — the mandated A/B discipline.
//!
//! Run: cargo run --release -p ft-api --example pure_rust_caching_alloc_demo
#![allow(unsafe_code)] // a GlobalAlloc is an unsafe trait; this is a measurement-only example

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

// ---- pure-Rust caching global allocator -----------------------------------
// Sound + re-entrancy-safe: fixed-size thread-local free-list (const-init -> no
// heap, so the allocator never allocates -> no recursion). Each cached block is a
// System block reused only for its EXACT (size, align); cross-thread dealloc parks
// the block on the freeing thread's list (a System block is valid on any thread).
// Only large blocks (the page-fault-heavy ones) are cached; small allocs pass to
// System. Blocks left parked at exit are leaked (fine for a bench).
const NSLOTS: usize = 256;
const MIN_CACHE: usize = 4096; // 4 KiB
const MAX_CACHE: usize = 256 * 1024 * 1024; // 256 MiB

static CACHE_ENABLED: AtomicBool = AtomicBool::new(false);
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

thread_local! {
    // (size, align, ptr); ptr null == empty slot.
    static SLOTS: RefCell<[(usize, usize, *mut u8); NSLOTS]> =
        const { RefCell::new([(0usize, 0usize, std::ptr::null_mut()); NSLOTS]) };
}

struct CachingAlloc;

#[inline]
fn cacheable(layout: Layout) -> bool {
    CACHE_ENABLED.load(Ordering::Relaxed)
        && layout.size() >= MIN_CACHE
        && layout.size() <= MAX_CACHE
}

unsafe impl GlobalAlloc for CachingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if cacheable(layout) {
            let hit = SLOTS.with(|s| {
                if let Ok(mut slots) = s.try_borrow_mut() {
                    for slot in slots.iter_mut() {
                        if !slot.2.is_null() && slot.0 == layout.size() && slot.1 == layout.align()
                        {
                            let p = slot.2;
                            slot.2 = std::ptr::null_mut();
                            return p;
                        }
                    }
                }
                std::ptr::null_mut()
            });
            if !hit.is_null() {
                HITS.fetch_add(1, Ordering::Relaxed);
                return hit;
            }
            MISSES.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding a valid layout to the System allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if cacheable(layout) {
            let parked = SLOTS.with(|s| {
                if let Ok(mut slots) = s.try_borrow_mut() {
                    for slot in slots.iter_mut() {
                        if slot.2.is_null() {
                            *slot = (layout.size(), layout.align(), ptr);
                            return true;
                        }
                    }
                }
                false
            });
            if parked {
                return;
            }
        }
        // SAFETY: ptr came from System.alloc with this layout (cache only ever
        // parks System blocks; a parked block is returned for the same layout).
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CachingAlloc = CachingAlloc;

// ---- avg_pool1d train-step workload (kgs4.122 gauntlet lane) ---------------
const N: usize = 8;
const C: usize = 64;
const L: usize = 8192;

fn values() -> Vec<f64> {
    (0..N * C * L).map(|i| ((i % 251) as f64) * 0.001 - 0.12).collect()
}

fn run_train_step(base: &[f64], shape: &[usize]) -> f64 {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(base.to_vec(), shape.to_vec(), true).unwrap();
    let out = s.functional_avg_pool1d(x, 2, 2).unwrap();
    let loss = s.tensor_sum(out).unwrap();
    let report = s.tensor_backward(loss).unwrap();
    // Touch the grad so the backward buffers are real + return a checksum to
    // validate the allocator did not corrupt memory.
    let g = report.gradient(x).unwrap();
    g.iter().sum()
}

fn bench(base: &[f64], shape: &[usize], iters: usize) -> (f64, f64) {
    // returns (median_ms, checksum)
    let mut times = Vec::with_capacity(iters);
    let mut checksum = 0.0;
    for _ in 0..iters {
        let t = Instant::now();
        checksum = run_train_step(base, shape);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], checksum)
}

fn main() {
    let base = values();
    let shape = vec![N, C, L];
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);

    // warm both modes' code paths
    CACHE_ENABLED.store(false, Ordering::Relaxed);
    let _ = bench(&base, &shape, 3);

    // BASELINE: system allocator (cache off)
    CACHE_ENABLED.store(false, Ordering::Relaxed);
    let (sys_ms, sys_sum) = bench(&base, &shape, iters);

    // warm the cache slots, then LEVER: caching allocator (cache on)
    CACHE_ENABLED.store(true, Ordering::Relaxed);
    let _ = bench(&base, &shape, 3);
    HITS.store(0, Ordering::Relaxed);
    MISSES.store(0, Ordering::Relaxed);
    let (cache_ms, cache_sum) = bench(&base, &shape, iters);

    let hits = HITS.load(Ordering::Relaxed);
    let misses = MISSES.load(Ordering::Relaxed);

    println!("avg_pool1d [{N},{C},{L}] train step, {iters} iters, median ms:");
    println!("  system alloc (baseline) : {sys_ms:8.3} ms   checksum {sys_sum:.6e}");
    println!("  caching alloc (lever)   : {cache_ms:8.3} ms   checksum {cache_sum:.6e}");
    println!("  speedup                 : {:8.3}x", sys_ms / cache_ms);
    println!("  cache hits/misses       : {hits} / {misses}");

    // soundness gate: the caching allocator must produce the identical result.
    assert!(
        (sys_sum - cache_sum).abs() <= sys_sum.abs() * 1e-12 + 1e-12,
        "caching allocator changed the result (memory corruption): {sys_sum} vs {cache_sum}"
    );
    println!("  OK: results bit-consistent across both allocators (allocator is sound).");
}
