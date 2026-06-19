//! Same-worker A/B for the deferred-QL eigenvector replay block size
//! (frankentorch-x53r3). block_rows=1 is the ANCHOR (legacy per-row replay,
//! re-streams the ops Vec once per row). Larger blocks read ops once per block.
//!   rch exec -- cargo run --release -q -p ft-kernel-cpu --example eigh_replay_block_ab

use ft_kernel_cpu::eigh_tql2_replay_block_ab;

fn lcg_sym(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    let mut x: u64 = 0x9E3779B97F4A7C15;
    for slot in a.iter_mut() {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *slot = (x >> 11) as f64 / 9007199254740992.0 * 2.0 - 1.0;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    a
}

fn time_block(a: &[f64], n: usize, block: usize, it: usize) -> f64 {
    let _ = eigh_tql2_replay_block_ab(a, n, block); // warm
    let mut acc = 0u128;
    for _ in 0..it {
        acc += eigh_tql2_replay_block_ab(a, n, block);
    }
    acc as f64 / it as f64 / 1e6
}

fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &n in &[512usize, 1024, 2048] {
        let a = lcg_sym(n);
        let it = if n <= 512 {
            5
        } else if n <= 1024 {
            3
        } else {
            1
        };
        let anchor = time_block(&a, n, 1, it);
        print!("n={n:5} anchor(block=1)={anchor:9.2}ms");
        for &b in &[2usize, 4, 8, 16] {
            let t = time_block(&a, n, b, it);
            print!("  b={b}={t:8.2}({:.2}x)", anchor / t);
        }
        println!();
    }
}
