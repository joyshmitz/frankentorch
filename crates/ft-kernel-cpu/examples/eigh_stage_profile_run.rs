//! eigh phase breakdown (frankentorch-x53r3): reduce / backtransform(form-Q) /
//! tql2. The serial unblocked back-transform is the suspected compute-bound
//! lever distinct from the bandwidth-bound reduction.
//!   rch exec -- cargo run --release -q -p ft-kernel-cpu --example eigh_stage_profile_run

use ft_kernel_cpu::eigh_stage_profile_f64;

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

fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &n in &[512usize, 1024] {
        let a = lcg_sym(n);
        // warm
        let _ = eigh_stage_profile_f64(&a, n);
        let it = if n <= 512 { 4 } else { 2 };
        let (mut r, mut b, mut t) = (0u128, 0u128, 0u128);
        for _ in 0..it {
            let (rr, bb, tt) = eigh_stage_profile_f64(&a, n);
            r += rr;
            b += bb;
            t += tt;
        }
        let d = it as f64 * 1e6;
        println!(
            "n={n:5} reduce={:.2}ms  backtransform(formQ)={:.2}ms  tql2={:.2}ms  total~{:.2}ms",
            r as f64 / d,
            b as f64 / d,
            t as f64 / d,
            (r + b + t) as f64 / d
        );
    }
}
