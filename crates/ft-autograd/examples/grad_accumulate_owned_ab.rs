//! Same-process A/B for the CustomFunction first-contribution gradient accumulate.
//! OLD (borrowed): allocate a fresh target Vec + copy the closure's din into it via
//! `push(0.0 + v)`. NEW (owned): normalize din in place (`*v = 0.0 + *v`, bit-identical)
//! and MOVE it into the slot — no fresh allocation, no second-buffer copy. Each `din`
//! is freshly produced per iter (clone of a reused source = the cache-hot kernel output)
//! and excluded from the timer; only the accumulate step is timed. BlackThrush, 96e5d.
use std::time::Instant;

fn main() {
    let m = 8 * 64 * 8192usize; // avg_pool1d [8,64,8192] din = 4M f64
    let src: Vec<f64> = (0..m).map(|i| ((i % 251) as f64) * 0.001 - 0.12).collect();
    let reps = 60usize;

    // warm up allocator / pages
    for _ in 0..3 {
        let d = src.clone();
        let mut t = Vec::with_capacity(m);
        for &v in &d {
            t.push(0.0 + v);
        }
        std::hint::black_box(&t);
        let mut d2 = src.clone();
        for v in d2.iter_mut() {
            *v += 0.0;
        }
        std::hint::black_box(&d2);
    }

    let (mut old_best, mut new_best) = (u128::MAX, u128::MAX);
    let (mut old_sum, mut new_sum) = (0u128, 0u128);
    for _ in 0..reps {
        // OLD: fresh din (kernel output, untimed) -> alloc target + copy
        let din = src.clone();
        let a = Instant::now();
        let mut target = Vec::with_capacity(m);
        for &v in &din {
            target.push(0.0 + v);
        }
        std::hint::black_box(&target);
        let e = a.elapsed().as_micros();
        old_best = old_best.min(e);
        old_sum += e;
        drop(din);
        drop(target);

        // NEW: fresh din (kernel output, untimed) -> in-place normalize + move
        let mut din2 = src.clone();
        let a = Instant::now();
        for v in din2.iter_mut() {
            *v += 0.0;
        }
        let target = din2; // move, no alloc
        std::hint::black_box(&target);
        let e = a.elapsed().as_micros();
        new_best = new_best.min(e);
        new_sum += e;
    }
    println!("CustomFunction first-contribution accumulate A/B (m={m} f64, {reps} reps)");
    println!(
        "OLD alloc+copy : min {:>7} us  mean {:>7.1} us",
        old_best,
        old_sum as f64 / reps as f64
    );
    println!(
        "NEW normalize+move: min {:>7} us  mean {:>7.1} us",
        new_best,
        new_sum as f64 / reps as f64
    );
    println!(
        "ratio: min {:>5.2}x  mean {:>5.2}x",
        old_best as f64 / new_best as f64,
        old_sum as f64 / new_sum as f64
    );
}
