//! Golden-fixture generator pinning `tensor_fft` (1-D) output bit-for-bit for the
//! FFT parallelization work (`frankentorch-kgs4.13`). Covers both the
//! power-of-two Cooley-Tukey butterfly path (256-pt) and the naive
//! non-power-of-two fallback (100-pt) so the shared `dft_inplace_1d` primitive
//! used by fft/fft2/fftn is regression-pinned. (A twiddle-precompute tweak to
//! `dft_inplace_1d` was tried against this fixture and rejected — it regressed
//! large memory-bound FFTs — so the primitive is unchanged.)
//!
//!   rch exec -- cargo run -p ft-api --example fft_golden

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn dump(session: &mut FrankenTorchSession, n: usize) {
    let data: Vec<f64> = (0..n).map(|i| ((i % 37) as f64) * 0.123 - 2.0).collect();
    let x = session.tensor_variable(data, vec![n], false).unwrap();
    let y = session.tensor_fft(x, None).unwrap();
    let re = session.tensor_real(y).unwrap();
    let im = session.tensor_imag(y).unwrap();
    let re_v = session.tensor_values(re).unwrap();
    let im_v = session.tensor_values(im).unwrap();
    println!("fft n={n}");
    for (k, (r, i)) in re_v.iter().zip(im_v.iter()).enumerate() {
        println!("{k}: {:#018x} {:#018x}", r.to_bits(), i.to_bits());
    }
}

fn main() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    println!("frankentorch-kgs4.13 fft_twiddle_golden");
    dump(&mut session, 256); // power-of-two Cooley-Tukey path
    dump(&mut session, 100); // non-power-of-two naive fallback
}
