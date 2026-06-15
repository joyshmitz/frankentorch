use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;
fn main() {
    let nthreads = rayon::current_num_threads();
    let (rows, cols) = (16384usize, 64usize);
    let x: Vec<f64> = (0..rows).map(|i| 0.5 + (i % 100) as f64 * 0.01).collect();
    let run = |s: &mut FrankenTorchSession| { let v=s.tensor_variable(x.clone(),vec![rows],false).unwrap(); std::hint::black_box(s.tensor_vander(v, Some(cols), true).unwrap()); };
    let bench = |reps: usize| { let mut s=FrankenTorchSession::new(ExecutionMode::Strict); run(&mut s); let mut best=f64::INFINITY; for _ in 0..reps { let t=Instant::now(); run(&mut s); best=best.min(t.elapsed().as_secs_f64()*1e3);} best };
    let p1=rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let old=p1.install(|| bench(20));
    let pn=rayon::ThreadPoolBuilder::new().build().unwrap();
    let new=pn.install(|| bench(20));
    println!("vander [{rows},{cols}]: OLD(1t) {old:.2}ms NEW({nthreads}t) {new:.2}ms => {:.2}x", old/new);
}
