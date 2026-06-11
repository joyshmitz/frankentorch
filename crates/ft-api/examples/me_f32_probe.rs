//! A/B: public tensor_matrix_exp f32 — native kernel (b3o90) vs f64-upcast reference.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;
fn main(){
    let n=512usize;
    let a: Vec<f32>=(0..n*n).map(|i|((i*2654435761usize)%101)as f32*0.0002-0.01).collect();
    let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
    let t=s.tensor_variable_f32(a.clone(),vec![n,n],false).unwrap();
    // warm + time native f32
    let _=s.tensor_matrix_exp(t).unwrap();
    let it=10;
    let s0=Instant::now(); for _ in 0..it { let tt=s.tensor_variable_f32(a.clone(),vec![n,n],false).unwrap(); std::hint::black_box(s.tensor_matrix_exp(tt).unwrap()); } let native=s0.elapsed().as_secs_f64()*1e3/it as f64;
    // reference: cast f32->f64, expm, cast back (the OLD path)
    let s1=Instant::now(); for _ in 0..it { let tt=s.tensor_variable_f32(a.clone(),vec![n,n],false).unwrap(); let t64=s.tensor_to_dtype(tt, ft_core::DType::F64).unwrap(); let o64=s.tensor_matrix_exp(t64).unwrap(); std::hint::black_box(s.tensor_to_dtype(o64, ft_core::DType::F32).unwrap()); } let upcast=s1.elapsed().as_secs_f64()*1e3/it as f64;
    println!("matrix_exp f32 {n}x{n}: native={native:.2}ms  f64-upcast={upcast:.2}ms  speedup={:.2}x", upcast/native);
}
