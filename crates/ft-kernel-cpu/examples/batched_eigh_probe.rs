use std::time::Instant;
use ft_core::{DType, Device, TensorMeta};
fn main() {
    for (bb, k) in [(100000usize,4usize),(20000,16),(4000,32)] {
        let mut data = vec![0.0f64; bb*k*k];
        for b in 0..bb { for i in 0..k { for j in 0..k {
            data[b*k*k+i*k+j] = (((b*7+i*13+j*5)%97) as f64)*0.01;
        }}}
        for b in 0..bb {
            for i in 0..k { for j in 0..k {
                let s=(data[b*k*k+i*k+j]+data[b*k*k+j*k+i])*0.5; data[b*k*k+i*k+j]=s;
            }}
            for i in 0..k { data[b*k*k+i*k+i]+=k as f64; }
        }
        let meta = TensorMeta::from_shape(vec![bb,k,k], DType::F64, Device::Cpu);
        let kmeta = TensorMeta::from_shape(vec![k,k], DType::F64, Device::Cpu);
        let mut best=f64::INFINITY; let mut esum=0.0;
        for _ in 0..6 {
            let t=Instant::now();
            let (evals,_evecs)=ft_kernel_cpu::eigh_batched_contiguous_f64(&data,&meta).unwrap();
            let el=t.elapsed().as_secs_f64()*1e3;
            if el<best {best=el; esum=evals.iter().sum();}
        }
        // bit-exact vs looping the 2-D eigh (first 50 planes)
        let (bv,_)=ft_kernel_cpu::eigh_batched_contiguous_f64(&data,&meta).unwrap();
        let mut ok=true;
        for b in 0..bb.min(50) {
            let r=ft_kernel_cpu::eigh_contiguous_f64(&data[b*k*k..(b+1)*k*k],&kmeta).unwrap();
            for i in 0..k { if bv[b*k+i].to_bits()!=r.eigenvalues[i].to_bits() {ok=false;} }
        }
        println!("B={bb} k={k}: FT batched eigh {best:.1} ms  esum {esum:.3e}  bit-exact-vs-loop:{}", if ok {"MATCH"} else {"MISMATCH"});
    }
}
