//! max_pool2d full gradient-penalty value check vs torch f32. frankentorch-rosw9.
use ft_api::FrankenTorchSession; use ft_autograd::BackwardOptions; use ft_core::{DType,ExecutionMode};
fn main(){
    for f32 in [false,true]{
        let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
        let xv:Vec<f64>=(0..36).map(|i|((i*48271)%211) as f64*0.013-1.3).collect();
        let mut x=s.tensor_variable(xv,vec![1,1,6,6],true).unwrap();
        if f32 { x=s.tensor_to_dtype(x,DType::F32).unwrap(); }
        let r=(||{let y=s.functional_max_pool2d(x,(2,2),(2,2))?;let sq=s.tensor_mul(y,y)?;let l=s.tensor_sum(sq)?;
            let o=BackwardOptions::for_mode(s.mode()).with_create_graph(true).with_retain_graph(true);
            let r1=s.tensor_backward_with_options(l,o)?;let g=r1.gradient_node(x).unwrap();
            let g2=s.tensor_mul(g,g)?;let pen=s.tensor_sum(g2)?;let pv=s.tensor_values(pen)?[0];
            let gr=s.tensor_autograd_grad(&[pen],&[x],None,false,false)?;
            Ok::<_,ft_autograd::AutogradError>((pv,gr[0].clone().unwrap().iter().sum::<f64>()))})();
        match r{Ok((p,gxs))=>println!("mp_{}: pen={p:.4} gx_sum={gxs:.4}",if f32{"f32"}else{"f64"}),Err(e)=>println!("ERR {e:?}")}
    }
}
