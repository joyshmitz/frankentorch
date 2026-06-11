//! torch-f32 parity check for the native f32 cholesky/cholesky_solve/lu_solve
//! kernels (b3o90). Same fixed input as the torch oracle; prints ft native f32
//! results to diff against torch.linalg.* f32.
//!   cargo run -q -p ft-api --example chol_f32_parity
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = vec![
        4.0f32, 2.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 2.0, 5.0,
    ];
    let b = vec![1.0f32, 2.0, 3.0, 4.0];

    let at = s.tensor_variable_f32(a.clone(), vec![4, 4], false).unwrap();
    let l = s.tensor_linalg_cholesky(at, false).unwrap();
    println!("CHOL {:?}", s.tensor_values_f32(l).unwrap());

    let bt = s.tensor_variable_f32(b.clone(), vec![4, 1], false).unwrap();
    let x = s.tensor_cholesky_solve(bt, l, false).unwrap();
    println!("CHSOLVE {:?}", s.tensor_values_f32(x).unwrap());

    let alu = s.tensor_variable_f32(a.clone(), vec![4, 4], false).unwrap();
    let (lu, piv) = s.tensor_lu_factor(alu).unwrap();
    let bt2 = s.tensor_variable_f32(b.clone(), vec![4, 1], false).unwrap();
    let xlu = s.tensor_lu_solve(lu, &piv, bt2).unwrap();
    println!("LUSOLVE {:?}", s.tensor_values_f32(xlu).unwrap());
}
