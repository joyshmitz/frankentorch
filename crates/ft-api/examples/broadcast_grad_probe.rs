//! Probe: do ft binary ops auto-broadcast, and do their gradients sum-reduce
//! back to the original operand shapes (torch broadcasting-backward semantics)?
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn try_binop(name: &str, lshape: Vec<usize>, rshape: Vec<usize>) {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let ln: usize = lshape.iter().product();
    let rn: usize = rshape.iter().product();
    let a = s
        .tensor_variable(
            (1..=ln).map(|i| i as f64 * 0.5).collect(),
            lshape.clone(),
            true,
        )
        .unwrap();
    let b = s
        .tensor_variable(
            (1..=rn).map(|i| i as f64 * 0.3 + 0.1).collect(),
            rshape.clone(),
            true,
        )
        .unwrap();
    let out = match name {
        "add" => s.tensor_add(a, b),
        "mul" => s.tensor_mul(a, b),
        "sub" => s.tensor_sub(a, b),
        "div" => s.tensor_div(a, b),
        _ => unreachable!(),
    };
    match out {
        Ok(o) => {
            let oshape = s.tensor_shape(o).unwrap();
            let loss = s.tensor_sum(o).unwrap();
            match s.tensor_backward(loss) {
                Ok(report) => {
                    let ga = s.tensor_gradient(&report, a).map(<[f64]>::len);
                    let gb = s.tensor_gradient(&report, b).map(<[f64]>::len);
                    println!(
                        "{name} {lshape:?} op {rshape:?} -> out {oshape:?}; grad_a_len={ga:?} (want {ln}); grad_b_len={gb:?} (want {rn})"
                    );
                }
                Err(e) => println!(
                    "{name} {lshape:?} op {rshape:?} -> out {oshape:?}; BACKWARD_ERR {e:?}"
                ),
            }
        }
        Err(e) => println!("{name} {lshape:?} op {rshape:?} -> FWD_ERR {e:?}"),
    }
}

fn main() {
    for op in ["add", "mul", "sub", "div"] {
        try_binop(op, vec![3, 1], vec![1, 4]);
        try_binop(op, vec![2, 3], vec![3]);
        try_binop(op, vec![2, 3], vec![1, 3]);
    }
}
