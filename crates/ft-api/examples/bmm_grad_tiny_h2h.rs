use std::error::Error;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill(n: usize, salt: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (((i + salt) % 17) as f64 - 8.0) * 0.05)
        .collect()
}

fn run_ft(batch: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..4 {
        let ad = fill(batch * n * n, 0);
        let bd = fill(batch * n * n, 7);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(ad, vec![batch, n, n], true)
            .map_err(boxed)?;
        let b = s
            .tensor_variable(bd, vec![batch, n, n], true)
            .map_err(boxed)?;
        let start = Instant::now();
        let c = s.tensor_matmul(a, b).map_err(boxed)?;
        let loss = s.tensor_sum(c).map_err(boxed)?;
        s.tensor_backward(loss).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn run_addmm_ft(m: usize, k: usize, n: usize) -> Result<(f64, f64), Box<dyn Error>> {
    let input_data = fill(m * n, 3);
    let mat1_data = fill(m * k, 11);
    let mat2_data = fill(k * n, 19);
    let mut best = f64::INFINITY;
    let mut checksum = 0.0;
    for _ in 0..3 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = s
            .tensor_variable(input_data.clone(), vec![m, n], true)
            .map_err(boxed)?;
        let mat1 = s
            .tensor_variable(mat1_data.clone(), vec![m, k], true)
            .map_err(boxed)?;
        let mat2 = s
            .tensor_variable(mat2_data.clone(), vec![k, n], true)
            .map_err(boxed)?;
        let start = Instant::now();
        let y = s.tensor_addmm(input, mat1, mat2, 0.5, 1.0).map_err(boxed)?;
        let loss = s.tensor_sum(y).map_err(boxed)?;
        let report = s.tensor_backward(loss).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            let grad_mat1 = s
                .tensor_gradient(&report, mat1)
                .ok_or_else(|| boxed("missing mat1 gradient"))?;
            let grad_mat2 = s
                .tensor_gradient(&report, mat2)
                .ok_or_else(|| boxed("missing mat2 gradient"))?;
            checksum = grad_mat1.iter().sum::<f64>() + grad_mat2.iter().sum::<f64>();
            best = elapsed_ms;
        }
    }
    Ok((best, checksum))
}

fn main() -> Result<(), Box<dyn Error>> {
    for (batch, n) in [
        (20000usize, 16usize),
        (10000usize, 32usize),
        (5000usize, 64usize),
    ] {
        let ft_ms = run_ft(batch, n)?;
        println!("B={batch} n={n}: FT {ft_ms:.1} ms");
    }
    for (m, k, n) in [
        (2048usize, 2048usize, 2048usize),
        (4096usize, 1024usize, 4096usize),
    ] {
        let (ft_ms, ft_sum) = run_addmm_ft(m, k, n)?;
        println!("addmm M={m} K={k} N={n}: FT {ft_ms:.1} ms grad_sum {ft_sum:.6e}");
    }
    Ok(())
}
