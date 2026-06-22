use std::error::Error;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill_f64(batch: usize, n: usize, salt: usize) -> Vec<f64> {
    (0..batch * n * n)
        .map(|i| (((i + salt) % 17) as f64 - 8.0) * 0.05)
        .collect()
}

fn fill_f32(batch: usize, n: usize, salt: usize) -> Vec<f32> {
    (0..batch * n * n)
        .map(|i| (((i + salt) % 17) as f32 - 8.0) * 0.05)
        .collect()
}

fn run_ft(batch: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let ad = fill_f64(batch, n, 0);
        let bd = fill_f64(batch, n, 7);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(ad, vec![batch, n, n], false)
            .map_err(boxed)?;
        let b = s
            .tensor_variable(bd, vec![batch, n, n], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let _c = s.tensor_matmul(a, b).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn run_ft_f32(batch: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let ad = fill_f32(batch, n, 0);
        let bd = fill_f32(batch, n, 7);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable_f32(ad, vec![batch, n, n], false)
            .map_err(boxed)?;
        let b = s
            .tensor_variable_f32(bd, vec![batch, n, n], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let _c = s.tensor_matmul(a, b).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn run_ft_f32_4d(batch0: usize, batch1: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let batch = batch0 * batch1;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let ad = fill_f32(batch, n, 0);
        let bd = fill_f32(batch, n, 7);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable_f32(ad, vec![batch0, batch1, n, n], false)
            .map_err(boxed)?;
        let b = s
            .tensor_variable_f32(bd, vec![batch0, batch1, n, n], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let _c = s.tensor_matmul(a, b).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn main() -> Result<(), Box<dyn Error>> {
    for (batch, n) in [
        (20000usize, 16usize),
        (10000usize, 32usize),
        (5000usize, 64usize),
    ] {
        let ft_ms = run_ft(batch, n)?;
        println!("f64 3D B={batch} n={n}: FT {ft_ms:.1} ms");
    }
    for (batch, n) in [
        (100000usize, 4usize),
        (20000usize, 16usize),
        (10000usize, 32usize),
    ] {
        let ft_ms = run_ft_f32(batch, n)?;
        println!("f32 3D B={batch} n={n}: FT {ft_ms:.3} ms");
    }
    for (b0, b1, n) in [(10000usize, 8usize, 4usize), (10000usize, 8usize, 16usize)] {
        let ft_ms = run_ft_f32_4d(b0, b1, n)?;
        println!("f32 4D [{b0},{b1},{n},{n}]: FT {ft_ms:.3} ms");
    }
    Ok(())
}
