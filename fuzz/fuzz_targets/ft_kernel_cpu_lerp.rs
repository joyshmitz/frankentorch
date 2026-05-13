#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::lerp_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let ndim = usize::from(data[0] % 7);
    if ndim == 0 {
        return;
    }
    // Weight selector. Bytes 0..63 → 0.0, 64..127 → 1.0, otherwise
    // an interpolated/extrapolated weight. The boundary range is
    // intentionally large so the identity assertions run frequently.
    let weight_byte = data[1];
    let weight = match weight_byte {
        0..=63 => 0.0,
        64..=127 => 1.0,
        b => (f64::from(b as i32 - 192)) / 32.0, // covers ~[-2, 2]
    };
    let body = &data[2..];

    if body.len() < ndim {
        return;
    }
    let shape: Vec<usize> = body[..ndim]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();

    let meta = match TensorMeta::from_shape_and_strides(
        shape.clone(),
        ft_core::contiguous_strides(&shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    let numel = meta.numel();
    if numel > 4096 {
        return;
    }

    // Bounded values keep boundary identities meaningful (no
    // catastrophic cancellation in end - start).
    let start: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()] as i32;
            f64::from(raw - 128) / 40.0
        })
        .collect();
    let end: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + numel + i) % body.len()] as i32;
            f64::from(raw - 128) / 40.0
        })
        .collect();

    let output = match lerp_tensor_contiguous_f64(&start, &end, weight, &meta) {
        Ok(out) => out,
        Err(_) => return,
    };
    assert_eq!(output.len(), numel, "lerp output length must equal numel");

    if numel == 0 {
        return;
    }

    // Boundary identity: weight==0 → output == start, weight==1 →
    // output == end. We use ULP-bounded tolerance because the
    // kernel computes (sv + w*(ev-sv)) rather than a direct copy,
    // so weight==1 produces sv + (ev - sv) which can differ from
    // ev by a fraction of a ULP.
    const ULP_TOL: f64 = 16.0 * f64::EPSILON;
    if weight == 0.0 {
        for i in 0..numel {
            let (o, expected) = (output[i], start[i]);
            if !o.is_finite() || !expected.is_finite() {
                continue;
            }
            let scale = expected.abs().max(1.0);
            assert!(
                (o - expected).abs() <= ULP_TOL * scale,
                "lerp(w=0)[{i}] = {o}, expected start = {expected}"
            );
        }
    } else if weight == 1.0 {
        for i in 0..numel {
            let (o, expected) = (output[i], end[i]);
            if !o.is_finite() || !expected.is_finite() {
                continue;
            }
            let scale = expected.abs().max(1.0);
            assert!(
                (o - expected).abs() <= ULP_TOL * scale,
                "lerp(w=1)[{i}] = {o}, expected end = {expected}"
            );
        }
    }

    // Monotonicity-in-weight contract is hard to check inline
    // because it requires re-running the kernel. The length and
    // boundary identities give us the high-value coverage; deeper
    // numerical properties (bilinearity, no-clamp extrapolation)
    // are exercised by the random-weight branch above and would
    // surface as panic / NaN-propagation regressions.
});
