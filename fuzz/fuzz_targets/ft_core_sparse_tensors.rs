#![no_main]

use ft_core::{DenseI64Tensor, DenseTensor, Device, SparseCOOTensor, SparseCSRTensor};
use libfuzzer_sys::fuzz_target;

const MAX_SPARSE_INPUT_BYTES: usize = 4096;
const MAX_NNZ: usize = 8;
const MAX_RANK: usize = 4;
const MAX_DIM: usize = 4;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > MAX_SPARSE_INPUT_BYTES {
        return;
    }

    match data[0] % 3 {
        0 => fuzz_coo_scalar(data),
        1 => fuzz_coo_hybrid(data),
        _ => fuzz_csr(data),
    }
});

fn byte(data: &[u8], index: usize) -> u8 {
    data.get(index).copied().unwrap_or(0)
}

fn bounded_dim(data: &[u8], index: usize) -> usize {
    usize::from(byte(data, index) % (MAX_DIM as u8 + 1))
}

fn positive_dim(data: &[u8], index: usize) -> usize {
    usize::from(byte(data, index) % MAX_DIM as u8) + 1
}

fn shape_product(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn index_value(data: &[u8], index: usize, dim: usize) -> i64 {
    let span = i64::try_from(dim.saturating_add(3)).unwrap_or(i64::MAX);
    i64::from(byte(data, index)) % span - 1
}

fn value_at(data: &[u8], index: usize) -> f64 {
    f64::from(byte(data, index)) / 8.0 - 16.0
}

fn fuzz_coo_scalar(data: &[u8]) {
    let rank = usize::from(byte(data, 1) % MAX_RANK as u8) + 1;
    let dense_shape = (0..rank)
        .map(|dim| bounded_dim(data, 2 + dim))
        .collect::<Vec<_>>();
    let nnz = usize::from(byte(data, 8) % (MAX_NNZ as u8 + 1));
    let sparse_dim = rank;
    let indices_data = (0..sparse_dim * nnz)
        .map(|offset| {
            let dim = offset / nnz.max(1);
            index_value(data, 9 + offset, dense_shape[dim])
        })
        .collect::<Vec<_>>();
    let values = (0..nnz)
        .map(|offset| value_at(data, 32 + offset))
        .collect::<Vec<_>>();

    let Ok(indices) =
        DenseI64Tensor::from_contiguous_values(indices_data, vec![sparse_dim, nnz], Device::Cpu)
    else {
        return;
    };
    let Ok(values) = DenseTensor::from_contiguous_values(values, vec![nnz], Device::Cpu) else {
        return;
    };

    if let Ok(sparse) = SparseCOOTensor::new(indices, values, dense_shape.clone(), false) {
        let dense = sparse.to_dense().expect("validated COO should densify");
        assert_eq!(dense.meta().shape(), dense_shape.as_slice());
        assert_eq!(
            dense
                .contiguous_values_as_f64()
                .expect("dense values")
                .len(),
            shape_product(&dense_shape)
        );
    }
}

fn fuzz_coo_hybrid(data: &[u8]) {
    let rank = 2 + usize::from(byte(data, 1) % (MAX_RANK as u8 - 1));
    let sparse_dim = 1 + usize::from(byte(data, 2) % rank as u8);
    let dense_shape = (0..rank)
        .map(|dim| positive_dim(data, 3 + dim))
        .collect::<Vec<_>>();
    let nnz = usize::from(byte(data, 10) % (MAX_NNZ as u8 + 1));
    let dense_tail = &dense_shape[sparse_dim..];
    let dense_tail_numel = shape_product(dense_tail);
    let indices_data = (0..sparse_dim * nnz)
        .map(|offset| {
            let dim = offset / nnz.max(1);
            index_value(data, 11 + offset, dense_shape[dim])
        })
        .collect::<Vec<_>>();
    let values_len = nnz * dense_tail_numel;
    let values = (0..values_len)
        .map(|offset| value_at(data, 48 + offset))
        .collect::<Vec<_>>();
    let mut values_shape = Vec::with_capacity(1 + dense_tail.len());
    values_shape.push(nnz);
    values_shape.extend_from_slice(dense_tail);

    let Ok(indices) =
        DenseI64Tensor::from_contiguous_values(indices_data, vec![sparse_dim, nnz], Device::Cpu)
    else {
        return;
    };
    let Ok(values) = DenseTensor::from_contiguous_values(values, values_shape, Device::Cpu) else {
        return;
    };

    if let Ok(sparse) = SparseCOOTensor::new(indices, values, dense_shape.clone(), false) {
        let dense = sparse
            .to_dense()
            .expect("validated hybrid COO should densify");
        assert_eq!(dense.meta().shape(), dense_shape.as_slice());
        assert_eq!(
            dense
                .contiguous_values_as_f64()
                .expect("dense values")
                .len(),
            shape_product(&dense_shape)
        );
    }
}

fn fuzz_csr(data: &[u8]) {
    let nrows = bounded_dim(data, 1);
    let ncols = bounded_dim(data, 2);
    let valid_layout = byte(data, 3) & 1 == 0;
    let mut crow = Vec::with_capacity(nrows + 1);
    let mut nnz = usize::from(byte(data, 4) % (MAX_NNZ as u8 + 1));

    if valid_layout {
        crow.push(0);
        let mut remaining = nnz;
        let mut used = 0usize;
        for row in 0..nrows {
            let rows_left = nrows.saturating_sub(row);
            let take = if rows_left == 0 {
                0
            } else {
                usize::from(byte(data, 5 + row)) % (remaining + 1)
            };
            used += take;
            remaining = remaining.saturating_sub(take);
            crow.push(i64::try_from(used).unwrap_or(i64::MAX));
        }
        nnz = used;
    } else {
        for offset in 0..=nrows {
            crow.push(i64::from(byte(data, 5 + offset) % (MAX_NNZ as u8 + 3)) - 1);
        }
    }

    let col_indices = (0..nnz)
        .map(|offset| index_value(data, 16 + offset, ncols))
        .collect::<Vec<_>>();
    let values = (0..nnz)
        .map(|offset| value_at(data, 32 + offset))
        .collect::<Vec<_>>();

    let Ok(crow_indices) =
        DenseI64Tensor::from_contiguous_values(crow, vec![nrows + 1], Device::Cpu)
    else {
        return;
    };
    let Ok(col_indices) =
        DenseI64Tensor::from_contiguous_values(col_indices, vec![nnz], Device::Cpu)
    else {
        return;
    };
    let Ok(values) = DenseTensor::from_contiguous_values(values, vec![nnz], Device::Cpu) else {
        return;
    };

    if let Ok(sparse) = SparseCSRTensor::new(crow_indices, col_indices, values, [nrows, ncols]) {
        let dense = sparse.to_dense().expect("validated CSR should densify");
        assert_eq!(dense.meta().shape(), &[nrows, ncols]);
        assert_eq!(
            dense
                .contiguous_values_as_f64()
                .expect("dense values")
                .len(),
            nrows * ncols
        );
    }
}
