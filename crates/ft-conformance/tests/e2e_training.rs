//! End-to-end training integration tests.
//!
//! These tests exercise the full FrankenTorch stack:
//! ft-data (DataLoader) -> ft-nn (modules) -> ft-api (forward/backward)
//! -> ft-autograd (gradient computation) -> ft-optim (parameter update)
//! -> ft-serialize (save/load state dict).
//!
//! If any component fails to compose correctly, these tests catch it.

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_data::{DataItem, DataLoader, DataLoaderConfig, TensorDataset};
use ft_nn::{Linear, Module};
use ft_optim::{Adam, LRScheduler, Optimizer, StepLR};

/// Generate a synthetic regression dataset: y = 2*x1 + 3*x2 + 1 (with slight noise via seed).
fn make_regression_dataset(n_samples: usize, seed: u64) -> TensorDataset {
    let mut rng_state = seed;
    let mut next_f64 = || -> f64 {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // Map to [-1, 1]
        ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };

    let items: Vec<DataItem> = (0..n_samples)
        .map(|_| {
            let x1 = next_f64();
            let x2 = next_f64();
            let noise = next_f64() * 0.01;
            let y = 2.0 * x1 + 3.0 * x2 + 1.0 + noise;
            DataItem::input_target(vec![x1, x2], vec![2], vec![y], vec![1])
        })
        .collect();
    TensorDataset::new(items)
}

/// Generate a synthetic binary classification dataset.
fn make_classification_dataset(n_samples: usize, seed: u64) -> TensorDataset {
    let mut rng_state = seed;
    let mut next_f64 = || -> f64 {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((rng_state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };

    let items: Vec<DataItem> = (0..n_samples)
        .map(|_| {
            let x1 = next_f64();
            let x2 = next_f64();
            // Class 1 if x1 + x2 > 0, else class 0
            let label = if x1 + x2 > 0.0 { 1.0 } else { 0.0 };
            DataItem::input_target(vec![x1, x2], vec![2], vec![label], vec![1])
        })
        .collect();
    TensorDataset::new(items)
}

#[test]
fn e2e_mlp_regression_training_with_dataloader() {
    // Full stack test: DataLoader -> Linear -> MSE loss -> backward -> Adam -> StepLR
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

    // 1. Build model: Linear(2 -> 8) -> ReLU -> Linear(8 -> 1)
    let l1 = Linear::new(&mut session, 2, 8, true).expect("l1");
    let l2 = Linear::new(&mut session, 8, 1, true).expect("l2");

    // 2. Create optimizer with all model parameters and scheduler
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut optimizer = Adam::new(params, 0.01);
    let mut scheduler = StepLR::new(&optimizer, 3);

    // 3. Create dataset and dataloader
    let dataset = make_regression_dataset(64, 42);

    // 4. Train for 5 epochs
    let mut epoch_losses = Vec::new();
    for epoch in 0..5 {
        let mut loader = DataLoader::new(&dataset, DataLoaderConfig::new(16));
        let mut epoch_loss_sum = 0.0;
        let mut batch_count = 0;

        while let Some(batch) = loader.next_batch(&mut session).expect("batch") {
            let input = batch.input().expect("input");
            let target = batch.target().expect("target");

            // Forward
            let h = l1.forward(&mut session, input).expect("l1_fwd");
            let h = session.tensor_relu(h).expect("relu");
            let pred = l2.forward(&mut session, h).expect("l2_fwd");
            let loss = session.mse_loss(pred, target).expect("mse_loss");
            let loss_val = session.tensor_values(loss).expect("loss_val")[0];

            // Backward
            let report = session.tensor_backward(loss).expect("backward");

            // Optimizer step
            optimizer.step(&mut session, &report).expect("step");

            epoch_loss_sum += loss_val;
            batch_count += 1;
        }

        let avg_loss = epoch_loss_sum / batch_count as f64;
        epoch_losses.push(avg_loss);
        eprintln!(
            "  [e2e_mlp_regression] epoch={epoch} avg_loss={avg_loss:.6} lr={:.6} batches={batch_count}",
            optimizer.get_lr()
        );

        // Step the scheduler
        scheduler.step(&mut optimizer, None);
    }

    // 5. Verify loss decreased overall
    assert!(
        epoch_losses.last().unwrap() < epoch_losses.first().unwrap(),
        "loss should decrease over training: first={:.6} last={:.6}",
        epoch_losses.first().unwrap(),
        epoch_losses.last().unwrap()
    );
    eprintln!(
        "  [e2e_mlp_regression] PASS: loss decreased {:.6} -> {:.6}",
        epoch_losses.first().unwrap(),
        epoch_losses.last().unwrap()
    );
}

#[test]
fn e2e_model_save_load_produces_identical_predictions() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

    // 1. Build and train a simple model briefly
    let linear = Linear::new(&mut session, 2, 1, true).expect("linear");
    let params = linear.parameters();
    let mut optimizer = Adam::new(params, 0.01);

    let input = session
        .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
        .expect("input");
    let target = session
        .tensor_variable(vec![7.0], vec![1, 1], false)
        .expect("target");

    for _ in 0..10 {
        let pred = linear.forward(&mut session, input).expect("forward");
        let loss = session.mse_loss(pred, target).expect("mse");
        let report = session.tensor_backward(loss).expect("backward");
        optimizer.step(&mut session, &report).expect("step");
    }

    // 2. Get prediction before save
    let pred_before = linear.forward(&mut session, input).expect("pred_before");
    let vals_before = session.tensor_values(pred_before).expect("vals_before");

    // 3. Save state dict
    let sd = linear.state_dict(&session).expect("state_dict");
    let path = std::env::temp_dir().join("ft_e2e_save_load_test.ftsv");
    ft_serialize::save_state_dict(&sd, &path).expect("save");

    // 4. Load into a fresh model
    let mut session2 = FrankenTorchSession::new(ExecutionMode::Strict);
    let linear2 = Linear::new(&mut session2, 2, 1, true).expect("linear2");
    let loaded_sd = ft_serialize::load_state_dict(&path).expect("load");
    linear2
        .load_state_dict(&mut session2, &loaded_sd, true)
        .expect("load_state_dict");
    let _ = std::fs::remove_file(&path);

    // 5. Get prediction after load
    let input2 = session2
        .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
        .expect("input2");
    let pred_after = linear2.forward(&mut session2, input2).expect("pred_after");
    let vals_after = session2.tensor_values(pred_after).expect("vals_after");

    // 6. Predictions must be identical
    assert_eq!(
        vals_before.len(),
        vals_after.len(),
        "prediction shapes must match"
    );
    for (i, (a, b)) in vals_before.iter().zip(vals_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "prediction[{i}] mismatch after save/load: before={a} after={b}"
        );
    }
    eprintln!("  [e2e_save_load] PASS: predictions identical after save/load round-trip");
}

#[test]
fn e2e_no_grad_eval_does_not_corrupt_gradient_state() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

    let linear = Linear::new(&mut session, 2, 1, true).expect("linear");
    let params = linear.parameters();
    let mut optimizer = Adam::new(params, 0.01);

    let input = session
        .tensor_variable(vec![1.0, 2.0], vec![1, 2], false)
        .expect("input");
    let target = session
        .tensor_variable(vec![5.0], vec![1, 1], false)
        .expect("target");

    // 1. Train one step
    let pred = linear.forward(&mut session, input).expect("forward");
    let loss = session.mse_loss(pred, target).expect("mse");
    let report = session.tensor_backward(loss).expect("backward");
    optimizer.step(&mut session, &report).expect("step");

    // 2. Evaluate in no_grad context
    assert!(
        session.is_grad_enabled(),
        "grad should be enabled before no_grad"
    );
    let _eval_pred = session.with_no_grad(|s| {
        assert!(
            !s.is_grad_enabled(),
            "grad should be disabled inside no_grad"
        );
        linear.forward(s, input).expect("eval forward")
    });
    assert!(
        session.is_grad_enabled(),
        "grad should be restored after no_grad"
    );

    // 3. Verify we can still train after eval
    let pred2 = linear.forward(&mut session, input).expect("forward2");
    let loss2 = session.mse_loss(pred2, target).expect("mse2");
    let report2 = session.tensor_backward(loss2).expect("backward2");
    optimizer.step(&mut session, &report2).expect("step2");

    let final_pred = linear.forward(&mut session, input).expect("final_forward");
    let final_loss = session.mse_loss(final_pred, target).expect("final_mse");
    let final_loss_val = session.tensor_values(final_loss).expect("final_loss")[0];

    eprintln!(
        "  [e2e_no_grad_eval] PASS: training continued after no_grad eval, final_loss={:.6}",
        final_loss_val
    );
}

#[test]
fn e2e_classification_with_cross_entropy() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

    // Binary classification: Linear(2 -> 2) with cross-entropy loss
    let linear = Linear::new(&mut session, 2, 2, true).expect("linear");
    let params = linear.parameters();
    let mut optimizer = Adam::new(params, 0.01);

    let dataset = make_classification_dataset(100, 123);

    let mut epoch_losses = Vec::new();
    for epoch in 0..10 {
        let mut loader = DataLoader::new(&dataset, DataLoaderConfig::new(20));
        let mut epoch_loss_sum = 0.0;
        let mut batch_count = 0;

        while let Some(batch) = loader.next_batch(&mut session).expect("batch") {
            let input = batch.input().expect("input");
            let target = batch.target().expect("target");

            let logits = linear.forward(&mut session, input).expect("forward");

            // Reshape target for cross_entropy: [batch, 1] -> [batch]
            let target_flat = session.tensor_reshape(target, vec![20]).expect("reshape");

            let loss = session
                .cross_entropy_loss(logits, target_flat)
                .expect("cross_entropy");
            let loss_val = session.tensor_values(loss).expect("loss_val")[0];

            let report = session.tensor_backward(loss).expect("backward");
            optimizer.step(&mut session, &report).expect("step");

            epoch_loss_sum += loss_val;
            batch_count += 1;
        }

        let avg_loss = epoch_loss_sum / batch_count as f64;
        epoch_losses.push(avg_loss);
        if epoch % 3 == 0 {
            eprintln!("  [e2e_classification] epoch={epoch} avg_loss={avg_loss:.6}");
        }
    }

    assert!(
        epoch_losses.last().unwrap() < epoch_losses.first().unwrap(),
        "classification loss should decrease: first={:.6} last={:.6}",
        epoch_losses.first().unwrap(),
        epoch_losses.last().unwrap()
    );
    eprintln!(
        "  [e2e_classification] PASS: loss decreased {:.6} -> {:.6}",
        epoch_losses.first().unwrap(),
        epoch_losses.last().unwrap()
    );
}

#[test]
fn e2e_dataloader_epoch_reset_and_shuffle() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);

    let dataset = make_regression_dataset(20, 99);
    let config = DataLoaderConfig::new(5).with_shuffle(true);
    let mut loader = DataLoader::new(&dataset, config).seed(42);
    loader.reset(); // trigger initial shuffle

    // Epoch 1
    let mut epoch1_values = Vec::new();
    while let Some(batch) = loader.next_batch(&mut session).expect("batch") {
        let target = batch.target().expect("target");
        let vals = session.tensor_values(target).expect("vals");
        epoch1_values.extend(vals);
    }

    // Epoch 2 (reset triggers reshuffle)
    loader.reset();
    let mut epoch2_values = Vec::new();
    while let Some(batch) = loader.next_batch(&mut session).expect("batch") {
        let target = batch.target().expect("target");
        let vals = session.tensor_values(target).expect("vals");
        epoch2_values.extend(vals);
    }

    assert_eq!(epoch1_values.len(), 20);
    assert_eq!(epoch2_values.len(), 20);

    // Both epochs should contain the same set of values (just different order)
    let mut sorted1 = epoch1_values.clone();
    let mut sorted2 = epoch2_values.clone();
    sorted1.sort_by(|a, b| a.total_cmp(b));
    sorted2.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(
        sorted1, sorted2,
        "epochs should have same data, different order"
    );

    // With high probability, the orders should differ
    assert_ne!(
        epoch1_values, epoch2_values,
        "shuffled epochs should have different ordering"
    );
    eprintln!("  [e2e_dataloader_shuffle] PASS: epochs shuffled correctly");
}
