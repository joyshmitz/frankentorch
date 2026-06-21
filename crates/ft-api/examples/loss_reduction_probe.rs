//! Differential probe for cross_entropy / nll_loss reduction + weight +
//! ignore_index semantics vs torch.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let p = |s: &mut FrankenTorchSession, name: &str, id| {
        println!("{name}: {:?}", s.tensor_values(id).unwrap());
    };
    let inp = s
        .tensor_variable(
            vec![2.0, 1.0, 0.1, 0.5, 2.5, 0.3, 1.2, 0.7, 2.1, 0.9, 1.1, 0.4],
            vec![4, 3],
            false,
        )
        .unwrap();
    let tgt = s
        .tensor_variable(vec![0.0, 1.0, 2.0, 1.0], vec![4], false)
        .unwrap();
    let w = s
        .tensor_variable(vec![1.0, 2.0, 0.5], vec![3], false)
        .unwrap();

    let r = s
        .tensor_cross_entropy_full(inp, tgt, None, None, "mean")
        .unwrap();
    p(&mut s, "ce_mean", r);
    let r = s
        .tensor_cross_entropy_full(inp, tgt, None, None, "sum")
        .unwrap();
    p(&mut s, "ce_sum", r);
    let r = s
        .tensor_cross_entropy_full(inp, tgt, None, None, "none")
        .unwrap();
    p(&mut s, "ce_none", r);
    let r = s
        .tensor_cross_entropy_full(inp, tgt, Some(w), None, "mean")
        .unwrap();
    p(&mut s, "ce_weight_mean", r);
    let r = s
        .tensor_cross_entropy_full(inp, tgt, Some(w), None, "sum")
        .unwrap();
    p(&mut s, "ce_weight_sum", r);
    let r = s
        .tensor_cross_entropy_full(inp, tgt, None, Some(1), "mean")
        .unwrap();
    p(&mut s, "ce_ignore1_mean", r);
    let r = s
        .tensor_cross_entropy_full(inp, tgt, Some(w), Some(1), "mean")
        .unwrap();
    p(&mut s, "ce_ignore1_weight_mean", r);

    let lp = s.tensor_log_softmax(inp, 1).unwrap();
    let r = s
        .tensor_nll_loss_full(lp, tgt, Some(w), None, "mean")
        .unwrap();
    p(&mut s, "nll_weight_mean", r);
    let r = s
        .tensor_nll_loss_full(lp, tgt, None, Some(2), "mean")
        .unwrap();
    p(&mut s, "nll_ignore2_mean", r);
}
