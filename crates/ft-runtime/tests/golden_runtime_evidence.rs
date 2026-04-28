#![forbid(unsafe_code)]

use ft_core::ExecutionMode;
use ft_runtime::{
    DecodeProof, DurabilityEnvelope, EvidenceEntry, EvidenceKind, RuntimeContext, ScrubStatus,
};

fn render_ledger_entries(entries: &[EvidenceEntry]) -> String {
    entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            format!(
                "{index}: kind={:?}; summary={:?}",
                entry.kind, entry.summary
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_decode_proofs(proofs: &[DecodeProof]) -> String {
    proofs
        .iter()
        .enumerate()
        .map(|(index, proof)| {
            format!(
                "{index}: reason={:?}; proof_hash={:?}",
                proof.reason, proof.proof_hash
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn runtime_context_evidence_ledger_snapshot() {
    let mut ctx = RuntimeContext::new(ExecutionMode::Strict);
    ctx.ledger_mut().record(
        EvidenceKind::Dispatch,
        "schema route op=matmul dtype=F32 device=Cpu",
    );
    ctx.set_mode(ExecutionMode::Hardened);
    ctx.ledger_mut().record(
        EvidenceKind::Backward,
        "backward graph nodes=4 edges=3 deterministic=true",
    );

    let rendered = render_ledger_entries(ctx.ledger().entries());

    insta::assert_snapshot!("runtime_context_evidence_ledger", rendered);
}

#[test]
fn durability_envelope_decode_proof_snapshot() {
    let mut envelope = DurabilityEnvelope::new(
        "phase2c-runtime-ledger-v1",
        "runtime_evidence_ledger",
        "blake3:ledger-source",
        64,
        8,
        0.125,
    );
    envelope.mark_scrub_status(ScrubStatus::Recovered);
    envelope.add_decode_proof("single-symbol repair", "blake3:proof-a");

    let rendered = format!(
        "\
artifact_id={:?}
artifact_type={:?}
source_hash={:?}
k={}
repair_symbols={}
overhead_ratio={:.3}
scrub_status={:?}
decode_proofs:
{}",
        envelope.artifact_id,
        envelope.artifact_type,
        envelope.source_hash,
        envelope.k,
        envelope.repair_symbols,
        envelope.overhead_ratio,
        envelope.scrub_status,
        render_decode_proofs(&envelope.decode_proofs)
    );

    insta::assert_snapshot!("durability_envelope_decode_proof", rendered);
}
