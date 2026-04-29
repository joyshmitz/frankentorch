#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_RAPTORQ_PAYLOAD_BYTES: usize = 2 * 1024;

fuzz_target!(|data: &[u8]| {
    let repair_control = data.first().copied().unwrap_or(0);
    let payload_bytes = data;
    if payload_bytes.len() > MAX_RAPTORQ_PAYLOAD_BYTES {
        return;
    }
    let Ok(payload) = std::str::from_utf8(payload_bytes) else {
        return;
    };

    let requested_repairs = usize::from(repair_control % 16);
    if let Ok((sidecar, proof)) = ft_serialize::generate_raptorq_sidecar(payload, requested_repairs)
    {
        assert_eq!(
            sidecar.schema_version,
            ft_serialize::RAPTORQ_SIDECAR_SCHEMA_VERSION
        );
        assert_eq!(sidecar.source_hash, proof.source_hash);
        assert!(sidecar.symbol_size > 0);
        assert!(sidecar.source_symbol_count > 0);
        assert_eq!(sidecar.repair_symbol_count, sidecar.repair_manifest.len());
        assert!(sidecar.repair_symbol_count >= requested_repairs.max(1));
        assert_eq!(proof.recovered_bytes, payload.len());
        assert!(proof.received_symbol_count >= sidecar.source_symbol_count);

        let (sidecar_repeat, proof_repeat) =
            ft_serialize::generate_raptorq_sidecar(payload, requested_repairs)
                .expect("successful RaptorQ sidecar generation must be deterministic");
        assert_eq!(sidecar, sidecar_repeat);
        assert_eq!(proof, proof_repeat);
    }
});
