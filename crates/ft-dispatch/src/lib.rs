#![forbid(unsafe_code)]

use std::fmt;

use ft_core::{Device, ExecutionMode, ScalarTensor};
use ft_kernel_cpu::{KernelError, add_scalar, mul_scalar};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DispatchKey {
    Undefined = 0,
    BackendSelect = 1,
    CompositeImplicitAutograd = 2,
    CompositeExplicitAutograd = 3,
    CPU = 4,
    AutogradCPU = 5,
}

impl DispatchKey {
    #[must_use]
    pub const fn all() -> &'static [DispatchKey] {
        &[
            DispatchKey::BackendSelect,
            DispatchKey::CompositeImplicitAutograd,
            DispatchKey::CompositeExplicitAutograd,
            DispatchKey::CPU,
            DispatchKey::AutogradCPU,
        ]
    }

    #[must_use]
    pub const fn bit(self) -> u64 {
        1u64 << (self as u8)
    }
}

const TYPE_PRIORITY: [DispatchKey; 5] = [
    DispatchKey::AutogradCPU,
    DispatchKey::CompositeExplicitAutograd,
    DispatchKey::CompositeImplicitAutograd,
    DispatchKey::CPU,
    DispatchKey::BackendSelect,
];

const BACKEND_PRIORITY: [DispatchKey; 1] = [DispatchKey::CPU];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DispatchKeySet {
    bits: u64,
}

impl DispatchKeySet {
    #[must_use]
    pub const fn empty() -> Self {
        Self { bits: 0 }
    }

    #[must_use]
    pub fn from_keys(keys: &[DispatchKey]) -> Self {
        let mut out = Self::empty();
        for key in keys {
            out.add(*key);
        }
        out
    }

    pub fn from_bits_checked(bits: u64) -> Result<Self, DispatchKeyError> {
        let known_mask = DispatchKey::all()
            .iter()
            .fold(0u64, |mask, key| mask | key.bit());
        let unknown = bits & !known_mask;
        if unknown != 0 {
            return Err(DispatchKeyError::UnknownBits {
                unknown_mask: unknown,
            });
        }
        Ok(Self { bits })
    }

    #[must_use]
    pub const fn bits(self) -> u64 {
        self.bits
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    pub fn add(&mut self, key: DispatchKey) {
        self.bits |= key.bit();
    }

    pub fn remove(&mut self, key: DispatchKey) {
        self.bits &= !key.bit();
    }

    #[must_use]
    pub const fn has(self, key: DispatchKey) -> bool {
        (self.bits & key.bit()) != 0
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self {
            bits: self.bits & other.bits,
        }
    }

    pub fn highest_priority_type_id(self) -> Result<DispatchKey, DispatchKeyError> {
        if self.is_empty() {
            return Err(DispatchKeyError::EmptySet);
        }
        TYPE_PRIORITY
            .iter()
            .find(|&&key| self.has(key))
            .copied()
            .ok_or(DispatchKeyError::NoTypeKey)
    }

    pub fn highest_priority_backend_type_id(self) -> Result<DispatchKey, DispatchKeyError> {
        if self.is_empty() {
            return Err(DispatchKeyError::EmptySet);
        }
        BACKEND_PRIORITY
            .iter()
            .find(|&&key| self.has(key))
            .copied()
            .ok_or(DispatchKeyError::NoBackendKey)
    }

    pub fn validate_for_scalar_binary(self) -> Result<(), DispatchKeyError> {
        if self.is_empty() {
            return Err(DispatchKeyError::EmptySet);
        }
        if self.has(DispatchKey::AutogradCPU) && !self.has(DispatchKey::CPU) {
            return Err(DispatchKeyError::IncompatibleSet {
                reason: "AutogradCPU requires CPU backend availability",
            });
        }
        self.highest_priority_type_id()?;
        self.highest_priority_backend_type_id()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchKeyError {
    EmptySet,
    NoTypeKey,
    NoBackendKey,
    UnknownBits { unknown_mask: u64 },
    IncompatibleSet { reason: &'static str },
}

impl fmt::Display for DispatchKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySet => write!(f, "dispatch keyset is empty"),
            Self::NoTypeKey => write!(f, "dispatch keyset has no resolvable type key"),
            Self::NoBackendKey => write!(f, "dispatch keyset has no backend key"),
            Self::UnknownBits { unknown_mask } => {
                write!(
                    f,
                    "dispatch keyset has unknown bitmask 0x{unknown_mask:016x}"
                )
            }
            Self::IncompatibleSet { reason } => {
                write!(f, "incompatible dispatch keyset: {reason}")
            }
        }
    }
}

impl std::error::Error for DispatchKeyError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchDecision {
    pub op: BinaryOp,
    pub mode: ExecutionMode,
    pub kernel: &'static str,
    pub selected_key: DispatchKey,
    pub backend_key: DispatchKey,
    pub keyset_bits: u64,
    pub fallback_used: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DispatchOutcome {
    pub tensor: ScalarTensor,
    pub decision: DispatchDecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchError {
    Kernel(KernelError),
    Key(DispatchKeyError),
}

impl fmt::Display for DispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(error) => write!(f, "kernel dispatch failure: {error}"),
            Self::Key(error) => write!(f, "dispatch key failure: {error}"),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<KernelError> for DispatchError {
    fn from(value: KernelError) -> Self {
        Self::Kernel(value)
    }
}

impl From<DispatchKeyError> for DispatchError {
    fn from(value: DispatchKeyError) -> Self {
        Self::Key(value)
    }
}

#[must_use]
pub fn dispatch_keyset_for_tensors(
    lhs: &ScalarTensor,
    _rhs: &ScalarTensor,
    requires_grad: bool,
) -> DispatchKeySet {
    let mut keyset = DispatchKeySet::empty();
    keyset.add(DispatchKey::BackendSelect);
    if lhs.meta().device() == Device::Cpu {
        keyset.add(DispatchKey::CPU);
    }
    if requires_grad {
        keyset.add(DispatchKey::AutogradCPU);
    }
    keyset
}

pub fn dispatch_scalar_binary(
    op: BinaryOp,
    mode: ExecutionMode,
    lhs: &ScalarTensor,
    rhs: &ScalarTensor,
    requires_grad: bool,
) -> Result<DispatchOutcome, DispatchError> {
    let keyset = dispatch_keyset_for_tensors(lhs, rhs, requires_grad);
    dispatch_scalar_binary_with_keyset(op, mode, lhs, rhs, keyset)
}

pub fn dispatch_scalar_binary_with_keyset(
    op: BinaryOp,
    mode: ExecutionMode,
    lhs: &ScalarTensor,
    rhs: &ScalarTensor,
    keyset: DispatchKeySet,
) -> Result<DispatchOutcome, DispatchError> {
    keyset.validate_for_scalar_binary()?;
    let selected_key = keyset.highest_priority_type_id()?;
    let backend_key = keyset.highest_priority_backend_type_id()?;

    let (effective_key, fallback_used) = match selected_key {
        DispatchKey::AutogradCPU | DispatchKey::CPU => (selected_key, false),
        DispatchKey::CompositeExplicitAutograd
        | DispatchKey::CompositeImplicitAutograd
        | DispatchKey::BackendSelect => match mode {
            ExecutionMode::Strict => {
                return Err(DispatchKeyError::IncompatibleSet {
                    reason: "strict mode forbids composite/backend fallback routing",
                }
                .into());
            }
            ExecutionMode::Hardened => (backend_key, true),
        },
        DispatchKey::Undefined => return Err(DispatchKeyError::NoTypeKey.into()),
    };

    let (tensor, kernel) = match (effective_key, op) {
        (DispatchKey::AutogradCPU, BinaryOp::Add) => {
            (add_scalar(lhs, rhs)?, "autograd_cpu::add_scalar")
        }
        (DispatchKey::AutogradCPU, BinaryOp::Mul) => {
            (mul_scalar(lhs, rhs)?, "autograd_cpu::mul_scalar")
        }
        (DispatchKey::CPU, BinaryOp::Add) => (add_scalar(lhs, rhs)?, "cpu::add_scalar"),
        (DispatchKey::CPU, BinaryOp::Mul) => (mul_scalar(lhs, rhs)?, "cpu::mul_scalar"),
        _ => {
            return Err(DispatchKeyError::IncompatibleSet {
                reason: "resolved dispatch key is unsupported for scalar binary ops",
            }
            .into());
        }
    };

    if effective_key != backend_key && effective_key != DispatchKey::AutogradCPU {
        return Err(DispatchKeyError::IncompatibleSet {
            reason: "resolved key/backend key drifted to incompatible pair",
        }
        .into());
    }

    Ok(DispatchOutcome {
        tensor,
        decision: DispatchDecision {
            op,
            mode,
            kernel,
            selected_key,
            backend_key,
            keyset_bits: keyset.bits(),
            fallback_used,
        },
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ft_core::{DType, Device, ExecutionMode, ScalarTensor};
    use proptest::prelude::*;

    use super::{
        BinaryOp, DispatchKey, DispatchKeyError, DispatchKeySet, TYPE_PRIORITY,
        dispatch_keyset_for_tensors, dispatch_scalar_binary, dispatch_scalar_binary_with_keyset,
    };

    fn det_seed(parts: &[u64]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in parts {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn build_property_log(
        test_id: &str,
        mode: &str,
        seed: u64,
        input_digest: u64,
        output_digest: u64,
        reason_code: &str,
    ) -> BTreeMap<String, String> {
        let mut log = BTreeMap::new();
        let scenario_id = format!("dispatch_key/{mode}:{test_id}");
        log.insert("ts_utc".to_string(), "1970-01-01T00:00:00Z".to_string());
        log.insert("suite_id".to_string(), "ft_dispatch_property".to_string());
        log.insert("test_id".to_string(), test_id.to_string());
        log.insert("packet_id".to_string(), "FT-P2C-002".to_string());
        log.insert(
            "fixture_id".to_string(),
            "ft_dispatch_property_generated".to_string(),
        );
        log.insert("scenario_id".to_string(), scenario_id);
        log.insert("mode".to_string(), mode.to_string());
        log.insert("seed".to_string(), seed.to_string());
        log.insert(
            "input_digest".to_string(),
            format!("det64:{input_digest:016x}"),
        );
        log.insert(
            "output_digest".to_string(),
            format!("det64:{output_digest:016x}"),
        );
        log.insert(
            "env_fingerprint".to_string(),
            "det64:ft-dispatch-test".to_string(),
        );
        log.insert(
            "artifact_refs".to_string(),
            "artifacts/phase2c/FT-P2C-002/fixture_manifest.json".to_string(),
        );
        log.insert(
            "replay_command".to_string(),
            format!("cargo test -p ft-dispatch {test_id} -- --nocapture"),
        );
        log.insert("duration_ms".to_string(), "0".to_string());
        log.insert("outcome".to_string(), "pass".to_string());
        log.insert("reason_code".to_string(), reason_code.to_string());
        log
    }

    fn assert_log_contract(log: &BTreeMap<String, String>) {
        for key in [
            "ts_utc",
            "suite_id",
            "test_id",
            "packet_id",
            "fixture_id",
            "scenario_id",
            "mode",
            "seed",
            "input_digest",
            "output_digest",
            "env_fingerprint",
            "artifact_refs",
            "replay_command",
            "duration_ms",
            "outcome",
            "reason_code",
        ] {
            assert!(
                log.contains_key(key),
                "property log missing required key '{key}'"
            );
        }
    }

    #[test]
    fn dispatch_keyset_set_algebra_is_stable() {
        let mut left = DispatchKeySet::from_keys(&[DispatchKey::CPU, DispatchKey::BackendSelect]);
        let right = DispatchKeySet::from_keys(&[DispatchKey::AutogradCPU, DispatchKey::CPU]);

        let union = left.union(right);
        assert!(union.has(DispatchKey::CPU));
        assert!(union.has(DispatchKey::AutogradCPU));
        assert!(union.has(DispatchKey::BackendSelect));

        let intersection = left.intersection(right);
        assert!(intersection.has(DispatchKey::CPU));
        assert!(!intersection.has(DispatchKey::AutogradCPU));

        left.remove(DispatchKey::BackendSelect);
        assert!(!left.has(DispatchKey::BackendSelect));
    }

    #[test]
    fn priority_resolution_prefers_autograd_cpu() {
        let keys = DispatchKeySet::from_keys(&[
            DispatchKey::BackendSelect,
            DispatchKey::CPU,
            DispatchKey::AutogradCPU,
        ]);
        let selected = keys
            .highest_priority_type_id()
            .expect("priority resolution should succeed");
        assert_eq!(selected, DispatchKey::AutogradCPU);
    }

    #[test]
    fn backend_priority_returns_cpu() {
        let keys = DispatchKeySet::from_keys(&[DispatchKey::BackendSelect, DispatchKey::CPU]);
        let backend = keys
            .highest_priority_backend_type_id()
            .expect("backend priority should resolve");
        assert_eq!(backend, DispatchKey::CPU);
    }

    #[test]
    fn unknown_bits_fail_closed() {
        let err =
            DispatchKeySet::from_bits_checked(1u64 << 63).expect_err("unknown bits must fail");
        let msg = err.to_string();
        assert!(msg.contains("unknown bitmask"));
    }

    #[test]
    fn known_bits_parse_successfully() {
        let known_bits = DispatchKey::BackendSelect.bit()
            | DispatchKey::CompositeImplicitAutograd.bit()
            | DispatchKey::CompositeExplicitAutograd.bit()
            | DispatchKey::CPU.bit()
            | DispatchKey::AutogradCPU.bit();
        let parsed = DispatchKeySet::from_bits_checked(known_bits)
            .expect("known dispatch key bits must parse successfully");
        assert_eq!(parsed.bits(), known_bits);
    }

    #[test]
    fn dispatch_keyset_for_tensors_tracks_requires_grad() {
        let lhs = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);

        let no_grad = dispatch_keyset_for_tensors(&lhs, &rhs, false);
        assert!(no_grad.has(DispatchKey::CPU));
        assert!(no_grad.has(DispatchKey::BackendSelect));
        assert!(!no_grad.has(DispatchKey::AutogradCPU));

        let with_grad = dispatch_keyset_for_tensors(&lhs, &rhs, true);
        assert!(with_grad.has(DispatchKey::CPU));
        assert!(with_grad.has(DispatchKey::BackendSelect));
        assert!(with_grad.has(DispatchKey::AutogradCPU));
    }

    #[test]
    fn validate_requires_cpu_for_autograd() {
        let keyset = DispatchKeySet::from_keys(&[DispatchKey::AutogradCPU]);
        let err = keyset
            .validate_for_scalar_binary()
            .expect_err("autograd without cpu must fail");
        assert!(matches!(err, DispatchKeyError::IncompatibleSet { .. }));
    }

    #[test]
    fn strict_mode_rejects_composite_fallback() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let keyset = DispatchKeySet::from_keys(&[
            DispatchKey::CompositeExplicitAutograd,
            DispatchKey::CPU,
            DispatchKey::BackendSelect,
        ]);

        let err = dispatch_scalar_binary_with_keyset(
            BinaryOp::Add,
            ExecutionMode::Strict,
            &lhs,
            &rhs,
            keyset,
        )
        .expect_err("strict mode must fail closed");
        assert!(err.to_string().contains("strict mode forbids"));
    }

    #[test]
    fn strict_mode_prefers_cpu_over_backendselect() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let keyset = DispatchKeySet::from_keys(&[DispatchKey::BackendSelect, DispatchKey::CPU]);

        let out = dispatch_scalar_binary_with_keyset(
            BinaryOp::Add,
            ExecutionMode::Strict,
            &lhs,
            &rhs,
            keyset,
        )
        .expect("strict mode should resolve directly to cpu when cpu type key exists");
        assert_eq!(out.tensor.value(), 5.0);
        assert_eq!(out.decision.selected_key, DispatchKey::CPU);
        assert_eq!(out.decision.backend_key, DispatchKey::CPU);
        assert!(!out.decision.fallback_used);
        assert_eq!(out.decision.kernel, "cpu::add_scalar");
    }

    #[test]
    fn hardened_mode_allows_composite_fallback() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let keyset = DispatchKeySet::from_keys(&[
            DispatchKey::CompositeExplicitAutograd,
            DispatchKey::CPU,
            DispatchKey::BackendSelect,
        ]);

        let out = dispatch_scalar_binary_with_keyset(
            BinaryOp::Add,
            ExecutionMode::Hardened,
            &lhs,
            &rhs,
            keyset,
        )
        .expect("hardened mode should fallback");
        assert_eq!(out.tensor.value(), 5.0);
        assert!(out.decision.fallback_used);
        assert_eq!(
            out.decision.selected_key,
            DispatchKey::CompositeExplicitAutograd
        );
        assert_eq!(out.decision.backend_key, DispatchKey::CPU);
    }

    #[test]
    fn hardened_mode_prefers_cpu_over_backendselect() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F64, Device::Cpu);
        let keyset = DispatchKeySet::from_keys(&[DispatchKey::BackendSelect, DispatchKey::CPU]);

        let out = dispatch_scalar_binary_with_keyset(
            BinaryOp::Mul,
            ExecutionMode::Hardened,
            &lhs,
            &rhs,
            keyset,
        )
        .expect("hardened mode should resolve directly to cpu when cpu type key exists");

        assert_eq!(out.tensor.value(), 8.0);
        assert!(!out.decision.fallback_used);
        assert_eq!(out.decision.selected_key, DispatchKey::CPU);
        assert_eq!(out.decision.backend_key, DispatchKey::CPU);
        assert_eq!(out.decision.kernel, "cpu::mul_scalar");
    }

    #[test]
    fn dispatch_returns_kernel_metadata() {
        let lhs = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let outcome =
            dispatch_scalar_binary(BinaryOp::Add, ExecutionMode::Strict, &lhs, &rhs, true)
                .expect("dispatch should succeed");

        assert_eq!(outcome.tensor.value(), 3.0);
        assert_eq!(outcome.decision.kernel, "autograd_cpu::add_scalar");
        assert_eq!(outcome.decision.mode, ExecutionMode::Strict);
        assert_eq!(outcome.decision.selected_key, DispatchKey::AutogradCPU);
        assert_eq!(outcome.decision.backend_key, DispatchKey::CPU);
        assert!(!outcome.decision.fallback_used);
    }

    #[test]
    fn property_log_contract_maps_to_dispatch_scenarios() {
        let seed = det_seed(&[0x13, 0x5, 0x2]);
        let log = build_property_log(
            "prop_contract_mapping",
            "strict",
            seed,
            0xdead_beef,
            0xbead_1337,
            "dispatch_property_contract_mapping_ok",
        );
        assert_log_contract(&log);

        let scenario_id = log
            .get("scenario_id")
            .expect("scenario_id must be present in property logs");
        assert!(scenario_id.starts_with("dispatch_key/strict:"));
        assert!(
            scenario_id.contains("prop_contract_mapping"),
            "scenario id should preserve test identity"
        );

        let replay = log
            .get("replay_command")
            .expect("replay command must be present in property logs");
        assert!(replay.contains("cargo test -p ft-dispatch"));
        assert!(replay.contains("prop_contract_mapping"));
    }

    proptest! {
        #[test]
        fn prop_known_bits_roundtrip(
            backend_select in any::<bool>(),
            composite_implicit in any::<bool>(),
            composite_explicit in any::<bool>(),
            cpu in any::<bool>(),
            autograd_cpu in any::<bool>(),
        ) {
            let mut keyset = DispatchKeySet::empty();
            if backend_select {
                keyset.add(DispatchKey::BackendSelect);
            }
            if composite_implicit {
                keyset.add(DispatchKey::CompositeImplicitAutograd);
            }
            if composite_explicit {
                keyset.add(DispatchKey::CompositeExplicitAutograd);
            }
            if cpu {
                keyset.add(DispatchKey::CPU);
            }
            if autograd_cpu {
                keyset.add(DispatchKey::AutogradCPU);
            }

            let bits = keyset.bits();
            let reparsed = DispatchKeySet::from_bits_checked(bits)
                .expect("known bit combinations must parse");
            prop_assert_eq!(reparsed.bits(), bits);

            let seed = det_seed(&[bits, backend_select as u64, cpu as u64, autograd_cpu as u64]);
            let log = build_property_log(
                "prop_known_bits_roundtrip",
                "strict",
                seed,
                bits,
                reparsed.bits(),
                "dispatch_known_bits_roundtrip_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_unknown_bits_mask_fail_closed(bit in prop_oneof![Just(0u8), 6u8..=63u8]) {
            let mask = 1u64 << u32::from(bit);
            let err = DispatchKeySet::from_bits_checked(mask)
                .expect_err("unknown bit masks must fail closed");
            match err {
                DispatchKeyError::UnknownBits { unknown_mask } => {
                    prop_assert_eq!(unknown_mask, mask);
                }
                other => prop_assert!(false, "expected UnknownBits, got {other:?}"),
            }

            let seed = det_seed(&[mask, bit as u64]);
            let log = build_property_log(
                "prop_unknown_bits_mask_fail_closed",
                "strict",
                seed,
                mask,
                mask,
                "dispatch_unknown_bits_fail_closed",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_priority_matches_explicit_table(
            backend_select in any::<bool>(),
            composite_implicit in any::<bool>(),
            composite_explicit in any::<bool>(),
            cpu in any::<bool>(),
            autograd_cpu in any::<bool>(),
        ) {
            let mut keyset = DispatchKeySet::empty();
            if backend_select {
                keyset.add(DispatchKey::BackendSelect);
            }
            if composite_implicit {
                keyset.add(DispatchKey::CompositeImplicitAutograd);
            }
            if composite_explicit {
                keyset.add(DispatchKey::CompositeExplicitAutograd);
            }
            if cpu {
                keyset.add(DispatchKey::CPU);
            }
            if autograd_cpu {
                keyset.add(DispatchKey::AutogradCPU);
            }

            let result = keyset.highest_priority_type_id();
            if keyset.is_empty() {
                prop_assert!(matches!(result, Err(DispatchKeyError::EmptySet)));
            } else {
                let expected = TYPE_PRIORITY
                    .iter()
                    .copied()
                    .find(|key| keyset.has(*key))
                    .expect("non-empty keyset should have a type key");
                prop_assert_eq!(result.expect("type key should resolve"), expected);
            }

            let bits = keyset.bits();
            let output = result.map_or(0u64, |key| key as u8 as u64);
            let seed = det_seed(&[bits, output]);
            let log = build_property_log(
                "prop_priority_matches_explicit_table",
                "strict",
                seed,
                bits,
                output,
                "dispatch_type_priority_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_backend_resolution_requires_cpu(
            backend_select in any::<bool>(),
            composite_implicit in any::<bool>(),
            composite_explicit in any::<bool>(),
            cpu in any::<bool>(),
            autograd_cpu in any::<bool>(),
        ) {
            let mut keyset = DispatchKeySet::empty();
            if backend_select {
                keyset.add(DispatchKey::BackendSelect);
            }
            if composite_implicit {
                keyset.add(DispatchKey::CompositeImplicitAutograd);
            }
            if composite_explicit {
                keyset.add(DispatchKey::CompositeExplicitAutograd);
            }
            if cpu {
                keyset.add(DispatchKey::CPU);
            }
            if autograd_cpu {
                keyset.add(DispatchKey::AutogradCPU);
            }

            let result = keyset.highest_priority_backend_type_id();
            if keyset.is_empty() {
                prop_assert!(matches!(result, Err(DispatchKeyError::EmptySet)));
            } else if cpu {
                prop_assert_eq!(result.expect("cpu backend should resolve"), DispatchKey::CPU);
            } else {
                prop_assert!(matches!(result, Err(DispatchKeyError::NoBackendKey)));
            }

            let bits = keyset.bits();
            let output = result.map_or(0u64, |key| key as u8 as u64);
            let seed = det_seed(&[bits, output, cpu as u64]);
            let log = build_property_log(
                "prop_backend_resolution_requires_cpu",
                "strict",
                seed,
                bits,
                output,
                "dispatch_backend_resolution_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_validate_requires_cpu_for_autograd(
            backend_select in any::<bool>(),
            composite_implicit in any::<bool>(),
            composite_explicit in any::<bool>(),
            cpu in any::<bool>(),
            autograd_cpu in any::<bool>(),
        ) {
            let mut keyset = DispatchKeySet::empty();
            if backend_select {
                keyset.add(DispatchKey::BackendSelect);
            }
            if composite_implicit {
                keyset.add(DispatchKey::CompositeImplicitAutograd);
            }
            if composite_explicit {
                keyset.add(DispatchKey::CompositeExplicitAutograd);
            }
            if cpu {
                keyset.add(DispatchKey::CPU);
            }
            if autograd_cpu {
                keyset.add(DispatchKey::AutogradCPU);
            }

            let validation = keyset.validate_for_scalar_binary();
            if keyset.is_empty() {
                prop_assert!(matches!(validation, Err(DispatchKeyError::EmptySet)));
            } else if autograd_cpu && !cpu {
                match validation {
                    Err(DispatchKeyError::IncompatibleSet { .. }) => {}
                    other => prop_assert!(false, "expected IncompatibleSet, got {other:?}"),
                }
            } else if !cpu {
                prop_assert!(matches!(validation, Err(DispatchKeyError::NoBackendKey)));
            } else {
                prop_assert!(validation.is_ok());
            }

            let bits = keyset.bits();
            let outcome = if validation.is_ok() { 1u64 } else { 0u64 };
            let seed = det_seed(&[bits, outcome, autograd_cpu as u64]);
            let log = build_property_log(
                "prop_validate_requires_cpu_for_autograd",
                "strict",
                seed,
                bits,
                outcome,
                "dispatch_validate_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_mode_split_for_composite_keysets(
            lhs_value in -1_000.0f64..1_000.0f64,
            rhs_value in -1_000.0f64..1_000.0f64,
            use_explicit in any::<bool>(),
        ) {
            let lhs = ScalarTensor::new(lhs_value, DType::F64, Device::Cpu);
            let rhs = ScalarTensor::new(rhs_value, DType::F64, Device::Cpu);
            let selected_key = if use_explicit {
                DispatchKey::CompositeExplicitAutograd
            } else {
                DispatchKey::CompositeImplicitAutograd
            };
            let keyset =
                DispatchKeySet::from_keys(&[selected_key, DispatchKey::CPU, DispatchKey::BackendSelect]);

            let strict_err = dispatch_scalar_binary_with_keyset(
                BinaryOp::Add,
                ExecutionMode::Strict,
                &lhs,
                &rhs,
                keyset,
            )
            .expect_err("strict mode must reject composite/backend fallback");
            prop_assert!(strict_err.to_string().contains("strict mode forbids"));

            let hardened_out = dispatch_scalar_binary_with_keyset(
                BinaryOp::Add,
                ExecutionMode::Hardened,
                &lhs,
                &rhs,
                keyset,
            )
            .expect("hardened mode should allow bounded fallback");

            prop_assert!(hardened_out.decision.fallback_used);
            prop_assert_eq!(hardened_out.decision.selected_key, selected_key);
            prop_assert_eq!(hardened_out.decision.backend_key, DispatchKey::CPU);
            prop_assert!((hardened_out.tensor.value() - (lhs_value + rhs_value)).abs() <= 1e-12);

            let seed = det_seed(&[
                lhs_value.to_bits(),
                rhs_value.to_bits(),
                use_explicit as u64,
            ]);
            let log = build_property_log(
                "prop_mode_split_for_composite_keysets",
                "hardened",
                seed,
                lhs.evidence_fingerprint64() ^ rhs.evidence_fingerprint64(),
                hardened_out.decision.keyset_bits,
                "dispatch_mode_split_contract_ok",
            );
            assert_log_contract(&log);
        }
    }
}
