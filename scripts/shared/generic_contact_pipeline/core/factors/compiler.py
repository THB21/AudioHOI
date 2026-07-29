from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping

from .activation import FactorActivationLedger
from .types import FactorKind, FactorSpec


@dataclass(frozen=True)
class FactorRuntimeConfig:
    """Numeric solver configuration bound to a factor capability.

    Object/case adapters may select these values from versioned configuration,
    but the compiler and solver only see factor ids/kinds and numeric units.
    """

    weight: float
    sigma: float | None
    sigma_unit: str | None
    source: str
    state_scales: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("factor runtime weight must be finite and non-negative")
        if self.sigma is not None and (not isfinite(self.sigma) or self.sigma <= 0.0):
            raise ValueError("factor runtime sigma must be finite and positive")
        if (self.sigma is None) != (self.sigma_unit is None):
            raise ValueError("factor runtime sigma and sigma_unit must be provided together")
        if not self.source:
            raise ValueError("factor runtime configuration requires provenance")
        if self.state_scales is not None and (
            not self.state_scales
            or any(not isfinite(value) or value <= 0.0 for value in self.state_scales)
        ):
            raise ValueError("factor runtime state_scales must be finite and positive")


@dataclass(frozen=True)
class CompiledFactor:
    factor_id: str
    kind: FactorKind
    residual_fn_ref: str
    robust_loss: str
    base_weight_source: str
    active_frames: int
    downweighted_frames: int
    inactive_frames: int
    input_ids: tuple[str, ...]
    gate_provenance: tuple[str, ...]
    runtime_config: FactorRuntimeConfig | None = None
    consumed_by_solver: bool = False

    def __post_init__(self) -> None:
        if not self.factor_id or not self.residual_fn_ref or not self.robust_loss or not self.base_weight_source:
            raise ValueError("CompiledFactor requires id, residual_fn_ref, robust_loss, and base_weight_source")
        if self.active_frames < 0 or self.downweighted_frames < 0 or self.inactive_frames < 0:
            raise ValueError("CompiledFactor frame counts must be non-negative")
        if not self.input_ids:
            raise ValueError("CompiledFactor requires input_ids")
        if self.consumed_by_solver:
            raise ValueError("CompiledFactor is shadow-only until the runtime executor is promoted")


@dataclass(frozen=True)
class CompiledFactorLedger:
    schema_version: int
    sample_id: str
    compiled_factors: tuple[CompiledFactor, ...]
    by_kind: dict[str, int]
    canonical_sha256: str
    consumed_by_solver: bool = False

    def __post_init__(self) -> None:
        if self.consumed_by_solver:
            raise ValueError("CompiledFactorLedger is shadow-only until the runtime executor is promoted")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _residual_fn_ref(kind: FactorKind) -> str:
    return f"shadow_residual::{kind.value}"


def _robust_loss(kind: FactorKind) -> str:
    if kind in {
        FactorKind.POINT_REPROJECTION,
        FactorKind.LINE_REPROJECTION,
        FactorKind.MASK_SILHOUETTE,
        FactorKind.METRIC_DEPTH,
        FactorKind.CONTACT_DISTANCE,
        FactorKind.SUPPORT_AND_PENETRATION,
    }:
        return "huber_shadow"
    if kind in {FactorKind.TEMPORAL_VELOCITY, FactorKind.TEMPORAL_ACCELERATION, FactorKind.STATIC_FREEZE}:
        return "quadratic_shadow"
    return "identity_shadow"


def _input_ids(factor: FactorSpec) -> tuple[str, ...]:
    return tuple(f"{ref.role}:{ref.source_ir}:{ref.source_id}" for ref in factor.input_refs)


def factor_runtime_configs_from_mapping(
    raw: Mapping[str, object] | None,
    *,
    source: str,
) -> dict[FactorKind, FactorRuntimeConfig]:
    """Parse factor-kind configuration without object identity dispatch."""

    configs: dict[FactorKind, FactorRuntimeConfig] = {}
    for raw_kind, raw_config in (raw or {}).items():
        try:
            kind = FactorKind(str(raw_kind))
        except ValueError as exc:
            raise ValueError(f"unknown factor runtime kind: {raw_kind}") from exc
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"factor runtime config for {kind.value} must be a mapping")
        unknown = set(raw_config) - {"weight", "sigma", "sigma_unit", "source", "state_scales"}
        if unknown:
            raise ValueError(
                f"unknown factor runtime fields for {kind.value}: {','.join(sorted(str(value) for value in unknown))}"
            )
        if "weight" not in raw_config:
            raise ValueError(f"factor runtime config for {kind.value} requires weight")
        config_source = str(raw_config.get("source") or f"{source}:{kind.value}")
        sigma_raw = raw_config.get("sigma")
        unit_raw = raw_config.get("sigma_unit")
        scales_raw = raw_config.get("state_scales")
        if scales_raw is not None and not isinstance(scales_raw, (list, tuple)):
            raise ValueError(f"factor runtime state_scales for {kind.value} must be a sequence")
        configs[kind] = FactorRuntimeConfig(
            weight=float(raw_config["weight"]),
            sigma=None if sigma_raw is None else float(sigma_raw),
            sigma_unit=None if unit_raw is None else str(unit_raw),
            source=config_source,
            state_scales=(
                None
                if scales_raw is None
                else tuple(float(value) for value in scales_raw)
            ),
        )
    return configs


def build_compiled_factor_ledger(
    sample_id: str,
    factors: tuple[FactorSpec, ...],
    activation_ledger: FactorActivationLedger,
    runtime_configs: Mapping[FactorKind, FactorRuntimeConfig] | None = None,
) -> CompiledFactorLedger:
    activation_by_id = {record.factor_id: record for record in activation_ledger.records}
    compiled: list[CompiledFactor] = []
    for factor in factors:
        activation = activation_by_id.get(factor.factor_id)
        if activation is None:
            raise ValueError(f"missing activation record for factor {factor.factor_id}")
        provenance = [
            f"activation_policy:{activation.activation_policy}",
            *[f"gate_axis:{axis}" for axis in activation.gate_provenance],
        ]
        if factor.gate_source:
            provenance.append(f"legacy_gate_source:{factor.gate_source}")
        compiled.append(
            CompiledFactor(
                factor_id=factor.factor_id,
                kind=factor.kind,
                residual_fn_ref=_residual_fn_ref(factor.kind),
                robust_loss=_robust_loss(factor.kind),
                base_weight_source=factor.weight_source,
                active_frames=activation.active_frames,
                downweighted_frames=activation.downweighted_frames,
                inactive_frames=activation.inactive_frames,
                input_ids=_input_ids(factor),
                gate_provenance=tuple(provenance),
                runtime_config=(runtime_configs or {}).get(factor.kind),
                consumed_by_solver=False,
            )
        )
    by_kind: dict[str, int] = {}
    for factor in compiled:
        by_kind[factor.kind.value] = by_kind.get(factor.kind.value, 0) + 1
    payload = {
        "schema_version": 1,
        "sample_id": sample_id,
        "compiled_factors": [compiled_factor_record(factor) for factor in compiled],
        "by_kind": dict(sorted(by_kind.items())),
    }
    return CompiledFactorLedger(
        schema_version=1,
        sample_id=sample_id,
        compiled_factors=tuple(compiled),
        by_kind=dict(sorted(by_kind.items())),
        canonical_sha256=_canonical_hash(payload),
        consumed_by_solver=False,
    )


def compiled_factor_record(factor: CompiledFactor) -> dict[str, object]:
    payload = asdict(factor)
    payload["kind"] = factor.kind.value
    if factor.runtime_config is None:
        payload.pop("runtime_config")
    return payload
