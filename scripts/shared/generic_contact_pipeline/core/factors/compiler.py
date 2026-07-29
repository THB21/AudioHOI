from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping

from .activation import ACTIVATION_STATES, FactorActivationInterval, FactorActivationLedger
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
    activation_weight_tiers: tuple[tuple[str, float], ...] = (
        ("active", 1.0),
        ("downweighted", 1.0),
        ("inactive", 0.0),
    )

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
        tier_names = tuple(name for name, _value in self.activation_weight_tiers)
        if tier_names != ACTIVATION_STATES:
            raise ValueError(f"factor runtime activation tiers must be ordered as {ACTIVATION_STATES}")
        if any(not isfinite(value) or value < 0.0 or value > 1.0 for _name, value in self.activation_weight_tiers):
            raise ValueError("factor runtime activation tier values must be finite within [0, 1]")


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
    activation_intervals: tuple[FactorActivationInterval, ...] = ()
    consumed_by_solver: bool = False

    def __post_init__(self) -> None:
        if not self.factor_id or not self.residual_fn_ref or not self.robust_loss or not self.base_weight_source:
            raise ValueError("CompiledFactor requires id, residual_fn_ref, robust_loss, and base_weight_source")
        if self.active_frames < 0 or self.downweighted_frames < 0 or self.inactive_frames < 0:
            raise ValueError("CompiledFactor frame counts must be non-negative")
        if not self.input_ids:
            raise ValueError("CompiledFactor requires input_ids")
        if self.activation_intervals and self.runtime_config is None:
            raise ValueError("compiled activation intervals require numeric runtime configuration")
        if self.activation_intervals:
            counts = {status: 0 for status in ACTIVATION_STATES}
            for interval in self.activation_intervals:
                counts[interval.status] += interval.end_frame - interval.start_frame + 1
            expected = {
                "active": self.active_frames,
                "downweighted": self.downweighted_frames,
                "inactive": self.inactive_frames,
            }
            if counts != expected:
                raise ValueError("compiled activation intervals do not match factor frame counts")
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
        unknown = set(raw_config) - {
            "weight",
            "sigma",
            "sigma_unit",
            "source",
            "state_scales",
            "activation_weight_tiers",
        }
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
        tiers_raw = raw_config.get("activation_weight_tiers")
        if scales_raw is not None and not isinstance(scales_raw, (list, tuple)):
            raise ValueError(f"factor runtime state_scales for {kind.value} must be a sequence")
        if tiers_raw is not None and not isinstance(tiers_raw, Mapping):
            raise ValueError(f"factor runtime activation_weight_tiers for {kind.value} must be a mapping")
        tier_values = tiers_raw or {"active": 1.0, "downweighted": 1.0, "inactive": 0.0}
        if set(tier_values) != set(ACTIVATION_STATES):
            raise ValueError(
                f"factor runtime activation_weight_tiers for {kind.value} must define {ACTIVATION_STATES}"
            )
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
            activation_weight_tiers=tuple(
                (status, float(tier_values[status]))
                for status in ACTIVATION_STATES
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
                activation_intervals=(
                    activation.intervals
                    if factor.kind in (runtime_configs or {})
                    else ()
                ),
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
    if not factor.activation_intervals:
        payload.pop("activation_intervals")
    return payload
