"""Strict, serializable contracts shared by the designer and evaluator."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TOPOLOGIES = (
    "buffer",
    "inverting_amplifier",
    "noninverting_amplifier",
    "passive_lowpass",
    "passive_highpass",
    "active_lowpass_1",
    "active_highpass_1",
    "active_lowpass_2",
    "active_highpass_2",
    "common_emitter",
    "emitter_follower",
    "amplifier",
    "lowpass",
    "highpass",
)
_ALIASES = {
    "inverting": "inverting_amplifier",
    "noninverting": "noninverting_amplifier",
    "rc_lowpass": "passive_lowpass",
    "rc_highpass": "passive_highpass",
    "voltage_follower": "buffer",
    "non_inverting_amplifier": "noninverting_amplifier",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DesignSpec(StrictModel):
    """All quantities use SI units. Amplitude means peak, never peak-to-peak."""

    topology: str
    supply_positive_v: float
    supply_negative_v: float = 0.0
    input_amplitude_v: float = Field(gt=0, le=100)
    input_frequency_hz: float = Field(gt=0, le=1e8)
    load_ohm: float = Field(gt=0, le=1e12)
    source_resistance_ohm: float = Field(default=50, ge=0, le=1e9)
    input_offset_v: float = 0
    gain: float | None = None
    cutoff_hz: float | None = Field(default=None, gt=0, le=1e8)
    gain_tolerance: float = Field(default=0.05, gt=0, le=1)
    cutoff_tolerance: float = Field(default=0.1, gt=0, le=1)
    max_thd: float = Field(default=0.03, gt=0, le=1)
    max_supply_current_a: float | None = Field(default=None, gt=0)
    max_output_offset_v: float | None = Field(default=None, ge=0)
    min_output_swing_v: float | None = Field(default=None, gt=0)
    max_parts: int | None = Field(default=None, ge=1, le=1000)
    opamp_model: str = "LM358"
    transistor_model: str = "2N3904"
    temperature_c: float = Field(default=27, ge=-55, le=150)
    resistor_tolerance: float = Field(default=0.01, ge=0, le=0.2)
    capacitor_tolerance: float = Field(default=0.05, ge=0, le=0.3)
    corner_policy: Literal["nominal", "tolerance"] = "nominal"
    max_candidates: int = Field(default=20, ge=1, le=256)
    min_filter_order: int = Field(default=1, ge=1, le=2)
    stopband_frequency_hz: float | None = Field(default=None, gt=0, le=1e9)
    max_stopband_gain: float | None = Field(default=None, gt=0)

    @field_validator("topology")
    @classmethod
    def topology_name(cls, value: str) -> str:
        value = _ALIASES.get(value.lower().replace("-", "_"), value.lower().replace("-", "_"))
        if value not in TOPOLOGIES:
            raise ValueError(f"unsupported topology {value!r}; expected one of {TOPOLOGIES}")
        return value

    @field_validator("opamp_model", "transistor_model")
    @classmethod
    def safe_model(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", value):
            raise ValueError("model must be an allowlisted catalog identifier")
        return value

    @model_validator(mode="after")
    def coherent(self) -> DesignSpec:
        if not 0 < self.supply_positive_v - self.supply_negative_v <= 60:
            raise ValueError("supply span must be between 0 and 60 V")
        if self.gain == 0:
            raise ValueError("gain target cannot be zero")
        if "pass" in self.topology and self.cutoff_hz is None:
            raise ValueError("filter topologies require cutoff_hz")
        if "pass" not in self.topology and self.cutoff_hz is not None:
            raise ValueError("cutoff_hz is measured only for filter topologies; omit it for non-filter circuits")
        if "pass" in self.topology and self.gain is not None and self.gain <= 0:
            raise ValueError("supported filter templates require positive gain")
        if self.min_filter_order == 2 and (
            self.topology.startswith("passive_") or self.topology in ("active_lowpass_1", "active_highpass_1")
        ):
            raise ValueError("explicit first-order topology conflicts with min_filter_order=2")
        if (
            self.topology in ("inverting_amplifier", "noninverting_amplifier", "amplifier", "common_emitter")
            and self.gain is None
        ):
            raise ValueError("amplifier topology requires gain")
        if self.topology == "noninverting_amplifier" and self.gain is not None and self.gain < 1:
            raise ValueError("noninverting gain must be at least one")
        if self.topology == "amplifier" and self.gain is not None and 0 < self.gain < 1:
            raise ValueError("no supported amplifier template realizes a positive gain below one")
        if self.gain is not None and abs(self.gain) > 1e9:
            raise ValueError("gain magnitude must not exceed 1e9")
        if (
            self.topology in ("inverting_amplifier", "amplifier")
            and self.gain is not None
            and self.gain < 0
            and abs(self.gain) * (47e3 + self.source_resistance_ohm) > 1e15
        ):
            raise ValueError("inverting feedback resistor would exceed the 1e15 ohm component limit")
        if (
            self.topology in ("inverting_amplifier", "common_emitter")
            and self.gain is not None
            and self.gain >= 0
        ):
            raise ValueError("inverting topologies require a negative signed gain")
        if (self.stopband_frequency_hz is None) != (self.max_stopband_gain is None):
            raise ValueError("stopband_frequency_hz and max_stopband_gain must be specified together")
        return self

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


class Component(StrictModel):
    ref: str
    kind: Literal["R", "C", "L", "OPAMP", "BJT", "DIODE"]
    nodes: tuple[str, ...]
    value: float | None = Field(default=None, gt=0, le=1e15)
    model: str | None = None
    package: str | None = None
    role: str = "signal"
    parameters: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check(self) -> Component:
        for token in (self.ref, *self.nodes, *([self.package] if self.package else [])):
            if not re.fullmatch(r"[A-Za-z0-9_]{1,48}", token):
                raise ValueError("references and nodes must be simple alphanumeric identifiers")
        expected = {"R": 2, "C": 2, "L": 2, "OPAMP": 5, "BJT": 3, "DIODE": 2}[self.kind]
        if len(self.nodes) != expected or len(set(self.nodes)) < 2:
            raise ValueError(f"{self.kind} requires {expected} nodes, with distinct terminals")
        if self.kind in ("R", "C", "L") and self.value is None:
            raise ValueError("passive components require a numeric value")
        if self.kind in ("OPAMP", "BJT", "DIODE") and not self.model:
            raise ValueError("active components require a catalog model")
        if self.model and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", self.model):
            raise ValueError("invalid model identifier")
        if self.parameters:
            raise ValueError("arbitrary device parameters are not supported in v1")
        if self.package and self.kind != "OPAMP" and self.package != self.ref:
            raise ValueError("only opamp channels may share a physical package")
        return self


class Circuit(StrictModel):
    name: str = "candidate"
    topology: str
    components: tuple[Component, ...]
    spec: DesignSpec | None = None
    input_node: str = "in"
    output_node: str = "out"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check(self) -> Circuit:
        if len(self.components) > 200:
            raise ValueError("circuit exceeds the 200-component limit")
        refs = [c.ref.lower() for c in self.components]
        if len(refs) != len(set(refs)):
            raise ValueError("component references must be unique (case insensitive)")
        for node in (self.input_node, self.output_node):
            if not re.fullmatch(r"[A-Za-z0-9_]{1,48}", node):
                raise ValueError("invalid terminal node")
        packages = {}
        for part in self.components:
            packages.setdefault((part.package or part.ref).lower(), []).append(part)
        for group in packages.values():
            if len(group) > 1 and (
                any(p.kind != "OPAMP" for p in group) or len({p.model for p in group}) != 1
            ):
                raise ValueError("physical packages may only group channels of the same opamp model")
        return self


class SimulationResult(StrictModel):
    status: Literal["ok", "invalid", "failed", "timeout", "unavailable"]
    metrics: dict[str, float] = Field(default_factory=dict)
    analyses: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    logs: dict[str, str] = Field(default_factory=dict)
    artifact_dir: str | None = None
    model_checksums: dict[str, str] = Field(default_factory=dict)
    spec_fingerprint: str | None = None
    fixture_fingerprint: str | None = None
    simulator_version: str | None = None


class Scorecard(StrictModel):
    valid: bool
    feasible: bool
    violations: dict[str, float]
    max_violation: float
    sum_violation: float
    package_count: int
    active_count: int
    rank_key: tuple[float, ...]
    explanation: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    points: float = 0
    points_breakdown: dict[str, float] = Field(default_factory=dict)


def normalize_spec(spec: DesignSpec | dict[str, Any]) -> DesignSpec:
    return spec if isinstance(spec, DesignSpec) else DesignSpec.model_validate(spec)
