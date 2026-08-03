"""Fail-closed simulation protocol validation.

The names in this module describe simulated events only.  There is deliberately
no transport, device adapter, or actuator API in this project.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple


class ProtocolSafetyError(ValueError):
    """Raised when a simulated protocol violates a guardrail."""


@dataclass(frozen=True)
class StimulationPulse:
    start_ms: float
    channel: int
    amplitude: float
    duration_ms: float

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: Tuple[str, ...] = ()

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise ProtocolSafetyError("; ".join(self.errors))


@dataclass(frozen=True)
class Protocol:
    duration_ms: float
    pulses: Tuple[StimulationPulse, ...] = ()
    schema_version: int = 1

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "duration_ms": self.duration_ms,
            "pulses": [
                {
                    "start_ms": pulse.start_ms,
                    "channel": pulse.channel,
                    "amplitude": pulse.amplitude,
                    "duration_ms": pulse.duration_ms,
                }
                for pulse in self.pulses
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Protocol":
        if set(payload) != {"schema_version", "duration_ms", "pulses"}:
            raise ProtocolSafetyError("protocol contains unknown or missing fields")
        if int(payload["schema_version"]) != 1:
            raise ProtocolSafetyError("unsupported protocol schema version")
        pulses = []
        for raw in payload["pulses"]:
            if set(raw) != {"start_ms", "channel", "amplitude", "duration_ms"}:
                raise ProtocolSafetyError("pulse contains unknown or missing fields")
            pulses.append(StimulationPulse(**raw))
        return cls(float(payload["duration_ms"]), tuple(pulses), int(payload["schema_version"]))


@dataclass(frozen=True)
class SimulationSafetyPolicy:
    max_amplitude: float = 1.0
    max_duration_ms: float = 2.0
    min_inter_pulse_ms: float = 1.0
    max_pulses: int = 1000
    channel_count: int = 32
    max_total_exposure: float = 100.0

    def validate(self, pulses: Sequence[StimulationPulse], duration_ms: float | None = None) -> ValidationReport:
        errors: List[str] = []
        if self.max_amplitude <= 0 or self.max_duration_ms <= 0 or self.min_inter_pulse_ms < 0:
            errors.append("safety limits must be positive")
        if self.max_pulses < 0 or self.channel_count <= 0:
            errors.append("safety counts must be positive")
        if len(pulses) > self.max_pulses:
            errors.append("pulse count exceeds simulation limit")
        by_channel: dict[int, List[StimulationPulse]] = {}
        total_exposure = 0.0
        for index, pulse in enumerate(pulses):
            values = (pulse.start_ms, pulse.amplitude, pulse.duration_ms)
            if not all(math.isfinite(float(value)) for value in values):
                errors.append(f"pulse {index} contains a non-finite value")
                continue
            if int(pulse.channel) != pulse.channel or not 0 <= pulse.channel < self.channel_count:
                errors.append(f"pulse {index} channel is outside the simulation range")
            if pulse.start_ms < 0.0:
                errors.append(f"pulse {index} starts before time zero")
            if abs(pulse.amplitude) > self.max_amplitude:
                errors.append(f"pulse {index} amplitude exceeds the simulation guardrail")
            if pulse.duration_ms <= 0.0 or pulse.duration_ms > self.max_duration_ms:
                errors.append(f"pulse {index} duration exceeds the simulation guardrail")
            total_exposure += abs(pulse.amplitude) * max(0.0, pulse.duration_ms)
            by_channel.setdefault(int(pulse.channel), []).append(pulse)
            if duration_ms is not None and pulse.end_ms > duration_ms + 1e-9:
                errors.append(f"pulse {index} ends after the protocol duration")
        if total_exposure > self.max_total_exposure + 1e-9:
            errors.append("cumulative simulated exposure exceeds the guardrail")
        for channel, channel_pulses in by_channel.items():
            channel_pulses.sort(key=lambda pulse: (pulse.start_ms, pulse.duration_ms, pulse.amplitude))
            for previous, current in zip(channel_pulses, channel_pulses[1:]):
                if current.start_ms < previous.end_ms + self.min_inter_pulse_ms - 1e-9:
                    errors.append(f"channel {channel} violates the minimum inter-pulse interval")
        return ValidationReport(not errors, tuple(errors))


SafetyPolicy = SimulationSafetyPolicy


@dataclass(frozen=True)
class ClosedLoopResult:
    steps: int
    duration_ms: float
    spikes: list
    protocol: Protocol
