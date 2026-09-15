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


MAX_PROTOCOL_PULSES = 1_000


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolSafetyError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ProtocolSafetyError(f"{label} must be a finite number") from None
    if not math.isfinite(number):
        raise ProtocolSafetyError(f"{label} must be a finite number")
    return number


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

    def __post_init__(self) -> None:
        if not isinstance(self.pulses, (tuple, list)):
            raise ProtocolSafetyError("protocol pulses must be a list or tuple")
        if len(self.pulses) > MAX_PROTOCOL_PULSES:
            raise ProtocolSafetyError("protocol pulse count exceeds the parse limit")
        object.__setattr__(self, "pulses", tuple(self.pulses))

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
        if not isinstance(payload, dict):
            raise ProtocolSafetyError("protocol must be an object")
        if set(payload) != {"schema_version", "duration_ms", "pulses"}:
            raise ProtocolSafetyError("protocol contains unknown or missing fields")
        version = payload["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ProtocolSafetyError("unsupported protocol schema version")
        duration_ms = _finite_number(payload["duration_ms"], "protocol duration")
        if duration_ms < 0.0:
            raise ProtocolSafetyError("protocol duration must be non-negative")
        raw_pulses = payload["pulses"]
        if not isinstance(raw_pulses, list):
            raise ProtocolSafetyError("protocol pulses must be an array")
        if len(raw_pulses) > MAX_PROTOCOL_PULSES:
            raise ProtocolSafetyError("protocol pulse count exceeds the parse limit")
        pulses = []
        for index, raw in enumerate(raw_pulses):
            if not isinstance(raw, dict):
                raise ProtocolSafetyError(f"pulse {index} must be an object")
            if set(raw) != {"start_ms", "channel", "amplitude", "duration_ms"}:
                raise ProtocolSafetyError("pulse contains unknown or missing fields")
            channel = raw["channel"]
            if isinstance(channel, bool) or not isinstance(channel, int):
                raise ProtocolSafetyError(f"pulse {index} channel must be an integer")
            pulses.append(
                StimulationPulse(
                    start_ms=_finite_number(raw["start_ms"], f"pulse {index} start"),
                    channel=channel,
                    amplitude=_finite_number(raw["amplitude"], f"pulse {index} amplitude"),
                    duration_ms=_finite_number(raw["duration_ms"], f"pulse {index} duration"),
                )
            )
        return cls(duration_ms, tuple(pulses), version)


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
        try:
            max_amplitude, max_duration_ms, min_inter_pulse_ms, max_total_exposure = (
                _finite_number(value, "safety limit")
                for value in (
                    self.max_amplitude,
                    self.max_duration_ms,
                    self.min_inter_pulse_ms,
                    self.max_total_exposure,
                )
            )
        except ProtocolSafetyError:
            errors.append("safety limits must be finite numbers")
            return ValidationReport(False, tuple(errors))
        if max_amplitude <= 0 or max_duration_ms <= 0 or min_inter_pulse_ms < 0 or max_total_exposure <= 0:
            errors.append("safety limits must be positive (inter-pulse interval may be zero)")
        if (
            isinstance(self.max_pulses, bool)
            or not isinstance(self.max_pulses, int)
            or self.max_pulses < 0
            or isinstance(self.channel_count, bool)
            or not isinstance(self.channel_count, int)
            or self.channel_count <= 0
        ):
            errors.append("max_pulses must be non-negative and channel_count must be positive integers")
        if not isinstance(pulses, (list, tuple)):
            errors.append("pulses must be a list or tuple")
        if errors:
            return ValidationReport(False, tuple(errors))

        protocol_duration = None
        if duration_ms is not None:
            try:
                protocol_duration = _finite_number(duration_ms, "protocol duration")
            except ProtocolSafetyError:
                protocol_duration = None
                errors.append("protocol duration must be finite and non-negative")
            if protocol_duration is not None and protocol_duration < 0.0:
                errors.append("protocol duration must be finite and non-negative")
        if errors:
            return ValidationReport(False, tuple(errors))
        if len(pulses) > self.max_pulses:
            errors.append("pulse count exceeds simulation limit")
        by_channel: dict[int, List[Tuple[float, float]]] = {}
        total_exposure = 0.0
        for index, pulse in enumerate(pulses):
            if not isinstance(pulse, StimulationPulse):
                errors.append(f"pulse {index} must be a stimulation pulse")
                continue
            try:
                start_ms = _finite_number(pulse.start_ms, f"pulse {index} start")
                amplitude = _finite_number(pulse.amplitude, f"pulse {index} amplitude")
                pulse_duration_ms = _finite_number(pulse.duration_ms, f"pulse {index} duration")
            except ProtocolSafetyError:
                errors.append(f"pulse {index} contains a non-finite or non-numeric value")
                continue
            valid_channel = isinstance(pulse.channel, int) and not isinstance(pulse.channel, bool)
            channel = pulse.channel if valid_channel else -1
            if not valid_channel or not 0 <= channel < self.channel_count:
                errors.append(f"pulse {index} channel is outside the simulation range")
            if start_ms < 0.0:
                errors.append(f"pulse {index} starts before time zero")
            if abs(amplitude) > max_amplitude:
                errors.append(f"pulse {index} amplitude exceeds the simulation guardrail")
            if pulse_duration_ms <= 0.0 or pulse_duration_ms > max_duration_ms:
                errors.append(f"pulse {index} duration exceeds the simulation guardrail")
            total_exposure += abs(amplitude) * max(0.0, pulse_duration_ms)
            if valid_channel:
                by_channel.setdefault(channel, []).append((start_ms, pulse_duration_ms))
            if protocol_duration is not None and start_ms + pulse_duration_ms > protocol_duration + 1e-9:
                errors.append(f"pulse {index} ends after the protocol duration")
        if total_exposure > max_total_exposure + 1e-9:
            errors.append("cumulative simulated exposure exceeds the guardrail")
        for channel, channel_pulses in by_channel.items():
            channel_pulses.sort(key=lambda pulse: (pulse[0], pulse[1]))
            for previous, current in zip(channel_pulses, channel_pulses[1:]):
                if current[0] < previous[0] + previous[1] + min_inter_pulse_ms - 1e-9:
                    errors.append(f"channel {channel} violates the minimum inter-pulse interval")
        return ValidationReport(not errors, tuple(errors))


SafetyPolicy = SimulationSafetyPolicy


@dataclass(frozen=True)
class ClosedLoopResult:
    steps: int
    duration_ms: float
    spikes: list
    protocol: Protocol
