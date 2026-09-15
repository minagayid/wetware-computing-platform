from __future__ import annotations

import math
from typing import Iterable, Sequence

from .model import (
    MAX_SIMULATION_FRAMES,
    MAX_SIMULATION_OPERATIONS,
    MAX_SIMULATION_SAMPLES,
    Spike,
    SpikingReservoir,
)
from .protocol import MAX_PROTOCOL_PULSES, ClosedLoopResult, Protocol, ProtocolSafetyError, SafetyPolicy, StimulationPulse


class ClosedLoopExperiment:
    """Run a digital closed-loop experiment over immutable simulated events."""

    def __init__(self, reservoir: SpikingReservoir, policy: SafetyPolicy | None = None):
        self.reservoir = reservoir
        self.policy = policy or SafetyPolicy(channel_count=reservoir.config.input_channels)

    def run(self, frames: Iterable[Sequence[float]], pulses: Iterable[StimulationPulse] = ()) -> ClosedLoopResult:
        frame_list = []
        operations_per_step = self.reservoir._worst_case_operations_per_step()
        for frame in frames:
            if len(frame_list) >= MAX_SIMULATION_FRAMES:
                raise ValueError(f"simulation exceeds the {MAX_SIMULATION_FRAMES}-frame limit")
            next_frame_count = len(frame_list) + 1
            sample_count = next_frame_count * self.reservoir.config.input_channels
            operation_count = next_frame_count * operations_per_step
            if sample_count > MAX_SIMULATION_SAMPLES:
                raise ValueError(f"simulation exceeds the {MAX_SIMULATION_SAMPLES}-sample limit")
            if operation_count > MAX_SIMULATION_OPERATIONS:
                raise ValueError(f"simulation exceeds the {MAX_SIMULATION_OPERATIONS}-operation limit")
            values = []
            iterator = iter(frame)
            for _ in range(self.reservoir.config.input_channels + 1):
                try:
                    value = next(iterator)
                except StopIteration:
                    break
                if len(values) >= self.reservoir.config.input_channels:
                    raise ValueError(f"expected {self.reservoir.config.input_channels} input channels")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("input values must be finite")
                values.append(number)
            if len(values) != self.reservoir.config.input_channels:
                raise ValueError(f"expected {self.reservoir.config.input_channels} input channels")
            frame_list.append(tuple(values))
        duration_ms = len(frame_list) * self.reservoir.config.dt_ms
        pulse_list = []
        for pulse in pulses:
            if len(pulse_list) >= MAX_PROTOCOL_PULSES:
                raise ProtocolSafetyError(f"protocol pulse count exceeds the parse limit of {MAX_PROTOCOL_PULSES}")
            pulse_list.append(pulse)
        pulse_list = tuple(pulse_list)
        protocol = Protocol(duration_ms=duration_ms, pulses=pulse_list)
        self.policy.validate(pulse_list, duration_ms).raise_if_invalid()
        if any(pulse.channel >= self.reservoir.config.input_channels for pulse in pulse_list):
            raise ProtocolSafetyError("pulse channel is not represented by the reservoir input")
        enriched = []
        for index, frame in enumerate(frame_list):
            values = list(frame)
            if len(values) != self.reservoir.config.input_channels:
                raise ValueError(f"expected {self.reservoir.config.input_channels} input channels")
            time_ms = index * self.reservoir.config.dt_ms
            for pulse in pulse_list:
                if pulse.start_ms <= time_ms < pulse.end_ms:
                    values[pulse.channel] += pulse.amplitude
            if not all(math.isfinite(value) for value in values):
                raise ValueError("enriched input values must be finite")
            enriched.append(values)
        spikes = self.reservoir.run(enriched)
        return ClosedLoopResult(len(frame_list), duration_ms, spikes, protocol)
