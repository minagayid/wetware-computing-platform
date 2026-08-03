from __future__ import annotations

from typing import Iterable, Sequence

from .model import Spike, SpikingReservoir
from .protocol import ClosedLoopResult, Protocol, ProtocolSafetyError, SafetyPolicy, StimulationPulse


class ClosedLoopExperiment:
    """Run a digital closed-loop experiment over immutable simulated events."""

    def __init__(self, reservoir: SpikingReservoir, policy: SafetyPolicy | None = None):
        self.reservoir = reservoir
        self.policy = policy or SafetyPolicy(channel_count=reservoir.config.input_channels)

    def run(self, frames: Iterable[Sequence[float]], pulses: Iterable[StimulationPulse] = ()) -> ClosedLoopResult:
        frame_list = [tuple(float(value) for value in frame) for frame in frames]
        duration_ms = len(frame_list) * self.reservoir.config.dt_ms
        pulse_list = tuple(pulses)
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
            enriched.append(values)
        spikes = self.reservoir.run(enriched)
        return ClosedLoopResult(len(frame_list), duration_ms, spikes, protocol)
