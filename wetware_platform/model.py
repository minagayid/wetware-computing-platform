"""Deterministic spiking-reservoir digital twin.

This module models signal processing and learning in software only.  It has no
device, network, stimulation, or cell-culture interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class ReservoirConfig:
    neuron_count: int = 32
    input_channels: int = 2
    dt_ms: float = 1.0
    membrane_tau_ms: float = 20.0
    synaptic_tau_ms: float = 5.0
    rest_mV: float = -65.0
    threshold_mV: float = -50.0
    reset_mV: float = -68.0
    refractory_ms: float = 2.0
    connection_probability: float = 0.1
    weight_scale: float = 1.5
    max_weight: float = 5.0
    plasticity_rate: float = 0.002
    input_gain: float = 1.0
    noise_std: float = 0.0
    seed: int = 7

    def __post_init__(self) -> None:
        if int(self.neuron_count) != self.neuron_count or self.neuron_count <= 0:
            raise ValueError("neuron_count must be a positive integer")
        if int(self.input_channels) != self.input_channels or self.input_channels <= 0:
            raise ValueError("input_channels must be a positive integer")
        for name in ("dt_ms", "membrane_tau_ms", "synaptic_tau_ms", "refractory_ms", "max_weight"):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("weight_scale", "plasticity_rate", "input_gain", "noise_std"):
            if _finite(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        probability = _finite(self.connection_probability, "connection_probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("connection_probability must be between 0 and 1")
        for name in ("rest_mV", "threshold_mV", "reset_mV"):
            _finite(getattr(self, name), name)


@dataclass(frozen=True)
class Spike:
    time_ms: float
    neuron_id: int


class SpikingReservoir:
    """A compact leaky integrate-and-fire recurrent reservoir."""

    SNAPSHOT_SCHEMA = "wetware-platform.reservoir.v1"

    def __init__(self, config: ReservoirConfig | None = None):
        self.config = config or ReservoirConfig()
        self._rng = random.Random(self.config.seed)
        n = self.config.neuron_count
        self.input_weights: List[List[float]] = [
            [self._rng.uniform(-self.config.input_gain, self.config.input_gain) for _ in range(self.config.input_channels)]
            for _ in range(n)
        ]
        self.weights: List[Dict[int, float]] = [dict() for _ in range(n)]
        self.incoming: List[List[int]] = [[] for _ in range(n)]
        for source in range(n):
            for target in range(n):
                if source != target and self._rng.random() < self.config.connection_probability:
                    self.weights[source][target] = self._rng.uniform(-self.config.weight_scale, self.config.weight_scale)
                    self.incoming[target].append(source)
        self._initialize_state()

    def _initialize_state(self) -> None:
        n = self.config.neuron_count
        self.membrane: List[float] = [self.config.rest_mV] * n
        self.refractory_until: List[float] = [0.0] * n
        self.pending_current: List[float] = [0.0] * n
        self.pre_trace: List[float] = [0.0] * n
        self.post_trace: List[float] = [0.0] * n
        self.time_ms = 0.0
        self.step_index = 0

    def reset(self, seed: int | None = None) -> None:
        """Reset state and optionally recreate the seeded network."""
        config = self.config if seed is None else ReservoirConfig(**{**asdict(self.config), "seed": int(seed)})
        self.__init__(config)

    def step(self, input_values: Sequence[float], enable_plasticity: bool = True) -> List[Spike]:
        values = tuple(float(value) for value in input_values)
        if len(values) != self.config.input_channels:
            raise ValueError(f"expected {self.config.input_channels} input channels")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("input values must be finite")

        dt = self.config.dt_ms
        membrane_decay = dt / self.config.membrane_tau_ms
        synaptic_decay = math.exp(-dt / self.config.synaptic_tau_ms)
        trace_decay = math.exp(-dt / max(self.config.membrane_tau_ms, self.config.synaptic_tau_ms))
        self.pending_current = [current * synaptic_decay for current in self.pending_current]
        self.pre_trace = [trace * trace_decay for trace in self.pre_trace]
        self.post_trace = [trace * trace_decay for trace in self.post_trace]

        spikes: List[Spike] = []
        for neuron_id in range(self.config.neuron_count):
            if self.time_ms < self.refractory_until[neuron_id]:
                self.membrane[neuron_id] = self.config.reset_mV
                self.pending_current[neuron_id] = 0.0
                continue
            external = sum(
                values[channel] * self.input_weights[neuron_id][channel]
                for channel in range(self.config.input_channels)
            )
            noise = self._rng.gauss(0.0, self.config.noise_std) if self.config.noise_std else 0.0
            current = self.pending_current[neuron_id] + external + noise
            self.pending_current[neuron_id] = 0.0
            self.membrane[neuron_id] += membrane_decay * (self.config.rest_mV - self.membrane[neuron_id] + current)
            if self.membrane[neuron_id] >= self.config.threshold_mV:
                spikes.append(Spike(self.time_ms, neuron_id))
                self.membrane[neuron_id] = self.config.reset_mV
                self.refractory_until[neuron_id] = self.time_ms + self.config.refractory_ms

        if enable_plasticity and spikes:
            self._apply_plasticity(spikes)
        for spike in spikes:
            for target, weight in self.weights[spike.neuron_id].items():
                self.pending_current[target] += weight

        self.time_ms += dt
        self.step_index += 1
        return spikes

    def _apply_plasticity(self, spikes: Iterable[Spike]) -> None:
        learning_rate = self.config.plasticity_rate
        spike_ids = [spike.neuron_id for spike in spikes]
        for source in spike_ids:
            for target in list(self.weights[source]):
                self.weights[source][target] = self._bounded_weight(
                    self.weights[source][target] + learning_rate * self.post_trace[target]
                )
            self.pre_trace[source] += 1.0
        for target in spike_ids:
            for source in self.incoming[target]:
                self.weights[source][target] = self._bounded_weight(
                    self.weights[source][target] - learning_rate * self.pre_trace[source]
                )
            self.post_trace[target] += 1.0

    def _bounded_weight(self, weight: float) -> float:
        limit = self.config.max_weight
        return max(-limit, min(limit, float(weight)))

    def run(self, frames: Iterable[Sequence[float]], enable_plasticity: bool = True) -> List[Spike]:
        spikes: List[Spike] = []
        for frame in frames:
            spikes.extend(self.step(frame, enable_plasticity=enable_plasticity))
        return spikes

    def weights_snapshot(self) -> List[Dict[int, float]]:
        return [{int(target): float(weight) for target, weight in sorted(row.items())} for row in self.weights]

    def snapshot(self) -> dict:
        return {
            "schema": self.SNAPSHOT_SCHEMA,
            "config": asdict(self.config),
            "time_ms": self.time_ms,
            "step_index": self.step_index,
            "input_weights": self.input_weights,
            "weights": self.weights_snapshot(),
            "membrane": self.membrane,
            "refractory_until": self.refractory_until,
            "pending_current": self.pending_current,
            "pre_trace": self.pre_trace,
            "post_trace": self.post_trace,
            "rng_state": _to_jsonable(self._rng.getstate()),
        }

    def digest(self) -> str:
        canonical = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "SpikingReservoir":
        if snapshot.get("schema") != cls.SNAPSHOT_SCHEMA:
            raise ValueError("unsupported reservoir snapshot schema")
        reservoir = cls(ReservoirConfig(**snapshot["config"]))
        n = reservoir.config.neuron_count
        for name in ("input_weights", "membrane", "refractory_until", "pending_current", "pre_trace", "post_trace"):
            if len(snapshot[name]) != n:
                raise ValueError(f"snapshot field has wrong length: {name}")
        if len(snapshot["weights"]) != n:
            raise ValueError("snapshot field has wrong length: weights")
        reservoir.input_weights = [[float(value) for value in row] for row in snapshot["input_weights"]]
        if any(len(row) != reservoir.config.input_channels for row in reservoir.input_weights):
            raise ValueError("snapshot input_weights has wrong channel count")
        if not all(math.isfinite(value) for row in reservoir.input_weights for value in row):
            raise ValueError("snapshot contains non-finite input weights")
        reservoir.weights = [{int(target): reservoir._bounded_weight(float(weight)) for target, weight in row.items()} for row in snapshot["weights"]]
        reservoir.incoming = [[] for _ in range(n)]
        for source, row in enumerate(reservoir.weights):
            for target in row:
                if not 0 <= target < n or target == source:
                    raise ValueError("snapshot contains an invalid synapse")
                reservoir.incoming[target].append(source)
        reservoir.membrane = [float(value) for value in snapshot["membrane"]]
        reservoir.refractory_until = [float(value) for value in snapshot["refractory_until"]]
        reservoir.pending_current = [float(value) for value in snapshot["pending_current"]]
        reservoir.pre_trace = [float(value) for value in snapshot["pre_trace"]]
        reservoir.post_trace = [float(value) for value in snapshot["post_trace"]]
        reservoir.time_ms = float(snapshot["time_ms"])
        reservoir.step_index = int(snapshot["step_index"])
        continuous = reservoir.membrane + reservoir.refractory_until + reservoir.pending_current + reservoir.pre_trace + reservoir.post_trace
        if not math.isfinite(reservoir.time_ms) or reservoir.time_ms < 0.0 or not all(math.isfinite(value) for value in continuous):
            raise ValueError("snapshot contains non-finite state")
        reservoir._rng.setstate(_from_jsonable(snapshot["rng_state"]))
        return reservoir


def _to_jsonable(value):
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _from_jsonable(value):
    if isinstance(value, list):
        return tuple(_from_jsonable(item) for item in value)
    return value
