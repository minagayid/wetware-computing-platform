from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .model import Spike


def summarize_spikes(spikes: Iterable[Spike], neuron_count: int, duration_ms: float) -> dict:
    spikes = list(spikes)
    if neuron_count <= 0 or duration_ms <= 0:
        raise ValueError("neuron_count and duration_ms must be positive")
    bins = defaultdict(set)
    for spike in spikes:
        if 0 <= spike.neuron_id < neuron_count:
            bins[round(float(spike.time_ms), 9)].add(spike.neuron_id)
    if bins:
        synchrony = sum(max(0, len(active) - 1) / max(1, neuron_count - 1) for active in bins.values()) / len(bins)
    else:
        synchrony = 0.0
    return {
        "spike_count": len(spikes),
        "spike_rate_hz": len(spikes) / (neuron_count * (duration_ms / 1000.0)),
        "active_neurons": len({spike.neuron_id for spike in spikes}),
        "synchrony_index": max(0.0, min(1.0, synchrony)),
        "duration_ms": float(duration_ms),
    }
