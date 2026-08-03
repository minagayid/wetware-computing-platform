from __future__ import annotations

import argparse
import json
import math

from .metrics import summarize_spikes
from .model import ReservoirConfig, SpikingReservoir


def simulate(steps: int = 120, seed: int = 7) -> dict:
    if steps <= 0:
        raise ValueError("steps must be positive")
    config = ReservoirConfig(neuron_count=24, input_channels=2, seed=seed, noise_std=0.0, connection_probability=0.15, input_gain=20.0)
    reservoir = SpikingReservoir(config)
    # Use a positive synthetic drive so the demo visibly exercises spike output;
    # real recordings should supply their own calibrated input projection.
    reservoir.input_weights = [[20.0, 20.0] for _ in range(config.neuron_count)]
    frames = [
        (0.8 + 0.2 * math.sin(index / 7.0), 0.3 + 0.3 * math.cos(index / 11.0))
        for index in range(steps)
    ]
    spikes = reservoir.run(frames)
    return {"seed": seed, "steps": steps, "digest": reservoir.digest(), **summarize_spikes(spikes, config.neuron_count, steps * config.dt_ms)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the local wetware-computing digital twin.")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    print(json.dumps(simulate(args.steps, args.seed), sort_keys=True))
    return 0
