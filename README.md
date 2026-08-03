# Wetware Computing Platform

An offline, simulation-first research platform for studying computation inspired
by neural tissue. It provides a deterministic leaky integrate-and-fire reservoir,
bounded plasticity, read-only recorded-data ingestion, fail-closed simulated
protocol guardrails, reproducible snapshots, manifests, and metrics.

## Safety boundary

This repository is software-only. It cannot connect to, stimulate, culture,
monitor, or control living tissue, laboratory equipment, pumps, incubators,
electrode arrays, serial devices, GPIO, or network services. `StimulationPulse`
objects are immutable numbers used inside the digital simulation; they are not
hardware commands and must not be treated as biological safety limits.

The reservoir is a computational abstraction, not an organoid digital twin, a
consciousness model, a clinical tool, or evidence of biological performance.
Any physical wetware research requires qualified personnel, institutional review,
appropriate biosafety governance, consent/provenance controls, and independent
validation outside this project.

## Run locally

Python 3.10+ and the standard library are sufficient:

```bash
cd wetware-computing-platform
python -S run_demo.py --steps 120 --seed 7
python -S -m unittest discover -v
```

The demo prints deterministic JSON containing the model digest, spike count,
rate, and synchrony. There is no required backend, cloud API, hosted LLM, or
network connection. Analysis can be added through local Python code or a local
OpenAI-compatible/open-source model without changing the simulation core.

## Minimal API

```python
from wetware_platform import ReservoirConfig, SpikingReservoir

reservoir = SpikingReservoir(ReservoirConfig(neuron_count=32, seed=7))
spikes = reservoir.run([[0.2, 0.4], [0.5, 0.1]] * 100)
snapshot = reservoir.snapshot()
restored = SpikingReservoir.from_snapshot(snapshot)
assert reservoir.digest() == restored.digest()
```

Closed-loop experiments remain simulation-only and validate their complete
protocol before execution:

```python
from wetware_platform import ClosedLoopExperiment, SafetyPolicy, StimulationPulse

experiment = ClosedLoopExperiment(reservoir, SafetyPolicy(channel_count=2))
result = experiment.run(
    [[0.0, 0.0]] * 100,
    [StimulationPulse(start_ms=20.0, channel=0, amplitude=0.5, duration_ms=1.0)],
)
```

For recorded data, use `RecordedDataSource.from_json(path)`. It only reads a
strict versioned JSON recording, checks finite values, channel counts, and
monotonic timestamps, and exposes values to the simulation. It cannot write to
the source or emit commands.

## Data format

Recorded input is local JSON with exactly these fields:

```json
{
  "schema_version": 1,
  "channel_count": 2,
  "frames": [
    {"timestamp_ms": 0.0, "values": [0.1, 0.2]},
    {"timestamp_ms": 1.0, "values": [0.2, 0.3]}
  ]
}
```

Do not place identifiable human-derived data in the repository. Keep consent,
provenance, de-identification, retention, and access-control records alongside
any private dataset under the governing institution's policy.

## Model and reproducibility

The core uses a leaky integrate-and-fire update with deterministic seeded sparse
recurrent connections. Optional software STDP changes weights only within the
configured bound. Every snapshot contains the schema version, configuration,
network state, PRNG state, and simulation time. `digest()` hashes canonical JSON;
restoring a snapshot and continuing produces the same trajectory as uninterrupted
execution.

`RunManifest` records the model schema, seed, timestep, configuration, input
digest, event-ordering rule, and runtime version. Manifests are provenance aids,
not substitutes for scientific replication.

## Structure

- `wetware_platform/model.py` — LIF reservoir, bounded plasticity, snapshots
- `wetware_platform/protocol.py` — immutable simulated pulses and fail-closed validation
- `wetware_platform/experiment.py` — local closed-loop simulation runner
- `wetware_platform/recordings.py` — strict read-only recording loader
- `wetware_platform/metrics.py` — spike rate and synchrony metrics
- `wetware_platform/manifest.py` — canonical run provenance and digest
- `wetware_platform/cli.py` — standard-library-only demo command
- `run_demo.py` — direct local entry point
- `test_wetware_platform.py` — deterministic, safety, replay, and CLI tests
- `SOURCES.md` — background references; claims should be checked against primary literature

## Research roadmap

Safe future extensions include richer software neuron models, calibration against
public non-identifying recordings, streaming feature extraction, benchmark task
suites, and uncertainty-aware decoders. Hardware drivers, actuator control,
bioreactor recipes, culture protocols, and biological safety claims remain out of
scope for this repository.
