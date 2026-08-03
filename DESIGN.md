# Design and safety contract

## One-way software boundary

The architecture intentionally has no transport layer. It is a pure function of
configuration, immutable input frames, simulated pulses, and a seeded PRNG:

```text
local recording ──read-only──▶ validation ──▶ LIF reservoir ──▶ metrics/snapshot
                                      ▲
                         simulated pulse protocol
```

There is no reverse edge from the simulation to a device or network. Any future
hardware work must be a separate, independently reviewed project.

## State update

For each neuron, the membrane state follows a discrete leaky integrate-and-fire
update:

```text
v[t+1] = v[t] + dt/tau * (v_rest - v[t] + I_external + I_recurrent + noise)
```

Crossing threshold emits a timestamped `Spike`, resets the membrane, and starts
the refractory interval. Recurrent weights are bounded. Software STDP uses
decaying pre/post traces and clamps every update to `[-max_weight, max_weight]`.

This is a useful computational baseline—not a biophysical organoid simulation.
It does not model cell types, morphology, gene expression, perfusion, MEA
calibration, tissue health, or consciousness.

## Protocol guardrails

`SimulationSafetyPolicy` validates the entire immutable protocol before a run.
It rejects unknown schema fields, non-finite values, negative time, invalid
channels, excessive amplitude or duration, excessive count/exposure, and
per-channel pulse overlap/cadence violations. There is no force flag and no
policy-disable switch. A failed report must be raised before any reservoir state
is advanced.

These values constrain a numerical experiment only. They are not stimulation
limits, therapeutic thresholds, or biological safety recommendations.

## Reproducibility

Snapshots include model schema, configuration, network weights, membrane and
trace state, elapsed time, and PRNG state. Canonical JSON digests make corruption
or accidental divergence visible. `RecordedDataSource` rejects malformed,
non-monotonic, non-finite, or wrong-width frames before they reach the model.

For human-derived or sensitive recordings, provenance and consent metadata must
remain in a governed private system. The sample software format intentionally
contains no identity fields.
