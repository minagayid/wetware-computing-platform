import json
import subprocess
import sys
import unittest
from pathlib import Path

from wetware_platform import (
    ClosedLoopExperiment,
    ProtocolSafetyError,
    ReservoirConfig,
    SafetyPolicy,
    SpikingReservoir,
    StimulationPulse,
    summarize_spikes,
)
from wetware_platform.recordings import RecordedDataSource
from wetware_platform.manifest import RunManifest


class ReservoirTests(unittest.TestCase):
    def _config(self, **overrides):
        values = {
            "neuron_count": 8,
            "input_channels": 1,
            "seed": 11,
            "connection_probability": 0.4,
            "noise_std": 0.0,
            "input_gain": 1.0,
        }
        values.update(overrides)
        return ReservoirConfig(**values)

    def test_config_rejects_unsafe_or_non_finite_parameters(self):
        with self.assertRaises(ValueError):
            ReservoirConfig(neuron_count=0)
        with self.assertRaises(ValueError):
            ReservoirConfig(dt_ms=0.0)
        with self.assertRaises(ValueError):
            ReservoirConfig(connection_probability=1.1)
        with self.assertRaises(ValueError):
            ReservoirConfig(noise_std=float("nan"))

    def test_same_seed_and_input_are_bitwise_replayable(self):
        config = self._config()
        frames = [[0.25], [1.0], [0.0], [0.5]] * 20
        first = SpikingReservoir(config)
        second = SpikingReservoir(config)
        first_spikes = first.run(frames)
        second_spikes = second.run(frames)
        self.assertEqual(first_spikes, second_spikes)
        self.assertEqual(first.digest(), second.digest())

    def test_lif_spikes_are_recorded_and_plasticity_is_bounded(self):
        reservoir = SpikingReservoir(self._config(connection_probability=1.0, max_weight=1.5, plasticity_rate=0.01))
        reservoir.input_weights = [[1.0] for _ in range(reservoir.config.neuron_count)]
        before = reservoir.weights_snapshot()
        spikes = reservoir.run([[100.0]] * 25, enable_plasticity=True)
        self.assertTrue(spikes)
        self.assertTrue(all(abs(weight) <= 1.5 for row in reservoir.weights_snapshot() for weight in row.values()))
        self.assertNotEqual(before, reservoir.weights_snapshot())
        self.assertTrue(all(0 <= spike.neuron_id < 8 for spike in spikes))

    def test_snapshot_round_trip_preserves_state_digest(self):
        reservoir = SpikingReservoir(self._config())
        reservoir.run([[0.2], [0.7], [0.1]] * 4)
        restored = SpikingReservoir.from_snapshot(reservoir.snapshot())
        self.assertEqual(reservoir.digest(), restored.digest())
        self.assertEqual(reservoir.weights_snapshot(), restored.weights_snapshot())

    def test_snapshot_resume_matches_uninterrupted_execution_and_reset_is_clean(self):
        config = self._config(noise_std=0.2)
        full = SpikingReservoir(config)
        all_frames = [[0.4], [0.9], [0.2], [0.7], [0.1], [0.6]]
        expected = full.run(all_frames)
        split = SpikingReservoir(config)
        split.run(all_frames[:3])
        restored = SpikingReservoir.from_snapshot(split.snapshot())
        actual = split.run(all_frames[3:])
        resumed = restored.run(all_frames[3:])
        self.assertEqual(expected[3:], actual)
        self.assertEqual(actual, resumed)
        split.reset()
        fresh = SpikingReservoir(config)
        self.assertEqual(split.digest(), fresh.digest())


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.reservoir = SpikingReservoir(ReservoirConfig(neuron_count=4, input_channels=2, seed=3, noise_std=0.0))
        self.policy = SafetyPolicy(
            max_amplitude=1.0,
            max_duration_ms=2.0,
            min_inter_pulse_ms=1.0,
            max_pulses=4,
            channel_count=2,
        )

    def test_safety_policy_fails_closed_for_invalid_pulses(self):
        invalid = [
            StimulationPulse(0.0, 0, 1.1, 1.0),
            StimulationPulse(0.0, 0, 0.5, 2.1),
            StimulationPulse(0.0, 2, 0.5, 1.0),
            StimulationPulse(-1.0, 0, 0.5, 1.0),
        ]
        for pulse in invalid:
            with self.assertRaises(ProtocolSafetyError):
                self.policy.validate([pulse]).raise_if_invalid()

    def test_cadence_and_count_limits_are_enforced(self):
        pulses = [StimulationPulse(0.0, 0, 0.5, 0.5), StimulationPulse(0.25, 0, 0.5, 0.5)]
        with self.assertRaises(ProtocolSafetyError):
            self.policy.validate(pulses).raise_if_invalid()
        too_many = [StimulationPulse(float(i * 2), 0, 0.5, 0.5) for i in range(5)]
        with self.assertRaises(ProtocolSafetyError):
            self.policy.validate(too_many).raise_if_invalid()

    def test_closed_loop_run_is_local_and_accepts_valid_protocol(self):
        experiment = ClosedLoopExperiment(self.reservoir, self.policy)
        result = experiment.run([[0.0, 0.0]] * 10, [StimulationPulse(2.0, 1, 0.8, 1.0)])
        self.assertEqual(result.steps, 10)
        self.assertEqual(result.duration_ms, 10.0)
        self.assertIsInstance(result.spikes, list)

    def test_invalid_protocol_is_rejected_before_reservoir_state_changes(self):
        experiment = ClosedLoopExperiment(self.reservoir, self.policy)
        before = self.reservoir.digest()
        with self.assertRaises(ProtocolSafetyError):
            experiment.run([[0.0, 0.0]] * 4, [StimulationPulse(0.0, 0, 2.0, 1.0)])
        self.assertEqual(before, self.reservoir.digest())


class MetricsAndCliTests(unittest.TestCase):
    def test_metrics_are_bounded(self):
        reservoir = SpikingReservoir(ReservoirConfig(neuron_count=4, input_channels=1, seed=1))
        spikes = reservoir.run([[0.0]] * 5)
        summary = summarize_spikes(spikes, neuron_count=4, duration_ms=5.0)
        self.assertGreaterEqual(summary["spike_rate_hz"], 0.0)
        self.assertLessEqual(summary["synchrony_index"], 1.0)
        self.assertGreaterEqual(summary["synchrony_index"], 0.0)

    def test_headless_cli_runs_without_site_packages(self):
        project = Path(__file__).parent
        result = subprocess.run(
            [sys.executable, "-S", "run_demo.py", "--steps", "12", "--seed", "5"],
            cwd=project,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["steps"], 12)
        self.assertEqual(payload["seed"], 5)
        self.assertIn("spike_rate_hz", payload)

    def test_recording_loader_is_read_only_and_strict(self):
        path = Path(__file__).parent / "_test_recording.json"
        path.write_text(json.dumps({"schema_version": 1, "channel_count": 2, "frames": [
            {"timestamp_ms": 0.0, "values": [0.1, 0.2]},
            {"timestamp_ms": 1.0, "values": [0.2, 0.3]},
        ]}), encoding="utf-8")
        try:
            source = RecordedDataSource.from_json(path)
            self.assertEqual(list(source), [(0.1, 0.2), (0.2, 0.3)])
        finally:
            path.unlink()

    def test_protocol_parser_rejects_unknown_fields_and_versions(self):
        from wetware_platform import Protocol

        with self.assertRaises(ProtocolSafetyError):
            Protocol.from_dict({"schema_version": 1, "duration_ms": 1, "pulses": [], "force": True})
        with self.assertRaises(ProtocolSafetyError):
            Protocol.from_dict({"schema_version": 2, "duration_ms": 1, "pulses": []}).schema_version

    def test_manifest_digest_is_canonical(self):
        reservoir = SpikingReservoir(ReservoirConfig(seed=4))
        manifest = RunManifest.create(reservoir, "abc")
        self.assertEqual(manifest.digest(), RunManifest.create(reservoir, "abc").digest())
        self.assertEqual(len(manifest.digest()), 64)


if __name__ == "__main__":
    unittest.main()
