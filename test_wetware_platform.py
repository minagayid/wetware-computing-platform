import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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
        with self.assertRaisesRegex(ValueError, "local simulation limit"):
            ReservoirConfig(neuron_count=513)
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

    def test_snapshot_is_independent_of_later_model_mutations(self):
        reservoir = SpikingReservoir(self._config())
        snapshot = reservoir.snapshot()
        initial_membrane = list(snapshot["membrane"])
        reservoir.run([[1.0], [0.5]])
        self.assertEqual(snapshot["step_index"], 0)
        self.assertEqual(snapshot["membrane"], initial_membrane)

    def test_snapshot_rejects_overlarge_weight_row_before_copying(self):
        reservoir = SpikingReservoir(self._config(neuron_count=2, input_channels=1))
        snapshot = reservoir.snapshot()
        snapshot["weights"][0] = {"0": 0.2, "1": 0.4}
        with self.assertRaisesRegex(ValueError, "malformed or too large"):
            SpikingReservoir.from_snapshot(snapshot)

    def test_run_rejects_unbounded_frame_streams(self):
        reservoir = SpikingReservoir(self._config())
        with patch("wetware_platform.model.MAX_SIMULATION_FRAMES", 1):
            with self.assertRaisesRegex(ValueError, "frame limit"):
                reservoir.run([[0.0], [0.0]])
        with patch("wetware_platform.model.MAX_SIMULATION_SAMPLES", 1):
            with self.assertRaisesRegex(ValueError, "sample limit"):
                reservoir.run([[0.0], [0.0]])

    def test_operation_budget_accounts_for_recurrent_edges_before_state_changes(self):
        reservoir = SpikingReservoir(self._config(connection_probability=1.0))
        before = reservoir.digest()
        with patch("wetware_platform.model.MAX_SIMULATION_OPERATIONS", 10):
            with self.assertRaisesRegex(ValueError, "operation limit"):
                reservoir.run([[0.0]])
        self.assertEqual(reservoir.digest(), before)

    def test_run_rolls_back_state_when_spike_output_limit_is_exceeded(self):
        reservoir = SpikingReservoir(self._config())
        reservoir.input_weights = [[100.0] for _ in range(reservoir.config.neuron_count)]
        before = reservoir.digest()
        with patch("wetware_platform.model.MAX_RECORDED_SPIKES", 0):
            with self.assertRaisesRegex(ValueError, "spike output limit"):
                reservoir.run([[100.0]])
        self.assertEqual(reservoir.digest(), before)

    def test_step_stops_reading_an_oversized_or_unbounded_input_frame(self):
        reservoir = SpikingReservoir(self._config())
        consumed = []

        def values():
            for index in range(100):
                consumed.append(index)
                yield 0.0

        with self.assertRaisesRegex(ValueError, "expected 1 input channels"):
            reservoir.step(values())
        self.assertEqual(consumed, [0, 1])

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
            StimulationPulse(0.0, 0.0, 0.5, 1.0),
            StimulationPulse(0.0, "0", 0.5, 1.0),
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

    def test_protocol_duration_and_policy_limits_must_be_finite(self):
        for duration in (-1.0, float("nan"), float("inf")):
            with self.subTest(duration=duration):
                with self.assertRaises(ProtocolSafetyError):
                    self.policy.validate([], duration).raise_if_invalid()

        self.assertTrue(self.policy.validate([], 0.0).valid)
        for limits in (
            {"max_amplitude": float("nan")},
            {"max_duration_ms": float("inf")},
            {"max_total_exposure": float("nan")},
        ):
            with self.subTest(limits=limits):
                with self.assertRaises(ProtocolSafetyError):
                    SafetyPolicy(**limits).validate([]).raise_if_invalid()

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

    def test_non_integer_channel_values_fail_closed_before_state_changes(self):
        experiment = ClosedLoopExperiment(self.reservoir, self.policy)
        before = self.reservoir.digest()
        for channel in (0.0, "0"):
            with self.subTest(channel=channel):
                with self.assertRaises(ProtocolSafetyError):
                    experiment.run([[0.0, 0.0]], [StimulationPulse(0.0, channel, 0.5, 0.5)])
                self.assertEqual(self.reservoir.digest(), before)

    def test_closed_loop_run_bounds_frame_and_pulse_iterables_before_materializing(self):
        experiment = ClosedLoopExperiment(self.reservoir, self.policy)
        consumed_frames = []

        def frame_stream():
            for index in range(20):
                consumed_frames.append(index)
                yield [0.0, 0.0]

        with patch("wetware_platform.experiment.MAX_SIMULATION_FRAMES", 2):
            with self.assertRaisesRegex(ValueError, "frame limit"):
                experiment.run(frame_stream())
        self.assertLess(len(consumed_frames), 20)
        self.assertEqual(self.reservoir.step_index, 0)

        consumed_pulses = []

        def pulse_stream():
            for index in range(20):
                consumed_pulses.append(index)
                yield StimulationPulse(0.0, 0, 0.1, 0.1)

        with patch("wetware_platform.experiment.MAX_PROTOCOL_PULSES", 1):
            with self.assertRaisesRegex(ProtocolSafetyError, "parse limit"):
                experiment.run([], pulse_stream())
        self.assertLess(len(consumed_pulses), 20)

    def test_closed_loop_run_enforces_combined_sample_budget_before_materializing(self):
        experiment = ClosedLoopExperiment(self.reservoir, self.policy)
        consumed = []

        def frame_stream():
            for index in range(20):
                consumed.append(index)
                yield [0.0, 0.0]

        with patch("wetware_platform.experiment.MAX_SIMULATION_SAMPLES", 2):
            with self.assertRaisesRegex(ValueError, "sample limit"):
                experiment.run(frame_stream())
        self.assertEqual(consumed, [0, 1])
        self.assertEqual(self.reservoir.step_index, 0)

    def test_closed_loop_run_rejects_non_finite_frame_before_model_mutation(self):
        experiment = ClosedLoopExperiment(self.reservoir, self.policy)
        before = self.reservoir.digest()
        with self.assertRaisesRegex(ValueError, "input values must be finite"):
            experiment.run([[0.0, 0.0], [float("nan"), 0.0]])
        self.assertEqual(self.reservoir.digest(), before)

    def test_closed_loop_rejects_pulse_sum_overflow_before_model_mutation(self):
        policy = SafetyPolicy(
            max_amplitude=1e308,
            max_duration_ms=1.0,
            min_inter_pulse_ms=0.0,
            max_pulses=1,
            channel_count=2,
            max_total_exposure=1e308,
        )
        experiment = ClosedLoopExperiment(self.reservoir, policy)
        before = self.reservoir.digest()
        pulse = StimulationPulse(1.0, 0, 1e308, 0.5)
        with self.assertRaisesRegex(ValueError, "enriched input values must be finite"):
            experiment.run([[0.0, 0.0], [1e308, 0.0]], [pulse])
        self.assertEqual(self.reservoir.digest(), before)


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

    def test_recording_loader_rejects_unbounded_frame_counts(self):
        path = Path(__file__).parent / "_test_recording.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "channel_count": 1,
            "frames": [{"timestamp_ms": 0, "values": [0]}] * 100_001,
        }), encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "frame or sample limit"):
                RecordedDataSource.from_json(path)
        finally:
            path.unlink()

    def test_protocol_parser_rejects_unknown_fields_and_versions(self):
        from wetware_platform import Protocol

        malformed = [
            None,
            [],
            {"schema_version": 1, "duration_ms": 1, "pulses": {}},
            {"schema_version": 1, "duration_ms": None, "pulses": []},
            {"schema_version": 1, "duration_ms": 1, "pulses": [None]},
            {"schema_version": 1, "duration_ms": 1, "pulses": [{"start_ms": 0}]},
        ]
        for payload in malformed:
            with self.subTest(payload=payload):
                with self.assertRaises(ProtocolSafetyError):
                    Protocol.from_dict(payload)
        with self.assertRaises(ProtocolSafetyError):
            Protocol.from_dict({"schema_version": 1, "duration_ms": 1, "pulses": [], "force": True})
        with self.assertRaises(ProtocolSafetyError):
            Protocol.from_dict({"schema_version": 2, "duration_ms": 1, "pulses": []}).schema_version
        with self.assertRaisesRegex(ProtocolSafetyError, "parse limit"):
            Protocol.from_dict({
                "schema_version": 1,
                "duration_ms": 1,
                "pulses": [{"start_ms": 0, "channel": 0, "amplitude": 0, "duration_ms": 1}] * 1001,
            })
        with patch("wetware_platform.protocol.MAX_PROTOCOL_PULSES", 1):
            with patch("wetware_platform.protocol._finite_number", wraps=__import__("wetware_platform.protocol", fromlist=["_finite_number"])._finite_number) as finite_number:
                with self.assertRaisesRegex(ProtocolSafetyError, "parse limit"):
                    Protocol.from_dict({
                        "schema_version": 1,
                        "duration_ms": 1,
                        "pulses": [
                            {"start_ms": 0, "channel": 0, "amplitude": 0, "duration_ms": 1},
                            {"start_ms": 1, "channel": 0, "amplitude": 0, "duration_ms": 1},
                        ],
                    })
                self.assertEqual(finite_number.call_count, 1)
        with self.assertRaises(ProtocolSafetyError):
            Protocol.from_dict({"schema_version": 1, "duration_ms": 10 ** 10000, "pulses": []})
        with self.assertRaisesRegex(ProtocolSafetyError, "pulse count"):
            Protocol(10.0, (StimulationPulse(0, 0, 0.5, 1.0),) * 1001)

    def test_safety_policy_handles_malformed_values_without_type_errors(self):
        for policy in (
            SafetyPolicy(max_amplitude=None),
            SafetyPolicy(max_pulses=None),
            SafetyPolicy(channel_count=None),
        ):
            with self.subTest(policy=policy):
                with self.assertRaises(ProtocolSafetyError):
                    policy.validate([]).raise_if_invalid()
        malformed_pulse_policy = SafetyPolicy()
        with self.assertRaises(ProtocolSafetyError):
            malformed_pulse_policy.validate([StimulationPulse("bad", 0, 0.5, 1.0)]).raise_if_invalid()
        with self.assertRaises(ProtocolSafetyError):
            malformed_pulse_policy.validate(None).raise_if_invalid()

    def test_manifest_digest_is_canonical(self):
        reservoir = SpikingReservoir(ReservoirConfig(seed=4))
        manifest = RunManifest.create(reservoir, "abc")
        self.assertEqual(manifest.digest(), RunManifest.create(reservoir, "abc").digest())
        self.assertEqual(len(manifest.digest()), 64)


if __name__ == "__main__":
    unittest.main()
