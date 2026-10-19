from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

# bandori.lstm sets a libomp compatibility env flag before importing torch;
# keep that import first so conda interpreters do not abort at import time.
from bandori.lstm import RawJointLSTM, relative_objective
import torch

from bandori.raw_sequence import (forecast_event, pack_streams, snapshot, stream,
                                  training_streams)
from fixtures import config, dataset


def tiny_bundle(data, cfg, fit_cutoff, ids=None):
    ids = ids if ids is not None else [e.event_id for e in data.eligible_events(cfg)][:3]
    streams = training_streams(data, ids, cfg.tiers)
    return pack_streams(streams, data, cfg, fit_cutoff)


class RelativeObjectiveTests(unittest.TestCase):
    def test_autograd_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        raw = torch.tensor(rng.normal(size=(7, 2)), dtype=torch.float64, requires_grad=True)
        target = torch.tensor(np.stack((rng.uniform(5, 10, (7, 2)), rng.uniform(1, 4, (7, 2)),
                                        rng.uniform(11, 14, (7, 2))), axis=-1), dtype=torch.float64)
        weights = torch.tensor(rng.uniform(0, 1, (7, 2)), dtype=torch.float64)
        bins = torch.tensor([0, 0, 1, 2, 3, 4, 4], dtype=torch.long)
        loss = relative_objective(raw, target, weights, bins, 3.0)
        loss.backward()
        for index in ((0, 0), (3, 1), (6, 0)):
            before = raw[index].item()
            epsilon = 1e-6
            raw.data[index] = before + epsilon
            high = relative_objective(raw, target, weights, bins, 3.0)
            raw.data[index] = before - epsilon
            low = relative_objective(raw, target, weights, bins, 3.0)
            raw.data[index] = before
            self.assertAlmostEqual(float((high - low) / (2 * epsilon)), float(raw.grad[index]), places=5)

    def test_bounded_correction_matches_linear_anchor_when_untouched(self):
        # delta = 0 must reproduce current + linear close anchor exactly.
        raw = torch.zeros(4, 2, dtype=torch.float64)
        target = torch.tensor([[[100., 50., 200.], [10., 2., 40.]]] * 4, dtype=torch.float64)
        weights = torch.ones(4, 2, dtype=torch.float64)
        bins = torch.zeros(4, dtype=torch.long)
        loss = relative_objective(raw, target, weights, bins, 0.0)
        expected = np.mean(((np.array([100., 10.]) + np.array([50., 2.]) - np.array([200., 40.]))
                            / np.array([200., 40.])) ** 2)
        self.assertAlmostEqual(float(loss), expected, places=10)


class RawLSTMTests(unittest.TestCase):
    def test_fit_reduces_loss_and_untrained_tiers_stay_zero(self):
        data = dataset(12);cfg = config();ids = [e.event_id for e in data.eligible_events(cfg)][:3]
        fit_cutoff = max(data.events[e].end for e in ids) + 86400
        bundle, target, weights, bins = tiny_bundle(data, cfg, fit_cutoff, ids)
        weights[..., 1] = 0  # tier 1500 has no supervised rows
        losses = []
        model = RawJointLSTM(4, 3, seed=5, device='cpu').fit(
            bundle, target, weights, bins, epochs=6, learning_rate=.02,
            progress=lambda step, loss: losses.append(loss))
        self.assertLess(losses[-1], losses[0])
        corrections = model.predict_corrections(bundle)
        self.assertTrue(np.all(corrections[..., 1] == 0))
        self.assertTrue(np.isfinite(corrections).all())

    def test_save_load_roundtrip_and_device(self):
        data = dataset(12);cfg = config();ids = [e.event_id for e in data.eligible_events(cfg)][:2]
        fit_cutoff = max(data.events[e].end for e in ids) + 86400
        bundle, target, weights, bins = tiny_bundle(data, cfg, fit_cutoff, ids)
        model = RawJointLSTM(4, 3, seed=5, device='cpu').fit(bundle, target, weights, bins, epochs=2)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'lstm.pt'
            model.save(path)
            other = RawJointLSTM.load(path, device='cpu')
            np.testing.assert_array_equal(model.predict_corrections(bundle), other.predict_corrections(bundle))
            self.assertTrue(torch.equal(model.trained, other.trained))

    def test_forecast_is_publication_causal_and_deterministic(self):
        data = dataset(12);cfg = config();ids = [e.event_id for e in data.eligible_events(cfg)][:2]
        fit_cutoff = max(data.events[e].end for e in ids) + 86400
        bundle, target, weights, bins = tiny_bundle(data, cfg, fit_cutoff, ids)
        model = RawJointLSTM(4, 3, seed=5, device='cpu').fit(bundle, target, weights, bins, epochs=2)
        event = data.events[ids[0]]
        cutoff = event.start + 36 * 3600 + 1234  # deliberately off-grid
        first, details = forecast_event(model, data, event, cfg.tiers, np.array([cutoff]))
        second, _ = forecast_event(model, data, event, cfg.tiers, np.array([cutoff]))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(details['present'].all())
        # Everything published after the query time must be invisible.
        for tier in cfg.tiers:
            curve = data.curves[(event.event_id, tier)]
            curve.scores[curve.times > cutoff] *= 10000
            data.labels[(event.event_id, tier)] *= 10000
        later, _ = forecast_event(model, data, event, cfg.tiers, np.array([cutoff]))
        np.testing.assert_array_equal(first, later)
        # A late-published score measured before the query time cannot enter either.
        data2 = dataset(12)
        for tier in cfg.tiers:
            curve = data2.curves[(event.event_id, tier)]
            mask = (curve.times > cutoff - 4 * 3600) & (curve.times <= cutoff)
            curve.available[mask] = cutoff + 86400
            curve.scores[mask] *= 7
        stale, _ = forecast_event(model, data2, event, cfg.tiers, np.array([cutoff]))
        for tier in cfg.tiers:
            curve = data2.curves[(event.event_id, tier)]
            mask = (curve.times > cutoff - 4 * 3600) & (curve.times <= cutoff)
            curve.scores[mask] /= 7
        clean, _ = forecast_event(model, data2, event, cfg.tiers, np.array([cutoff]))
        np.testing.assert_array_equal(stale, clean)

    def test_forecast_monotone_and_floored_at_current(self):
        data = dataset(12);cfg = config();ids = [e.event_id for e in data.eligible_events(cfg)][:2]
        fit_cutoff = max(data.events[e].end for e in ids) + 86400
        bundle, target, weights, bins = tiny_bundle(data, cfg, fit_cutoff, ids)
        model = RawJointLSTM(4, 3, seed=5, device='cpu').fit(bundle, target, weights, bins, epochs=2)
        event = data.events[ids[0]]
        for cutoff in (event.start + 3600, event.start + 50.5 * 3600, event.end - 3600):
            values, details = forecast_event(model, data, event, cfg.tiers, np.array([cutoff]))
            self.assertTrue(np.all(values[0] >= details['current'][0] - 1e-6))
            self.assertTrue(values[0, 0] >= values[0, 1] - 1e-6)  # T500 >= T1500

    def test_pack_streams_normalizes_weights_and_rejects_unavailable_labels(self):
        data = dataset(12);cfg = config();ids = [e.event_id for e in data.eligible_events(cfg)][:3]
        fit_cutoff = max(data.events[e].end for e in ids) + 86400
        _, _, weights, bins = tiny_bundle(data, cfg, fit_cutoff, ids)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        for group in range(5):
            self.assertLessEqual(float(weights[bins == group].sum()), 1.0 + 1e-6)
        eid = ids[0]
        original = data.events[eid]
        data.events[eid] = replace(original, end=original.end + 30 * 86400)
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            tiny_bundle(data, cfg, fit_cutoff, ids)
        data.events[eid] = original


class LSTMExperimentTests(unittest.TestCase):
    def test_experiment_training_and_saved_inference(self):
        from bandori.experiments import run_suite
        from bandori.lstm_experiment import predict, train
        data = dataset(24)
        cfg = config(epochs=2, hidden_sizes=(2,), characteristic_time_days=730)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td);registry = root / 'ledger.json'
            with redirect_stdout(io.StringIO()):
                run_suite(data, cfg, root / 'control', registry)
            ledger = registry.read_bytes()
            with redirect_stdout(io.StringIO()):
                evaluation = train(data, root / 'control', root / 'lstm', registry,
                                   device='cpu', hidden=4, decoder=3, epochs=2)
            self.assertEqual(registry.read_bytes(), ledger)
            self.assertEqual(evaluation['test_status'], 'previously_inspected_follow_up')
            self.assertIn('lstm', evaluation['partitions']['test'])
            self.assertIn('control', evaluation['partitions']['test'])
            champion = evaluation['champion']
            self.assertIsNotNone(evaluation['paired_bootstrap'])
            rows = (root / 'lstm/predictions.csv').read_text().splitlines()
            self.assertGreater(len(rows), 1)
            self.assertIn('lstm', rows[0])
            self.assertIn('control_' + champion, rows[0])
            split = json.loads((root / 'lstm/split.json').read_text())
            event = data.events[split['test'][0]]
            cutoff = event.start + 27 * 3600 + 1800  # off-grid is supported
            result = predict(data, root / 'lstm', event.event_id, cutoff, root / 'fixed.json')
            self.assertEqual([r['tier'] for r in result['predictions']], list(cfg.tiers))
            self.assertTrue(all(r['status'] == 'ok' and r['final'] >= r['current']
                                for r in result['predictions']))
            for tier in cfg.tiers:
                data.labels[(event.event_id, tier)] *= 1000
                curve = data.curves[(event.event_id, tier)]
                curve.scores[curve.times > cutoff] *= 1000
            self.assertEqual(result, predict(data, root / 'lstm', event.event_id, cutoff))
            with self.assertRaisesRegex(ValueError, 'already exists'):
                predict(data, root / 'lstm', event.event_id, cutoff, root / 'fixed.json')
            with self.assertRaisesRegex(ValueError, 'Training event'):
                predict(data, root / 'lstm', split['train'][0], cutoff)


if __name__ == '__main__':
    unittest.main()
