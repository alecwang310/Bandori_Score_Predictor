"""Replay the frozen production Stack recipe on the isolated pre-321 LOO folds.

Run from the repository root: python -m bandori.stack_sigma_calibration
This attaches sigma only. No deployment network weights are changed.
"""
import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from collect_bandori_data import atomic_json
from fresh_predictor.runtime import load_history
from .joint import JointNet
from .pipeline import digest
from .stack_loo_data import prediction
from .stack_loo_experiment import config_for, load_fold, protocol
from .stack_uncertainty import EDGES, METHOD, build_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold-run', type=Path, default=Path('runs/stack_architectures_loo_20261019_v1'))
    parser.add_argument('--output', type=Path, default=Path('runs/stack_sigma_20261019'))
    parser.add_argument('--package', type=Path, default=Path('dist/models/stack'))
    args = parser.parse_args()
    plan = protocol(args.fold_run)
    data = load_history(plan['training_bank'])
    deployment = json.loads((args.package / 'deployment.json').read_text())
    # Recipe fixed to the shipped single-seed model, never the LOO architecture winner.
    assert (deployment['hidden'], deployment['seed'], deployment['epochs'], deployment['gain'],
            deployment['learning_rate'], deployment['weight_decay'], deployment['bias_penalty']) == (8, 1729, 240, 1., .01, .003, 3)
    cfg = replace(config_for(plan), seed=1729, epochs=240, weight_decay=.003)
    assert cfg.learning_rate == .01 and cfg.bias_penalty == 3
    assert plan['loo_event_ids'] == [311, 314, 315, 316, 317, 318, 319, 320]
    assert not set(data.events) & set(range(321, 327))
    args.output.mkdir(parents=True, exist_ok=True)
    rows, folds = [], []
    for held in plan['loo_event_ids']:
        arrays, train, valid, meta = load_fold(args.fold_run, f'loo_{held}')
        assert held not in meta['reference_event_ids'] and held not in meta['train_case_events']
        assert not any(c.event_id == held or held in c.references for c in train)
        assert len(meta['train_case_events']) == 29
        model = JointNet(8, 1729).fit(arrays['x'], arrays['targets'], arrays['weights'], arrays['bins'], cfg)
        p = prediction(valid, arrays['valid_index'], model.predict(arrays['valid_x']))
        # Independent parity against the already-frozen seed-specific research replay.
        focal = [(v / data.labels[(held, c.tier)] - 1) for c, v in zip(valid, p)
                 if c.tier in (500, 1000, 1500, 2000) and (held, c.tier) in data.labels]
        cached = json.loads((args.fold_run / 'loo' / f'mlp8_standard_{held}.json').read_text())
        # The generic MLP's multiplication order differs at floating-point scale.
        metric_difference = abs(float(np.mean(np.abs(focal))) - cached['seed_metrics'][0]['overall']['mape'])
        assert metric_difference < 1e-8
        for c, value in zip(valid, p):
            if (held, c.tier) not in data.labels:
                continue
            actual = data.labels[(held, c.tier)]
            if actual <= 0:
                continue
            assert data.label_available(held, cfg) <= deployment['fit_cutoff']
            assert c.cutoff < data.events[held].end
            rows.append(dict(event_id=held, tier=c.tier, checkpoint=c.cutoff,
                hours_remaining=c.remaining_hours, prediction=math.ceil(value), actual=actual,
                event_end=data.events[held].end,
                event_duration_hours=(data.events[held].end-data.events[held].start)/3600))
        model.save(args.output / f'loo_{held}_model.npz')
        folds.append(dict(held=held, training_event_ids=meta['train_case_events'],
            reference_event_ids=meta['reference_event_ids'],
            fold_manifest_sha256=digest(args.fold_run / 'features' / f'loo_{held}' / 'manifest.json'),
            frozen_seed_metric_difference=metric_difference))
        print(f'LOO {held}: calibrated residuals; frozen seed parity within 1e-8', flush=True)
    residuals = args.output / 'residuals.csv'
    with residuals.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    durations = [r['event_duration_hours'] for r in rows]
    calibration = dict(version=1, point_model='stack', method=METHOD,
        source_model_sha256=digest(args.package / 'joint_model.npz'),
        source_protocol_sha256=digest(args.fold_run / 'protocol.json'),
        residuals_sha256=digest(residuals),
        calibration_event_ids=plan['loo_event_ids'], excluded_event_ids=list(range(321, 327)),
        available_at=deployment['fit_cutoff'], characteristic_days=730,
        post_rule_multiplier=3, minimum_events=4, max_score_age_hours=1.,
        checkpoint_hours_cn=[8, 20, 21],
        event_duration_range_hours=[min(durations), max(durations)],
        horizon_edges_hours=[h if math.isfinite(h) else None for h in EDGES],
        formula='r=(prediction-truth)/prediction; sigma=prediction*sqrt(age-weighted event mean(r^2))',
        cells=build_cells(rows, cfg.tiers, deployment['fit_cutoff']),
        limitations=[
            'Eight actual post-rule events (311, 314-320); each held event is removed from fold training and all references',
            'LOO can train on later pre-321 events: event transfer estimate, not a chronological backtest',
            'Model-family design used earlier inspected experiments; this is approximate calibration, not an independent coverage test',
            'Events 321-326 excluded from residual calibration; 321-324 remain in the production network fit',
            'RMS includes systematic bias; no normality or 68% coverage guarantee; proxy closing labels are not official finals',
            'Same-tier neighboring horizon bins are pooled until four event clusters; no cross-tier borrowing',
            'No sigma for unlabeled tiers or observations older than one hour; no event-type-specific or outage calibration',
            'Residual checkpoints are 08:00/20:00/21:00 CN; other issue times use the same hours-remaining buckets',
            'Production has more post-rule training events; sigma is not automatically reduced',
            'Eight events provide weak support for very long events; extrapolation is explicitly flagged'])
    atomic_json(args.output / 'sigma.json', calibration)
    atomic_json(args.output / 'audit.json', dict(folds=folds, rows=len(rows),
        calibration_event_ids=plan['loo_event_ids'], test_and_future_events_absent=True,
        training_outcome_poisoning_audit=json.loads((args.fold_run / 'preparation_audit.json').read_text()),
        deployment_model_sha256=calibration['source_model_sha256'],
        held_labels_used_only_for_residuals=True,
        source_code_sha256={name:digest(name) for name in
            ('bandori/stack_sigma_calibration.py','bandori/stack_uncertainty.py')}))
    # Attach without modifying any fitted weights or reference arrays.
    atomic_json(args.package / 'sigma.json', calibration)
    deployment['hashes']['sigma.json'] = digest(args.package / 'sigma.json')
    atomic_json(args.package / 'deployment.json', deployment)
    manifest = json.loads((args.package / 'manifest.json').read_text())
    manifest['assets'].update({name:digest(args.package / name) for name in ('sigma.json','deployment.json')})
    manifest['limitations'] = [s for s in manifest['limitations'] if 'Sigma is uncalibrated' not in s]
    for note in calibration['limitations']:
        if note not in manifest['limitations']:
            manifest['limitations'].append(note)
    manifest['training_provenance']['sigma'] = dict(method=METHOD, run=str(args.output),
        calibration_event_ids=plan['loo_event_ids'], excluded_event_ids=list(range(321, 327)))
    atomic_json(args.package / 'manifest.json', manifest)
    routing_path = args.package.parents[1] / 'ROUTING.json'
    if routing_path.exists():
        routing = json.loads(routing_path.read_text())
        routing['stack_deployment_sha256'] = digest(args.package / 'deployment.json')
        routing['stack_sigma_sha256'] = digest(args.package / 'sigma.json')
        atomic_json(routing_path, routing)
    print(f'Attached sigma: {len(rows)} residuals, {len(calibration["cells"])} tier/horizon cells')


if __name__ == '__main__':
    main()
