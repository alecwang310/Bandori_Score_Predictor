#!/usr/bin/env python3
"""Stack parity self-check: dist StackPredictor vs harness and joint_runtime.

Reproduce from the repo root (the research library is required on this side
only — inference inside dist/ never imports it):

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python dist/tools/verify_stack_parity.py

For each example (events 325/326 at the snapshot cutoff) the forecast is
re-derived from dist's frozen package twice and compared against the saved
dist output (/tmp/new_out/stack_<event>.json, override with --out-dir):

  harness   restrict -> attach_bank -> RoleAwareAnalog cases at the cutoff;
            ridge_inputs vs finish_context(make_context(...)) feature parity
            (allclose atol=1e-10); block_cols({'rule','expl'}) equals the
            package's recorded pca_columns; JointNet.predict on those 600
            columns -> correction_forecast(..., 1.0, 'nn') -> ceil.

  joint     the frozen weights zero-padded to the full 1168-wide layout in a
            throwaway /tmp package (mean 0 / scale 1 / a 0 outside the
            selected columns, mathematically identical), then run through
            bandori.joint_runtime.JointPredictor unchanged.

Exit 0 only when every compared tier matches on both paths for both events.
"""
import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from bandori.config import Config
from bandori.experiments import correction_forecast
from bandori.joint import JointNet
from bandori.joint_runtime import JointPredictor
from bandori.live_data import load_snapshot
from bandori.pipeline import attach_bank, digest
from bandori.regime_suite import finish_context, make_context
from bandori.representations import Compression
from bandori.ridge_runtime import ridge_inputs
from bandori.role_analog import RoleAwareAnalog
from bandori.sweep_features import block_cols
from fresh_predictor.data import restrict

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'dist' / 'models' / 'stack'
EXAMPLES = {
    325: ROOT / 'dist' / 'examples' / 'event_325_20261014_2000.json.gz',
    326: ROOT / 'dist' / 'examples' / 'event_326_20261017_2000.json.gz',
}
FOCAL = (500, 1000, 1500, 2000)


class _Masks(dict):
    """finish_context masks: the package 'pca' mask, 1.0 for unused matrices."""
    def __missing__(self, key):
        return 1.0


def padded_package(target, deployment, cols, width):
    """Zero-pad the 600-col weights to the full layout (identity on outputs)."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for path in sorted(PKG.iterdir()):
        if path.is_file() and path.name not in ('joint_model.npz', 'deployment.json'):
            shutil.copy(path, target / path.name)
    with np.load(PKG / 'joint_model.npz', allow_pickle=False) as arrays:
        net = {k: arrays[k].copy() for k in arrays.files}
    mean = np.zeros(width)
    scale = np.ones(width)
    a = np.zeros((width, len(net['b'])))
    mean[cols] = net['mean']
    scale[cols] = net['scale']
    a[cols] = net['a']
    np.savez_compressed(target / 'joint_model.npz', mean=mean, scale=scale, a=a,
                        b=net['b'], v=net['v'], d=net['d'], trained=net['trained'])
    manifest = dict(deployment)
    manifest['hashes'] = dict(deployment['hashes'])
    manifest['hashes']['joint_model.npz'] = digest(target / 'joint_model.npz')
    (target / 'deployment.json').write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return target


def check_event(eid, out_dir, padded_root):
    deployment = json.loads((PKG / 'deployment.json').read_text())
    assert deployment['gain'] == 1.0
    dist_doc = json.loads((out_dir / f'stack_{eid}.json').read_text())
    assert dist_doc['model']['name'] == 'stack'
    dist_finals = {r['tier']: r['final'] for r in dist_doc['predictions'] if r['status'] == 'ok'}
    dist_failed = {r['tier'] for r in dist_doc['predictions'] if r['status'] != 'ok'}
    cutoff = datetime.fromisoformat(dist_doc['checkpoint_cn']).timestamp()

    live = load_snapshot(EXAMPLES[eid])
    target = restrict(live, [eid], labels=False)
    config = Config.load(PKG / 'config.json')
    compression = Compression.load(PKG / 'compression.npz')
    with np.load(PKG / 'input_masks.npz', allow_pickle=False) as a:
        pca_mask = a['pca'].copy()
    ids = deployment['training_event_ids']
    data = attach_bank(PKG, target)
    event = data.known_event(data.events[eid], cutoff)[0]
    analog = RoleAwareAnalog(data, config, ids)
    cases, failed = [], []
    for tier in config.tiers:
        try:
            cases.append(analog.build(event, tier, cutoff))
        except ValueError as error:
            failed.append((tier, str(error)))
    assert cases, 'no usable cases at the cutoff'
    assert {t for t, _ in failed} == dist_failed, f'unavailable tiers differ: {failed} vs {dist_failed}'

    # --- harness path -------------------------------------------------------
    x_rt, index = ridge_inputs(cases, data, config, ids, compression, pca_mask)
    ctx = finish_context(make_context(cases, data, config, ids), compression,
                         _Masks(pca=pca_mask))[0]
    assert (index == ctx['index']).all(), 'index parity failed'
    assert np.allclose(x_rt, ctx['matrices']['pca'], atol=1e-10), \
        'runtime/harness feature parity failed'
    cols = np.asarray(block_cols({'rule', 'expl'}), dtype=int)
    assert np.array_equal(cols, np.asarray(deployment['pca_columns'], dtype=int)), \
        "block_cols({'rule','expl'}) != package pca_columns"
    assert cols.shape[0] == 600 and cols[0] == 16 and cols[-1] == 1167
    core = JointNet.load(PKG / 'joint_model.npz')
    delta = core.predict(np.ascontiguousarray(x_rt[:, cols]))
    values = correction_forecast(cases, index, delta, deployment['gain'], 'nn')
    harness_finals = {c.tier: math.ceil(values[i]) for i, c in enumerate(cases)}

    # --- joint_runtime path (padded weights) -------------------------------
    padded = padded_package(padded_root, deployment, cols, int(deployment['pca_layout']['total_columns']))
    native = JointPredictor(padded).predict(target, eid, cutoff)
    joint_finals = {r['tier']: r['final'] for r in native['predictions'] if r['status'] == 'ok'}

    # --- compare ------------------------------------------------------------
    problems = []
    print(f'--- event {eid} @ {dist_doc["checkpoint_cn"]} '
          f'({len(dist_finals)} ok tiers, cutoff {int(cutoff)})')
    for name, finals in (('harness', harness_finals), ('joint_runtime', joint_finals)):
        mismatches = {t: (dist_finals[t], finals.get(t)) for t in sorted(dist_finals)
                      if finals.get(t) != dist_finals[t]}
        extra = sorted(set(finals) - set(dist_finals))
        status = 'OK' if not mismatches and not extra else 'MISMATCH'
        print(f'    {name:14s} {len(dist_finals) - len(mismatches)}/{len(dist_finals)} tiers match  [{status}]')
        if mismatches:
            print(f'      (dist, {name}): ' + ', '.join(f'T{t}={d} vs {h}'
                                                        for t, (d, h) in mismatches.items()))
            problems.append(f'event {eid} {name}: {len(mismatches)} mismatched tiers')
        if extra:
            print(f'      extra {name} tiers: {extra}')
            problems.append(f'event {eid} {name}: extra tiers {extra}')
    print('    focal finals (dist / harness / joint_runtime):')
    for tier in FOCAL:
        if tier in dist_finals:
            print(f'      T{tier}: {dist_finals[tier]} / {harness_finals.get(tier)} / '
                  f'{joint_finals.get(tier)}')
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', default='/tmp/new_out',
                   help='directory with dist stack_<event>.json outputs (default /tmp/new_out)')
    p.add_argument('--events', nargs='+', type=int, default=[325, 326])
    p.add_argument('--padded-dir', default='/tmp/stack_padded_parity',
                   help='throwaway padded package for the joint_runtime path')
    a = p.parse_args(argv)
    out_dir = Path(a.out_dir)
    problems = []
    for eid in a.events:
        if eid not in EXAMPLES:
            problems.append(f'no example snapshot registered for event {eid}')
            continue
        if not (out_dir / f'stack_{eid}.json').exists():
            problems.append(f'missing dist output {out_dir}/stack_{eid}.json')
            continue
        problems += check_event(eid, out_dir, Path(a.padded_dir) / f'event_{eid}')
    if problems:
        print('\nFAIL:', *problems, sep='\n  ', file=sys.stderr)
        return 1
    print('\nOK: dist stack finals match harness and joint_runtime on every example')
    return 0


if __name__ == '__main__':
    sys.exit(main())
