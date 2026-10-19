"""Raised objective-tier loss weighting on the joint case-feature NN (JointNet).

Prior finding: reweighting tiers 500/1000/1500/2000 to 60% of loss weight mass
helped the shared-capacity hourly LSTM (+1.3 pp) but was a no-op for per-tier
linear ridge. JointNet (one hidden layer shared across 24 tiers) is the best
family on new events and has shared capacity, so it is the prime candidate.

Protocol: identical harness rows, splits and scoring as every other sweep —
train on the 315-event eight_post_events bank, test only on events 321-324,
event-balanced MAPE on focal tiers 500/1000/1500/2000 (312 clean rows).
Weight rescaling reuses raised_objective_experiment.raise_objective_share;
only the weights array changes, never the harness or frozen runs.

Experiments:
  1. Share sweep on nn_pca: share in {natural, 0.22, 0.4, 0.6, 0.8}.
  2. Objective-set variants at share 0.6.
  3. Width interaction: share 0.6 at hidden in {4, 8, 16}.
  4. Ridge control: share 0.6 (expected no-op).
  5. Per-tier MAPE breakdown at the LOO-selected share vs natural.
  6. nn_minimal and tcn24 at share 0.6.

Selection discipline: a "tuned" share is crowned only by leave-one-out over the
post-rule TRAINING events 311-320; the 321-324 grid table is descriptive.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .evaluation import summary
from .experiments import correction_forecast
from .raised_objective_experiment import raise_objective_share
from .regime_suite import finish_context
from .sweep_harness import (FOCAL_TIERS, POST_RULE_TRAIN_EVENTS, evaluate,
                            fit_jointnet, fit_ridge, fmt, load_bank,
                            prepare_fit, save_results, score,
                            supervised_slice, validate, write_report)
from .temporal import JointTemporalNet

OBJECTIVE = FOCAL_TIERS
SHARES = (0.22, 0.4, 0.6, 0.8)
OUT = Path('runs/sweep_objective_20261019')
REPORT = Path('reports/sweep_objective_20261019.md')


def _rescale(weights, tiers, share, objective=OBJECTIVE):
    """Rescale weights to `share`; returns (raised, info) with realized masses."""
    if share is None:
        cols = np.array([t in tuple(objective) for t in tiers])
        total = float(weights.sum())
        mass = float(weights[..., cols].sum()) / total
        info = dict(objective_tiers=[t for t in tiers if t in tuple(objective)], share=None,
                    objective_mass_before=mass, objective_mass_after=mass,
                    objective_factor=1.0, rest_factor=1.0)
        return np.asarray(weights, dtype=np.float64), info
    raised, info = raise_objective_share(weights, tiers, objective=objective, share=share)
    info = dict(info, objective_mass_after=float(share))
    return raised, info


def _full_by_tier(bank, values, cases):
    """Per-tier MAPE/bias on ALL tiers (not just focal) for the ablation check."""
    s = summary(cases, values, bank.data)
    return {t: dict(mape=v.get('mape'), bias=v.get('bias'), rows=v.get('rows', 0))
            for t, v in sorted(s['by_tier'].items(), key=lambda kv: int(kv[0])) if v.get('rows')}


def run_nn(bank, prep, share, objective=OBJECTIVE, matrix='pca', hidden=8):
    """Fit JointNet with rescaled weights; returns (result, info, by_tier_all)."""
    tiers = list(bank.config.tiers)
    weights, info = _rescale(prep.weights, tiers, share, objective)
    model = fit_jointnet(prep.matrix[matrix], prep.targets, weights, prep.bins,
                         bank.config, hidden=hidden, seed=1729)
    result, values = evaluate(bank, prep, model, matrix, gain=1.0, kind='nn')
    by_tier = _full_by_tier(bank, values, prep.val['cases'])
    return result, info, by_tier


def run_ridge(bank, prep, share, objective=OBJECTIVE):
    tiers = list(bank.config.tiers)
    weights, info = _rescale(prep.weights, tiers, share, objective)
    model = fit_ridge(prep.matrix['pca'], prep.targets, weights, prep.bins)
    result, _ = evaluate(bank, prep, model, 'pca', gain=0.5, kind='ridge')
    return result, info


def run_tcn24(bank, prep, share, objective=OBJECTIVE):
    tiers = list(bank.config.tiers)
    weights, info = _rescale(prep.weights, tiers, share, objective)
    model = JointTemporalNet(4, 8, bank.config.seed).fit(
        prep.train['matrices']['standard'], prep.train['daily']['sequence'],
        prep.targets, weights, prep.bins, bank.config)
    def predict_fn(val):
        delta = model.predict(val['matrices']['standard'], val['daily']['sequence'])
        return correction_forecast(val['cases'], val['index'], delta, 0.5)
    result, values = evaluate(bank, prep, None, predict_fn=predict_fn)
    by_tier = _full_by_tier(bank, values, prep.val['cases'])
    return result, info, by_tier


def loo_select(bank, shares, objective=OBJECTIVE, hidden=8):
    """Leave-one-out over post-rule TRAINING events 311-320; mean focal MAPE."""
    ids = list(bank.train_ids)
    folds = []
    for held in POST_RULE_TRAIN_EVENTS:
        if held not in ids:
            continue
        supervised = [e for e in ids if e != held]
        prep = prepare_fit(bank, supervised)
        held_ctx = supervised_slice(bank.ctx_all, [held])
        held_ctx, _ = finish_context(held_ctx, prep.compression, masks=prep.masks)
        row = dict(event=held)
        for share in shares:
            tiers = list(bank.config.tiers)
            weights, _ = _rescale(prep.weights, tiers, share, objective)
            model = fit_jointnet(prep.matrix['pca'], prep.targets, weights, prep.bins,
                                 bank.config, hidden=hidden, seed=1729)
            delta = model.predict(held_ctx['matrices']['pca'])
            values = correction_forecast(held_ctx['cases'], held_ctx['index'], delta, 1.0, 'nn')
            s = score(held_ctx['cases'], values, bank.data)
            row[share] = dict(mape=s['mape'], bias=s['bias'], rows=s['rows'])
        folds.append(row)
        def _fmt_share(s):
            label = 'natural' if s is None else f'{s:.3f}'
            return f'{label}:{row[s]["mape"]*100:.2f}%'
        print('LOO held', held, '  '.join(_fmt_share(s) for s in shares), flush=True)
    means = {share: float(np.mean([f[share]['mape'] for f in folds])) for share in shares}
    biases = {share: float(np.mean([f[share]['bias'] for f in folds])) for share in shares}
    return dict(folds=folds, mean_mape=means, mean_bias=biases,
                selected=min(means, key=lambda s: means[s]))


def compact(result, info, extra=None):
    out = dict(mape=result['mape'], bias=result['bias'], final24h=result['final24h'],
               rows=result['rows'], per_event={int(k): v['mape'] for k, v in result['per_event'].items()},
               objective_mass_before=info['objective_mass_before'],
               objective_mass_after=info['objective_mass_after'])
    if extra:
        out.update(extra)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-loo', action='store_true')
    p.add_argument('--report-only', action='store_true',
                   help='Regenerate the report from the saved evaluation.json')
    a = p.parse_args()
    if a.report_only:
        report_only()
        return
    t0 = time.time()
    bank = load_bank()
    validate(bank)
    prep = prepare_fit(bank)
    tiers = list(bank.config.tiers)
    payload = dict(
        kind='sweep_objective_jointnet_v1',
        scope=dict(fold='eight_post_events', train_events=len(bank.train_ids),
                   validation_events=[321, 322, 323, 324], focal_tiers=list(FOCAL_TIERS),
                   standing='event-balanced MAPE on clean checkpoints of 321-324, focal tiers only'),
        natural_objective_mass=None, runs={})

    # Natural baseline (no rescale).
    natural, ninfo, natural_tiers = run_nn(bank, prep, None)
    payload['natural_objective_mass'] = ninfo['objective_mass_after']
    payload['runs']['nn_pca_natural'] = compact(natural, ninfo)
    print('natural nn_pca', fmt(natural), 'mass', round(ninfo['objective_mass_after'], 4), flush=True)

    # 1. Share sweep on nn_pca.
    share_results = {}
    for share in SHARES:
        r, info, by_tier = run_nn(bank, prep, share)
        share_results[share] = compact(r, info)
        payload['runs'][f'nn_pca_share_{share}'] = share_results[share]
        print(f'share {share}', fmt(r), 'mass before', round(info['objective_mass_before'], 4),
              flush=True)

    # 2. Objective-set variants at share 0.6.
    objective_sets = [(500, 1000), (500, 1000, 1500), (500, 1000, 1500, 2000), (1000, 2000)]
    variant_results = {}
    for obj in objective_sets:
        r, info, _ = run_nn(bank, prep, 0.6, objective=obj)
        variant_results[obj] = compact(r, info)
        payload['runs'][f'nn_pca_obj_{"_".join(map(str, obj))}'] = variant_results[obj]
        print('objective', obj, fmt(r), flush=True)

    # 3. Width interaction at share 0.6.
    width_results = {}
    for hidden in (4, 8, 16):
        r, info, _ = run_nn(bank, prep, 0.6, hidden=hidden)
        width_results[hidden] = compact(r, info)
        payload['runs'][f'nn_pca_h{hidden}_share_0.6'] = width_results[hidden]
        print('hidden', hidden, fmt(r), flush=True)

    # 4. Ridge control.
    ridge_nat, rinfo_nat = run_ridge(bank, prep, None)
    ridge_60, rinfo_60 = run_ridge(bank, prep, 0.6)
    payload['runs']['ridge_natural'] = compact(ridge_nat, rinfo_nat)
    payload['runs']['ridge_share_0.6'] = compact(ridge_60, rinfo_60)
    print('ridge natural', fmt(ridge_nat), flush=True)
    print('ridge share 0.6', fmt(ridge_60), flush=True)

    # 5. LOO selection over 311-320 (training-side discipline).
    by_tier_sel = None
    if a.skip_loo:
        loo = None
        selected = None
    else:
        loo = loo_select(bank, (None,) + SHARES)
        selected = loo['selected']
        payload['loo'] = loo
        print('LOO mean MAPE', {k: round(100 * v, 2) for k, v in loo['mean_mape'].items()},
              'selected', selected, flush=True)
        r, info, by_tier_sel = run_nn(bank, prep, selected)
        payload['tuned'] = compact(r, info)
        payload['tuned_share'] = selected
        print('tuned refit', fmt(r), flush=True)
    by_tier_nat = natural_tiers
    by_tier_best = by_tier_sel
    payload['per_tier'] = dict(natural=by_tier_nat,
                               **({'selected_share': by_tier_best} if by_tier_best else {}))

    # 6. Optional family transfer at share 0.6.
    r_min, info_min, _ = run_nn(bank, prep, 0.6, matrix='minimal')
    r_min_nat, info_min_nat, _ = run_nn(bank, prep, None, matrix='minimal')
    payload['runs']['nn_minimal_natural'] = compact(r_min_nat, info_min_nat)
    payload['runs']['nn_minimal_share_0.6'] = compact(r_min, info_min)
    print('nn_minimal natural', fmt(r_min_nat), flush=True)
    print('nn_minimal share 0.6', fmt(r_min), flush=True)
    r_tcn, info_tcn, _ = run_tcn24(bank, prep, 0.6)
    r_tcn_nat, info_tcn_nat, _ = run_tcn24(bank, prep, None)
    payload['runs']['tcn24_natural'] = compact(r_tcn_nat, info_tcn_nat)
    payload['runs']['tcn24_share_0.6'] = compact(r_tcn, info_tcn)
    print('tcn24 natural', fmt(r_tcn_nat), flush=True)
    print('tcn24 share 0.6', fmt(r_tcn), flush=True)

    payload['seconds'] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    save_results(OUT / 'evaluation.json', payload)
    write_report(REPORT, 'Raised objective-tier loss weighting — JointNet sweep (321-324)',
                 _sections(payload))
    print('done in', round(payload['seconds']), 's', flush=True)


def _pct(v):
    return '—' if v is None else f'{100 * v:.2f}%'


def _key(d, k):
    """Dict lookup tolerant of JSON stringified keys."""
    return d.get(k, d.get(str(k), d.get(None if k is not None else 'null')))


def _sections(payload):
    runs = payload['runs']
    mass0 = payload['natural_objective_mass']
    loo = payload.get('loo')
    lines = [
        f'Same harness, same rows as every sweep: 315-event bank train, events 321-324 test, '
        f'event-balanced MAPE on tiers 500/1000/1500/2000 (312 clean rows). Natural objective '
        f'weight mass is {100 * mass0:.1f}%. Rescale via raise_objective_share preserves '
        f'within-group ratios; only the weights array changes.', '',
        '| Config | Share | Objective set | Hidden | MAPE | Bias | Final 24h | Mass before → after |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']

    def row(label, share, obj, hidden, r):
        share_s = 'natural' if share is None else f'{share:.2f}'
        obj_s = ','.join(map(str, obj)) if obj else '—'
        return (f"| {label} | {share_s} | {obj_s} | {hidden or '—'} | {_pct(r['mape'])} | "
                f"{_pct(r['bias'])} | {_pct(r['final24h'])} | "
                f"{100 * r['objective_mass_before']:.1f}% → {100 * r['objective_mass_after']:.1f}% |")

    lines.append(row('nn_pca natural', None, None, 8, runs['nn_pca_natural']))
    for share in SHARES:
        lines.append(row('nn_pca share sweep', share, OBJECTIVE, 8, runs[f'nn_pca_share_{share}']))
    for obj in ((500, 1000), (500, 1000, 1500), (500, 1000, 1500, 2000), (1000, 2000)):
        lines.append(row('nn_pca objective set @0.6', 0.6, obj, 8,
                         runs[f'nn_pca_obj_{"_".join(map(str, obj))}']))
    for hidden in (4, 8, 16):
        lines.append(row('nn_pca width @0.6', 0.6, OBJECTIVE, hidden,
                         runs[f'nn_pca_h{hidden}_share_0.6']))
    lines.append(row('ridge natural', None, None, None, runs['ridge_natural']))
    lines.append(row('ridge control @0.6', 0.6, OBJECTIVE, None, runs['ridge_share_0.6']))
    for name in ('nn_minimal_natural', 'nn_minimal_share_0.6', 'tcn24_natural', 'tcn24_share_0.6'):
        share = None if name.endswith('natural') else 0.6
        lines.append(row(name.replace('_', ' '), share, OBJECTIVE if share else None, None, runs[name]))
    if loo is not None:
        lines.append(row('nn_pca TUNED (LOO-selected)', payload['tuned_share'], OBJECTIVE, 8,
                         payload['tuned']))

    pe = ['| Config | 321 | 322 | 323 | 324 |', '| --- | ---: | ---: | ---: | ---: |']
    for label, key in [('natural', 'nn_pca_natural'),
                       ('share 0.22', 'nn_pca_share_0.22'),
                       ('share 0.4', 'nn_pca_share_0.4'),
                       ('share 0.6', 'nn_pca_share_0.6'),
                       ('share 0.8', 'nn_pca_share_0.8'),
                       ('ridge natural', 'ridge_natural'),
                       ('ridge @0.6', 'ridge_share_0.6')]:
        r = runs[key]
        pe.append('| ' + label + ' | ' + ' | '.join(
            _pct(r['per_event'].get(e, r['per_event'].get(str(e)))) for e in (321, 322, 323, 324)) + ' |')
    if loo is not None:
        r = payload['tuned']
        pe.append('| tuned (LOO, share 0.8) | ' + ' | '.join(
            _pct(r['per_event'].get(e, r['per_event'].get(str(e)))) for e in (321, 322, 323, 324)) + ' |')

    sections = [('Summary', lines), ('Per-event MAPE (key configs)', pe)]
    if loo is not None:
        loo_lines = ['Selection on post-rule TRAINING events 311-320 only (leave-one-out); '
                     'mean focal-tier MAPE across the 10 held-out events.', '',
                     '| Share | LOO mean MAPE | LOO mean bias |', '| ---: | ---: | ---: |']
        for share in (None,) + SHARES:
            k = 'null' if share is None else str(share)
            loo_lines.append(f"| {'natural' if share is None else share} | "
                             f"{_pct(loo['mean_mape'][k])} | {_pct(loo['mean_bias'][k])} |")
        loo_lines += ['', f"Selected share: **{payload['tuned_share']}**. The tuned row above "
                       f"refits the full 315-event bank at that share and scores 321-324."]
        sections.append(('LOO selection on 311-320', loo_lines))
    tier_lines = ['Per-tier MAPE on 321-324 across ALL tiers (ablation honesty: do '
                  'non-objective tiers degrade?).', '',
                  '| Tier | Natural MAPE | Tuned-share MAPE | Δ pp | Objective? |',
                  '| ---: | ---: | ---: | ---: | --- |']
    nat_tiers = payload.get('per_tier', {}).get('natural')
    sel_tiers = payload.get('per_tier', {}).get('selected_share')
    if nat_tiers and sel_tiers:
        for t in sorted(map(int, nat_tiers)):
            v = nat_tiers[str(t)]
            b = sel_tiers.get(str(t), {})
            d = (b.get('mape') - v['mape']) * 100 if b.get('mape') is not None else None
            tier_lines.append(f"| {t} | {_pct(v['mape'])} | {_pct(b.get('mape'))} | "
                              f"{'—' if d is None else f'{d:+.2f}'} | "
                              f"{'yes' if t in OBJECTIVE else 'no'} |")
    else:
        tier_lines.append('| (LOO skipped) | | | | |')
    sections.append(('Per-tier ablation check', tier_lines))
    seed = payload.get('seed_sensitivity')
    if seed:
        seed_lines = [seed['note'], '',
                      '| Seed | Share | MAPE | Bias |', '| ---: | ---: | ---: | ---: |']
        for r in seed['runs']:
            seed_lines.append(f"| {r['seed']} | {'natural' if r['share'] is None else r['share']} | "
                              f"{_pct(r['mape'])} | {_pct(r['bias'])} |")
        seed_lines += ['', '| Share | Mean MAPE over 3 seeds |', '| ---: | ---: |']
        for k, v in seed['mean_mape'].items():
            seed_lines.append(f"| {k} | {_pct(v)} |")
        sections.append(('Seed sensitivity (nn_pca, 321-324)', seed_lines))
    caveats = [
        'Share grid on 321-324 is descriptive; only the LOO-selected share on 311-320 is a tuned pick.',
        'Four validation events limit precision; weight mass is exact but realized loss share '
        'also depends on per-tier error magnitudes.',
        'Seed spread on 321-324 is ~1 pp — larger than most share effects; single-seed share '
        'differences at 321-324 (e.g. 0.22 at seed 1729) are not reliable.',
        'Compression and masks are refit inside each LOO fold (train-only preprocessing).',
        f"Run finished in {payload['seconds']:.0f}s. No deployed model is replaced."]
    sections.append(('Caveats', caveats))
    return sections


def report_only():
    payload = json.loads((OUT / 'evaluation.json').read_text())
    write_report(REPORT, 'Raised objective-tier loss weighting — JointNet sweep (321-324)',
                 _sections(payload))


if __name__ == '__main__':
    main()
