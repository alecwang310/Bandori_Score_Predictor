"""Feature-block ablation and selection for nn_pca on new (post-rule) events 321-324.

Uses the frozen sweep harness (identical rows, splits and scoring). All custom
matrices are built by explicit column selection on the finished pca layout:

    pca = [16 PCs | 96 explicit (progress/remaining/fraction/presence x24 tiers)
           | 552 daily summaries (23 x24) | 504 rule metadata (21 x24)]   = 1168

Column selection on the finished (masked) matrix is exact: finish_context's
masks are a per-column diagonal multiply, so dropping columns equals
re-finishing a column_stack of the kept blocks. Every custom matrix records its
shape and the block slices used; train/val column counts are asserted equal.

Selection discipline: descriptive sweeps on 321-324; the crowned lean set is
chosen by leave-one-out over post-rule TRAINING events 311-320 (full bank refit
per fold), and its 321-324 score is the tuned headline.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import sweep_harness as H
from .daily_observer import SUMMARIES
from .experiments import correction_forecast
from .rules import FEATURES as RULE_FEATURES

RUN='runs/sweep_features_20261019'
REPORT='reports/sweep_features_20261019.md'

# Finished pca layout block boundaries (columns).
PC_W, EXPL_W, DAILY_W, RULE_W = 16, 96, 552, 504
PCA_W = PC_W + EXPL_W + DAILY_W + RULE_W  # 1168
PC_SLICE = slice(0, PC_W)
EXPL_SLICE = slice(PC_W, PC_W + EXPL_W)
DAILY_SLICE = slice(PC_W + EXPL_W, PC_W + EXPL_W + DAILY_W)
RULE_SLICE = slice(PC_W + EXPL_W + DAILY_W, PCA_W)
N_TIERS = 24

# Daily-summary subgroups (indices into SUMMARIES).
PACE_SUMS = [i for i, s in enumerate(SUMMARIES) if s not in
             ('night_day_log_ratio', 'harmonic_sin', 'harmonic_cos', 'harmonic2_sin',
              'harmonic2_cos', 'recent_cycle', 'future_cycle', 'recent_early_log_ratio')]
HARMONIC_SUMS = [i for i, s in enumerate(SUMMARIES)
                 if i not in PACE_SUMS]

# Rule-metadata subgroups (indices into RULE_FEATURES; support = last 3).
RULE_TYPE = [i for i, f in enumerate(RULE_FEATURES) if f.startswith('type_')]
RULE_ERA = [i for i in range(len(RULE_FEATURES))
            if i not in RULE_TYPE and i < len(RULE_FEATURES) - 3]


def daily_cols(summary_indices):
    """Columns of the finished pca matrix for the given daily-summary indices."""
    return np.array([DAILY_SLICE.start + t * len(SUMMARIES) + j
                     for t in range(N_TIERS) for j in summary_indices])


def rule_cols(feature_indices):
    """Columns of the finished pca matrix for the given rule-feature indices."""
    return np.array([RULE_SLICE.start + t * len(RULE_FEATURES) + f
                     for t in range(N_TIERS) for f in feature_indices])


def block_cols(include):
    """Full-block column selection. include: iterable of 'pc','expl','daily','rule'."""
    parts = []
    if 'pc' in include: parts.append(np.arange(PC_W))
    if 'expl' in include: parts.append(np.arange(EXPL_W) + PC_W)
    if 'daily' in include: parts.append(np.arange(DAILY_W) + PC_W + EXPL_W)
    if 'rule' in include: parts.append(np.arange(RULE_W) + PC_W + EXPL_W + DAILY_W)
    return np.concatenate(parts)


class FeatureStudy:
    """Fit/evaluate custom feature subsets on the frozen matched rows."""

    def __init__(self, bank, prep):
        self.bank = bank
        self.prep = prep
        self.audit = {}  # name -> layout record
        self.results = {}

    def _matrices(self, name, cols):
        tr = self.prep.matrix['pca'][:, cols]
        va = self.prep.val['matrices']['pca'][:, cols]
        assert tr.shape[1] == va.shape[1] == len(cols), (name, tr.shape, va.shape)
        self.audit[name] = dict(shape=list(tr.shape), val_shape=list(va.shape),
                                columns=int(len(cols)))
        return tr, va

    def run_nn(self, name, cols, note=''):
        t0 = time.time()
        tr, va = self._matrices(name, cols)
        model = H.fit_jointnet(tr, self.prep.targets, self.prep.weights,
                               self.prep.bins, self.bank.config, hidden=8, seed=1729)
        delta = model.predict(va)
        values = correction_forecast(self.prep.val['cases'], self.prep.val['index'],
                                     delta, 1.0, 'nn')
        res = H.score(self.prep.val['cases'], values, self.bank.data)
        res.update(columns=int(tr.shape[1]), seconds=time.time() - t0, note=note)
        self.results[name] = res
        print(f'{name:28s} cols={tr.shape[1]:5d}  {H.fmt(res)}  '
              f'[{time.time()-t0:.1f}s] {H.per_event_line(res)}', flush=True)
        return res

    def run_ridge(self, name, cols, note='', matrix='pca'):
        t0 = time.time()
        if matrix == 'pca':
            tr, va = self._matrices(name, cols)
        else:
            tr = self.prep.matrix[matrix][:, cols]
            va = self.prep.val['matrices'][matrix][:, cols]
            assert tr.shape[1] == va.shape[1] == len(cols)
            self.audit[name] = dict(shape=list(tr.shape), val_shape=list(va.shape),
                                    columns=int(len(cols)), base_matrix=matrix)
        model = H.fit_ridge(tr, self.prep.targets, self.prep.weights, self.prep.bins)
        delta = model.predict(va)
        values = correction_forecast(self.prep.val['cases'], self.prep.val['index'],
                                     delta, 0.5, 'ridge')
        res = H.score(self.prep.val['cases'], values, self.bank.data)
        res.update(columns=int(tr.shape[1]), seconds=time.time() - t0, note=note)
        self.results[name] = res
        print(f'{name:28s} cols={tr.shape[1]:5d}  {H.fmt(res)}  '
              f'[{time.time()-t0:.1f}s] {H.per_event_line(res)}', flush=True)
        return res


def permutation_importance(bank, prep, model, seeds=(0, 1, 2)):
    """Block permutation on validation rows: mean MAPE after jointly shuffling
    each block's columns across prep.val rows (same permutation per block)."""
    base = H.evaluate(bank, prep, model, 'pca')[0]['mape']
    blocks = dict(pc=np.arange(PC_W),
                  expl=np.arange(EXPL_W) + PC_W,
                  daily=np.arange(DAILY_W) + PC_W + EXPL_W,
                  rule=np.arange(RULE_W) + PC_W + EXPL_W + DAILY_W,
                  rule_era=rule_cols(RULE_ERA),
                  rule_types=rule_cols(RULE_TYPE),
                  daily_pace=daily_cols(PACE_SUMS),
                  daily_harmonic=daily_cols(HARMONIC_SUMS))
    out = {}
    for bname, cols in blocks.items():
        mapes = []
        for s in seeds:
            rng = np.random.default_rng(s)
            X = prep.val['matrices']['pca'].copy()
            perm = rng.permutation(X.shape[0])
            X[:, cols] = X[perm][:, cols]
            delta = model.predict(X)
            values = correction_forecast(prep.val['cases'], prep.val['index'],
                                         delta, 1.0, 'nn')
            mapes.append(H.score(prep.val['cases'], values, bank.data)['mape'])
        out[bname] = dict(mape=float(np.mean(mapes)), std=float(np.std(mapes)),
                          delta=float(np.mean(mapes) - base),
                          per_seed=[float(m) for m in mapes],
                          columns=int(len(cols)))
        print(f'perm {bname:16s} cols={len(cols):4d}  MAPE {100*np.mean(mapes):.2f}% '
              f'(delta {100*(np.mean(mapes)-base):+.2f}pp, std {100*np.std(mapes):.2f})',
              flush=True)
    return dict(base_mape=base, blocks=out)


def loo_post_rule(bank, matrix_name, cols, label=None, kind='nn'):
    """Leave-one-out over post-rule TRAINING events 311-320.

    For each held-out event: refit everything supervised (compression, masks,
    model) on the other 314 events; score the held-out event's frozen Protocol-A
    context finished with that fold's compression/masks. Returns per-event and
    mean event-balanced MAPE.
    """
    from .regime_suite import finish_context
    per_event = {}
    label = label or matrix_name
    for held in H.POST_RULE_TRAIN_EVENTS:
        ids = [i for i in bank.train_ids if i != held]
        prep = H.prepare_fit(bank, supervised_ids=ids)
        tr = prep.matrix[matrix_name][:, cols]
        model = H.fit_jointnet(tr, prep.targets, prep.weights, prep.bins,
                               bank.config, hidden=8, seed=1729)
        held_ctx = H.supervised_slice(bank.ctx_all, [held])
        held_ctx, _ = finish_context(held_ctx, prep.compression, masks=prep.masks)
        va = held_ctx['matrices'][matrix_name][:, cols]
        delta = model.predict(va)
        values = correction_forecast(held_ctx['cases'], held_ctx['index'], delta,
                                     1.0, 'nn')
        res = H.score(held_ctx['cases'], values, bank.data)
        per_event[held] = dict(mape=float(res['mape']), bias=float(res['bias']),
                               rows=int(res['rows']))
        print(f'  LOO {label:26s} holdout {held}: MAPE {100*res["mape"]:.2f}% '
              f'rows {res["rows"]}', flush=True)
    mean = float(np.mean([v['mape'] for v in per_event.values()]))
    return dict(per_event=per_event, mean_mape=mean)


def main():
    t_start = time.time()
    Path(RUN).mkdir(parents=True, exist_ok=True)
    bank = H.load_bank()
    print('== validate frozen baselines ==', flush=True)
    nn_pca, ridge_pca = H.validate(bank)

    prep = H.prepare_fit(bank)
    study = FeatureStudy(bank, prep)

    # Full pca reference model (kept for permutation importance).
    full = H.fit_jointnet(prep.matrix['pca'], prep.targets, prep.weights,
                          prep.bins, bank.config, hidden=8, seed=1729)
    study.results['nn_pca_full'] = dict(nn_pca, columns=PCA_W,
                                        note='frozen reference')
    print(f'{"nn_pca_full":28s} cols={PCA_W:5d}  {H.fmt(nn_pca)}  '
          f'{H.per_event_line(nn_pca)}', flush=True)

    print('== 1. block ablation on pca layout ==', flush=True)
    for label, keep in (('drop_pcs', ['expl', 'daily', 'rule']),
                        ('drop_explicit', ['pc', 'daily', 'rule']),
                        ('drop_daily', ['pc', 'expl', 'rule']),
                        ('drop_rule', ['pc', 'expl', 'daily'])):
        study.run_nn(label, block_cols(keep), note=f'keep={keep}')

    print('== 2. family comparison ==', flush=True)
    fam = {}
    for name in ('minimal', 'standard', 'rich'):
        t0 = time.time()
        model = H.fit_jointnet(prep.matrix[name], prep.targets, prep.weights,
                               prep.bins, bank.config, hidden=8, seed=1729)
        res, _ = H.evaluate(bank, prep, model, name)
        res.update(columns=int(prep.matrix[name].shape[1]),
                   seconds=time.time() - t0, note='harness matrix')
        study.results[f'nn_{name}'] = res
        fam[name] = res
        print(f'{"nn_"+name:28s} cols={res["columns"]:5d}  {H.fmt(res)}  '
              f'[{time.time()-t0:.1f}s] {H.per_event_line(res)}', flush=True)
    study.results['nn_pca_full'] = dict(study.results['nn_pca_full'])

    print('== 3. rule-metadata deep dive ==', flush=True)
    study.run_nn('rule_only_plus_explicit',
                 np.concatenate([np.arange(EXPL_W) + PC_W,
                                 np.arange(RULE_W) + PC_W + EXPL_W + DAILY_W]),
                 note='explicit + rule metadata only')
    keep_no_types = np.concatenate([
        np.arange(PC_W + EXPL_W + DAILY_W),
        rule_cols([i for i in range(len(RULE_FEATURES)) if i not in RULE_TYPE])])
    study.run_nn('drop_rule_types', keep_no_types,
                 note='pca minus event-type one-hots (8x24)')
    keep_no_era = np.concatenate([
        np.arange(PC_W + EXPL_W + DAILY_W),
        rule_cols([i for i in range(len(RULE_FEATURES)) if i not in RULE_ERA])])
    study.run_nn('drop_rule_era', keep_no_era,
                 note='pca minus reward-era/boundary features (10x24)')

    print('== 3b. lean brackets ==', flush=True)
    expl_only = np.arange(EXPL_W) + PC_W
    pc_expl = np.arange(PC_W + EXPL_W)
    expl_daily = np.concatenate([expl_only, np.arange(DAILY_W) + PC_W + EXPL_W])
    rule_only = np.arange(RULE_W) + PC_W + EXPL_W + DAILY_W
    study.run_nn('explicit_only', expl_only, note='96 explicit cols only')
    study.run_nn('pc_plus_explicit', pc_expl, note='16 PCs + 96 explicit')
    study.run_nn('expl_plus_daily', expl_daily, note='explicit + daily, no rule')
    study.run_nn('rule_only', rule_only, note='rule metadata only (504)')

    print('== 4. daily-summary deep dive ==', flush=True)
    base_no_daily = block_cols(['pc', 'expl', 'rule'])
    study.run_nn('daily_pace_only',
                 np.concatenate([base_no_daily, daily_cols(PACE_SUMS)]),
                 note=f'pace/rate summaries only ({len(PACE_SUMS)}x24)')
    study.run_nn('daily_harmonic_only',
                 np.concatenate([base_no_daily, daily_cols(HARMONIC_SUMS)]),
                 note=f'clock-harmonic summaries only ({len(HARMONIC_SUMS)}x24)')

    print('== 5. permutation importance (fitted full pca) ==', flush=True)
    perm = permutation_importance(bank, prep, full)

    # Selection: shortlist of lean candidates (descriptive, within 0.5pp of full
    # pca on 321-324 or better), then pick by LOO mean MAPE on post-rule train
    # events 311-320 (the mandated selection protocol). Tie-break: fewer columns.
    base_mape = study.results['nn_pca_full']['mape']
    shortlist = [n for n, r in study.results.items()
                 if n != 'nn_pca_full' and n.startswith(
                     ('drop_', 'nn_minimal', 'nn_standard', 'nn_rich',
                      'daily_', 'rule_only', 'explicit_only',
                      'pc_plus_explicit', 'expl_plus_daily'))
                 and r['mape'] - base_mape <= 0.005]
    shortlist.sort(key=lambda n: (study.results[n]['columns'], study.results[n]['mape']))
    print(f'shortlist for LOO selection: {shortlist}', flush=True)

    print('== 6. LOO selection on post-rule train events 311-320 ==', flush=True)
    sel_cols = {'nn_pca_full': np.arange(PCA_W),
                'drop_pcs': block_cols(['expl', 'daily', 'rule']),
                'drop_explicit': block_cols(['pc', 'daily', 'rule']),
                'drop_daily': block_cols(['pc', 'expl', 'rule']),
                'drop_rule': block_cols(['pc', 'expl', 'daily']),
                'nn_minimal': np.arange(prep.matrix['minimal'].shape[1]),
                'nn_standard': np.arange(prep.matrix['standard'].shape[1]),
                'nn_rich': np.arange(prep.matrix['rich'].shape[1]),
                'daily_pace_only': np.concatenate([base_no_daily, daily_cols(PACE_SUMS)]),
                'daily_harmonic_only': np.concatenate([base_no_daily, daily_cols(HARMONIC_SUMS)]),
                'rule_only_plus_explicit': np.concatenate([
                    np.arange(EXPL_W) + PC_W,
                    np.arange(RULE_W) + PC_W + EXPL_W + DAILY_W]),
                'drop_rule_types': keep_no_types,
                'drop_rule_era': keep_no_era,
                'explicit_only': expl_only,
                'pc_plus_explicit': pc_expl,
                'expl_plus_daily': expl_daily,
                'rule_only': rule_only}
    matrix_of = {'nn_minimal': 'minimal', 'nn_standard': 'standard',
                 'nn_rich': 'rich'}
    loo_all = {}
    loo_full = loo_post_rule(bank, 'pca', np.arange(PCA_W), label='full pca')
    loo_all['nn_pca_full'] = loo_full
    for name in shortlist:
        m = matrix_of.get(name, 'pca')
        loo_all[name] = loo_post_rule(bank, m, sel_cols[name], label=name)
    # Select by LOO mean MAPE; tie-break (within 0.1pp) by fewer columns.
    ranked = sorted(loo_all.items(),
                    key=lambda kv: (kv[1]['mean_mape'],
                                    study.results[kv[0]]['columns'] if kv[0] in study.results else 10**6))
    best_mean = ranked[0][1]['mean_mape']
    tied = [n for n, v in ranked if v['mean_mape'] - best_mean <= 0.001]
    selected = min(tied, key=lambda n: study.results.get(n, {}).get('columns', PCA_W))
    loo_sel = loo_all[selected]
    print('LOO ranking (311-320 mean MAPE):', flush=True)
    for n, v in sorted(loo_all.items(), key=lambda kv: kv[1]['mean_mape']):
        print(f'  {n:26s} {100*v["mean_mape"]:.2f}%', flush=True)
    print(f'selected by LOO: {selected}  '
          f'(full pca {100*loo_full["mean_mape"]:.2f}%, selected {100*loo_sel["mean_mape"]:.2f}%)',
          flush=True)

    print('== 7. ridge on selected feature set ==', flush=True)
    if selected == 'nn_pca_full':
        ridge_sel = ridge_pca
        study.results['ridge_selected'] = dict(ridge_pca, note='same as frozen ridge')
    else:
        ridge_sel = study.run_ridge('ridge_selected', sel_cols[selected],
                                    note=f'ridge on {selected}',
                                    matrix=matrix_of.get(selected, 'pca'))

    payload = dict(
        protocol=dict(train_events=len(bank.train_ids), val_events=list(bank.val_ids),
                      tiers=list(H.FOCAL_TIERS), hidden=8, seed=1729, gain_nn=1.0,
                      gain_ridge=0.5),
        frozen_validate=dict(nn_pca=dict(mape=nn_pca['mape'], bias=nn_pca['bias'],
                                         final24h=nn_pca['final24h']),
                             ridge=dict(mape=ridge_pca['mape'], bias=ridge_pca['bias'],
                                        final24h=ridge_pca['final24h'])),
        layout=dict(pc=PC_W, explicit=EXPL_W, daily=DAILY_W, rule=RULE_W, total=PCA_W,
                    daily_pace=PACE_SUMS, daily_harmonic=HARMONIC_SUMS,
                    rule_types=RULE_TYPE, rule_era=RULE_ERA),
        audit=study.audit,
        results={k: {kk: vv for kk, vv in r.items() if kk != 'full'}
                 for k, r in study.results.items()},
        permutation=perm,
        selection=dict(selected=selected,
                       shortlist=shortlist,
                       loo_full=loo_full, loo_selected=loo_sel,
                       loo_all={k: v for k, v in loo_all.items()},
                       tuned_headline=study.results[selected]['mape']),
        elapsed_seconds=time.time() - t_start)
    H.save_results(Path(RUN) / 'evaluation.json', payload)
    write_md_report(bank, payload, study)
    print(f'done in {time.time()-t_start:.0f}s', flush=True)


def _table(study, names):
    lines = ['| feature set | columns | MAPE | bias | final24h |',
             '|---|---:|---:|---:|---:|']
    for n in names:
        r = study.results[n]
        lines.append(f'| {n} | {r.get("columns","—")} | {100*r["mape"]:.2f}% | '
                     f'{100*r["bias"]:+.2f}% | {100*r["final24h"]:.2f}% |')
    return lines


def write_md_report(bank, payload, study):
    res = study.results
    base_mape = res['nn_pca_full']['mape']
    ablation = ['nn_pca_full', 'drop_pcs', 'drop_explicit', 'drop_daily', 'drop_rule']
    families = ['nn_minimal', 'nn_standard', 'nn_rich', 'nn_pca_full']
    rules = ['nn_pca_full', 'drop_rule', 'rule_only_plus_explicit',
             'drop_rule_types', 'drop_rule_era', 'explicit_only',
             'pc_plus_explicit', 'expl_plus_daily', 'rule_only']
    dailys = ['nn_pca_full', 'drop_daily', 'daily_pace_only', 'daily_harmonic_only']
    perm_lines = ['| block | cols | MAPE (permuted) | ΔMAPE |',
                  '|---|---:|---:|---:|']
    for b, v in sorted(payload['permutation']['blocks'].items(),
                       key=lambda kv: -kv[1]['delta']):
        perm_lines.append(f'| {b} | {v["columns"]} | {100*v["mape"]:.2f}% | '
                          f'{100*v["delta"]:+.2f}pp |')
    sel = payload['selection']['selected']
    loo_f, loo_s = payload['selection']['loo_full'], payload['selection']['loo_selected']
    loo_rank = ['| feature set | LOO mean MAPE (311-320) | 321-324 (tuned) |',
                '|---|---:|---:|']
    for n, v in sorted(payload['selection']['loo_all'].items(),
                       key=lambda kv: kv[1]['mean_mape']):
        mark = ' **(selected)**' if n == sel else ''
        tuned = (f'{100*study.results[n]["mape"]:.2f}%' if n in study.results
                 else '—')
        loo_rank.append(f'| {n}{mark} | {100*v["mean_mape"]:.2f}% | {tuned} |')
    loo_lines = ['| holdout event | full pca MAPE | selected MAPE |',
                 '|---|---:|---:|']
    for e in sorted(loo_f['per_event']):
        a = loo_f['per_event'][e]['mape']
        b = loo_s['per_event'][e]['mape']
        loo_lines.append(f'| {e} | {100*a:.2f}% | {100*b:.2f}% |')
    loo_lines.append(f'| **mean** | **{100*loo_f["mean_mape"]:.2f}%** | '
                     f'**{100*loo_s["mean_mape"]:.2f}%** |')
    loo_lines += ['', *loo_rank]

    key = ['nn_pca_full', 'drop_daily', 'drop_rule', 'drop_pcs', sel]
    key = list(dict.fromkeys(key))
    key_lines = ['| key config | MAPE | per-event (321/322/323/324) |',
                 '|---|---:|---|']
    for n in key:
        r = res[n]
        key_lines.append(f'| {n} | {100*r["mape"]:.2f}% | {H.per_event_line(r)} |')

    sections = [
        ('Frozen validation', [
            f'nn_pca {100*payload["frozen_validate"]["nn_pca"]["mape"]:.2f}% MAPE '
            f'(bias {100*payload["frozen_validate"]["nn_pca"]["bias"]:+.2f}%, '
            f'final24h {100*payload["frozen_validate"]["nn_pca"]["final24h"]:.2f}%); '
            f'ridge {100*payload["frozen_validate"]["ridge"]["mape"]:.2f}%. '
            'Protocol: 315-event supervised bank (incl. post-rule 311-320), '
            'test on 321-324, focal tiers 500/1000/1500/2000, event-balanced MAPE.',
            '',
            f'pca layout: {PC_W} PCs + {EXPL_W} explicit + {DAILY_W} daily + '
            f'{RULE_W} rule = {PCA_W} columns. Custom matrices are exact column '
            'subsets of the finished (masked) layout; train/val counts asserted equal '
            '(see audit in evaluation.json).']),
        ('Main table: feature set | columns | MAPE | bias | final24h', []),
        ('1. Block ablation (pca layout)', []),
        ('2. Family comparison', []),
        ('3. Rule-metadata deep dive', [
            'All post-rule events (311-326) are under the new reward rule; rule '
            'metadata is exactly the block that encodes the era switch, boundaries '
            'and event type. Whether it earns its keep on 321-324 is decided by the '
            'drop_rule / drop_rule_* rows and the permutation deltas.']),
        ('4. Daily-summary deep dive', []),
        ('5. Permutation importance (fitted full pca, val rows)',
         ['Blocks jointly shuffled across the 78 validation rows; mean over 3 seeds. '
          'Agreement with the retrain ablations is the cross-check.']),
        ('6. LOO selection on post-rule train events 311-320', [
            f'Shortlist = descriptive sets within 0.5pp of full pca on 321-324. '
            f'Selection is by LOO mean MAPE over post-rule TRAINING events 311-320 '
            f'(labels-only holdout, Protocol A features). '
            f'Selected: **{sel}**. LOO mean MAPE: full pca '
            f'{100*loo_f["mean_mape"]:.2f}%, selected {100*loo_s["mean_mape"]:.2f}%. '
            f'Tuned headline on 321-324: {100*payload["selection"]["tuned_headline"]:.2f}%.']),
        ('7. Ridge on selected feature set', [
            f'ridge on {sel}: MAPE {100*res["ridge_selected"]["mape"]:.2f}% '
            f'(frozen ridge on full pca {100*payload["frozen_validate"]["ridge"]["mape"]:.2f}%).']),
        ('Key per-event detail', []),
        ('Caveats', [
            '- The 321-324 grid is descriptive (4 events); the tuned headline is the '
            'LOO-selected set scored once on 321-324.',
            '- LOO over 311-320 removes labels only (Protocol A features keep the '
            'full reference pool), same convention as the harness.',
            '- Permutation importance runs on 78 validation rows and can understate '
            'blocks whose information is duplicated by another block.']),
    ]
    body_map = {
        'Main table: feature set | columns | MAPE | bias | final24h':
            _table(study, ['nn_pca_full', 'drop_pcs', 'drop_explicit', 'drop_daily',
                           'drop_rule', 'nn_minimal', 'nn_standard', 'nn_rich',
                           'rule_only_plus_explicit', 'drop_rule_types',
                           'drop_rule_era', 'explicit_only', 'pc_plus_explicit',
                           'expl_plus_daily', 'rule_only', 'daily_pace_only',
                           'daily_harmonic_only', 'ridge_selected']),
        '1. Block ablation (pca layout)': _table(study, ablation),
        '2. Family comparison': _table(study, families),
        '3. Rule-metadata deep dive': _table(study, rules),
        '4. Daily-summary deep dive': _table(study, dailys),
        '5. Permutation importance (fitted full pca, val rows)': perm_lines,
        '6. LOO selection on post-rule train events 311-320': loo_lines,
        'Key per-event detail': key_lines,
    }
    sections = [(h, body_map.get(h, body)) for h, body in sections]
    H.write_report(REPORT, 'Feature ablation/selection for nn_pca on events 321-324',
                   sections)


if __name__ == '__main__':
    main()
