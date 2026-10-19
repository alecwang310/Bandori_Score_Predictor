"""Training-data-scope sweep: how much history should the post-rule models train with?

Question: does restricting the supervised training bank to (only) the post-rule
events help or hurt accuracy on the NEW events 321-324?

Everything runs through `bandori.sweep_harness` (frozen protocol, do not edit):

* **Protocol A** (all variants in the main table): matched rows. Validation case
  features are always the frozen full-history features (analog reference pool =
  all 315 training events). Varying the scope only removes old *labels* from the
  supervised fit and changes the compression/masks fit on those rows.
* **Protocol B** (separate table, never mixed with A): post_rule_only fit PLUS a
  restricted analog reference pool `[311..320]`, which also rebuilds the
  validation case features — rows are not matched to Protocol A.

Variants (Protocol A):

1. `post_rule_only` — supervised_ids = [311..320] ("remove old data altogether";
   note this block holds 10 ids, of which 8 start on/after the 2026-05-26 rule
   change — 312/313 are pre-rule).
2. `post_rule+{20,50,100}recent` — [311..320] plus the N most recent older events
   (ranked by event end time among train_ids \\ [311..320]).
3. `full_315` — the validated baseline.
4. Recency weighting — full bank, but supervised weights of events older than a
   cut (50 most recent events keep full weight, or events ending before
   2026-01-01) multiplied by {0.25, 0.5}, scaling prep.weights rows through
   prep.train['cases'] event_id -> prep.train['index'] cells. Total weight mass
   is intentionally not renormalized.

Both models per variant, frozen hyperparameters (no tuning anywhere):
JointNet hidden 8 seed 1729 gain 1.0 (`nn_pca`) and ridge strength .01 gain .5
(`ridge`). Standing scope: event-balanced MAPE on clean checkpoints of events
321-324, tiers 500/1000/1500/2000 (312 rows).

Run:
    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -u -B -m bandori.sweep_data_scope

Outputs:
    runs/sweep_data_scope_20261019/evaluation.json
    reports/sweep_data_scope_20261019.md
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .sweep_harness import (
    FOCAL_TIERS,
    FROZEN_REFERENCE,
    VAL_EVENTS,
    evaluate,
    fit_jointnet,
    fit_ridge,
    fmt,
    load_bank,
    per_event_line,
    prepare_fit,
    save_results,
    val_context,
    validate,
    write_report,
)

RUN_DIR = Path('runs/sweep_data_scope_20261019')
OUT_JSON = RUN_DIR / 'evaluation.json'
OUT_REPORT = Path('reports/sweep_data_scope_20261019.md')

POST_IDS = list(range(311, 321))  # spec's post_rule_only block: 10 ids, 8 truly post-rule
RECENT_N = (20, 50, 100)
SCALE_FACTORS = (0.25, 0.5)
CUT_2026 = datetime.fromisoformat('2026-01-01T00:00:00+08:00').timestamp()
MODEL_GAINS = (('nn_pca', 1.0, 'nn'), ('ridge', 0.5, 'ridge'))


# --------------------------------------------------------------------------- helpers

def _by_end_desc(bank, ids):
    return sorted(ids, key=lambda e: bank.data.events[e].end, reverse=True)


def post_plus_recent(bank, n):
    """[311..320] plus the n most recent older training events."""
    post = set(POST_IDS)
    older = [e for e in _by_end_desc(bank, bank.train_ids) if e not in post]
    return sorted(post | set(older[:n]))


def downweighted_ids(bank, cut):
    """Event ids whose supervised weight is scaled: cut = ('recent', n) | ('before', ts)."""
    if cut[0] == 'recent':
        recent = set(_by_end_desc(bank, bank.train_ids)[:cut[1]])
        return sorted(set(bank.train_ids) - recent)
    return sorted(e for e in bank.train_ids if bank.data.events[e].end < cut[1])


def scale_weights(prep, old_ids, factor):
    """Copy prep.weights with every labeled cell of `old_ids` multiplied by factor."""
    cases = prep.train['cases']
    index = prep.train['index']
    is_old = np.array([c.event_id in old_ids for c in cases])
    sel = np.zeros(index.shape, bool)
    m = index >= 0
    sel[m] = is_old[index[m]]
    weights = prep.weights.copy()
    weights[sel] *= factor
    return weights, int((sel & (weights > 0)).sum())


def scope_stats(prep, weights, extra=None):
    stats = dict(
        n_supervised_events=len({c.event_id for c in prep.train['cases']}),
        n_supervised_cells=int(np.count_nonzero(weights)),
        weight_sum=float(weights.sum()),
    )
    if extra:
        stats.update(extra)
    return stats


def _record(variant, model, result, stats):
    return dict(
        variant=variant, model=model,
        mape=float(result['mape']), bias=float(result['bias']),
        final24h=None if result['final24h'] is None else float(result['final24h']),
        rows=int(result['rows']), events=int(result['events']),
        per_event={str(e): dict(mape=float(v['mape']), bias=float(v['bias']),
                                rows=int(v.get('rows') or 0))
                   for e, v in sorted(result['per_event'].items())},
        **stats,
    )


def fit_and_eval(bank, prep, weights, variant, stats):
    """Fit nn_pca + ridge on `weights` and score both on the frozen 321-324 rows."""
    nn = fit_jointnet(prep.matrix['pca'], prep.targets, weights, prep.bins,
                      bank.config, hidden=8, seed=1729)
    res_nn, _ = evaluate(bank, prep, nn, 'pca', gain=1.0, kind='nn')
    ridge = fit_ridge(prep.matrix['pca'], prep.targets, weights, prep.bins)
    res_ridge, _ = evaluate(bank, prep, ridge, 'pca', gain=0.5, kind='ridge')
    rows = []
    for model, res in (('nn_pca', res_nn), ('ridge', res_ridge)):
        r = _record(variant, model, res, stats)
        rows.append(r)
        print(f"  {r['variant']:26s} {r['model']:7s} {fmt(res)}  "
              f"cells={r['n_supervised_cells']} sum_w={r['weight_sum']:.4f}  "
              + per_event_line(res), flush=True)
    return rows


def run_scope(bank, variant, supervised_ids, scopes, extra=None):
    t0 = time.time()
    prep = prepare_fit(bank, supervised_ids)
    ids = list(bank.train_ids) if supervised_ids is None else sorted(supervised_ids)
    stats = scope_stats(prep, prep.weights, dict(supervised_ids=ids, **(extra or {})))
    rows = fit_and_eval(bank, prep, prep.weights, variant, stats)
    stats['seconds'] = round(time.time() - t0, 2)
    scopes[variant] = stats
    return rows


# --------------------------------------------------------------------- report pieces

def _pct(v):
    return '—' if v is None else f'{100 * v:.2f}%'


def _table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join([' --- '] * len(headers)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return out


def ranked_table(results):
    rows = []
    for r in sorted(results, key=lambda x: x['mape']):
        rows.append([r['variant'], r['model'], _pct(r['mape']), _pct(r['bias']),
                     _pct(r['final24h']),
                     f"{r['n_supervised_cells']} ({r['weight_sum']:.4f})"])
    return _table(['variant', 'model', 'MAPE', 'bias', 'final24h',
                   'n_supervised cells (sum w)'], rows)


def per_event_table(results):
    rows = []
    for r in sorted(results, key=lambda x: x['mape']):
        pe = r['per_event']
        rows.append([r['variant'], r['model']]
                    + [_pct(pe.get(str(e), {}).get('mape')) for e in VAL_EVENTS]
                    + [_pct(r['mape'])])
    return _table(['variant', 'model'] + [str(e) for e in VAL_EVENTS] + ['pooled'], rows)


def paired_sections(results):
    """Per-event MAPE deltas of the best (non-baseline) variant vs full_315, per model."""
    sections = []
    for model, _, _ in MODEL_GAINS:
        base = next(r for r in results if r['variant'] == 'full_315' and r['model'] == model)
        cand = [r for r in results
                if r['model'] == model and r['variant'] != 'full_315']
        best = min(cand, key=lambda r: r['mape'])
        rows = []
        for e in VAL_EVENTS:
            b = base['per_event'][str(e)]['mape']
            w = best['per_event'][str(e)]['mape']
            rows.append([e, _pct(b), _pct(w), f'{100 * (w - b):+.2f} pp'])
        rows.append(['pooled', _pct(base['mape']), _pct(best['mape']),
                     f'{100 * (best["mape"] - base["mape"]):+.2f} pp'])
        head = (f'{model}: best Protocol A variant `{best["variant"]}` vs full_315 baseline '
                f'(negative delta = variant wins)')
        sections.append((head, _table(
            ['event', 'full_315', best['variant'], 'delta'], rows)))
    return sections


# --------------------------------------------------------------------------- main

def main():
    bank = load_bank()
    print('== step 1: reproduce frozen baselines (validate) ==', flush=True)
    val_nn, val_ridge = validate(bank)

    results = []
    scopes = {}

    print('== step 2: Protocol A scope variants ==', flush=True)
    results += run_scope(bank, 'full_315', None, scopes)
    results += run_scope(bank, 'post_rule_only', POST_IDS, scopes,
                         dict(block_note='ids 311-320 (8 of 10 start on/after rule change)'))
    for n in RECENT_N:
        ids = post_plus_recent(bank, n)
        results += run_scope(bank, f'post_rule+{n}recent', ids, scopes,
                             dict(n_recent_older=n))

    # Recency weighting: one full-bank prep, scaled weight copies.
    print('== step 3: recency weighting on the full bank ==', flush=True)
    prep_full = prepare_fit(bank)
    base_weights = prep_full.weights.copy()
    for cut_name, cut in (('beyond50recent', ('recent', 50)),
                          ('pre2026', ('date', CUT_2026))):
        old_ids = downweighted_ids(bank, cut)
        for factor in SCALE_FACTORS:
            variant = f'weight_{cut_name}_x{factor:g}'
            t0 = time.time()
            weights, cells_scaled = scale_weights(prep_full, old_ids, factor)
            stats = scope_stats(prep_full, weights, dict(
                supervised_ids=sorted(bank.train_ids),
                scale_cut=cut_name, scale_factor=factor,
                n_events_scaled=len(old_ids), n_cells_scaled=cells_scaled,
                weight_sum_pre_scale=float(base_weights.sum())))
            rows = fit_and_eval(bank, prep_full, weights, variant, stats)
            stats['seconds'] = round(time.time() - t0, 2)
            scopes[variant] = stats
            results += rows

    # Cross-check the full-bank rows against validate()'s reproduction.
    for model, res_val in (('nn_pca', val_nn), ('ridge', val_ridge)):
        got = next(r for r in results if r['variant'] == 'full_315' and r['model'] == model)
        if abs(got['mape'] - res_val['mape']) > 5e-4:
            raise ValueError(f'full_315 {model} drifted from validate(): '
                             f"{got['mape']:.4f} vs {res_val['mape']:.4f}")

    print('== step 4: Protocol B (restricted reference pool) ==', flush=True)
    bank_b = load_bank()
    bank_b.reference_pool = list(POST_IDS)
    bank_b.val = val_context(bank_b)
    results_b = run_scope(bank_b, 'post_rule_only+restricted_refs', POST_IDS, {},
                          dict(protocol='B', reference_pool=POST_IDS))

    print('== step 5: write outputs ==', flush=True)
    best_nn = min((r for r in results if r['model'] == 'nn_pca' and r['variant'] != 'full_315'),
                  key=lambda r: r['mape'])
    best_ridge = min((r for r in results if r['model'] == 'ridge' and r['variant'] != 'full_315'),
                     key=lambda r: r['mape'])
    payload = dict(
        title='Training-data scope sweep on new events 321-324',
        date='2026-10-19',
        command='OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 '
                '.venv/bin/python -u -B -m bandori.sweep_data_scope',
        standing_scope=dict(events=list(VAL_EVENTS), tiers=list(FOCAL_TIERS), rows=312),
        protocols=dict(
            A='Frozen full-history validation features for every variant; only the '
              'supervised labels (and compression/masks fit on those rows) change.',
            B='post_rule_only with reference_pool=[311..320]; validation case '
              'features rebuilt — not row-matched to Protocol A.'),
        frozen_reference=FROZEN_REFERENCE,
        hyperparameters='frozen; no tuning performed (all 321-324 scores descriptive)',
        scopes=scopes,
        results_protocol_a=results,
        results_protocol_b=results_b,
        best_variant=dict(nn_pca=best_nn['variant'], ridge=best_ridge['variant']),
        caveats=[
            'post_rule_only supervises ids 311-320 only (10 events; 8 start on/after '
            'the 2026-05-26 rule change, 312/313 are pre-rule) — a small supervised set.',
            'The frozen training context holds cases for 310 of the 315 training events '
            '(ids 1, 2, 3, 4, 10 have none); this is identical for every Protocol A '
            'variant including the full_315 baseline.',
            'Recency-weighted variants scale loss weights without renormalizing total '
            'mass, so fixed weight_decay/strength acts relatively stronger there; '
            'compression/masks stay full-bank-fitted for those variants.',
            'weight_sum < 1 everywhere because post-close-tracker proxy labels carry '
            'quality weight < 1.',
            "Protocol B's restricted reference pool logged case-build failures while "
            'rebuilding validation context (tiny tiers 1/10 lost all references, plus '
            'stale-score checkpoints); the scored focal set nonetheless stayed at '
            'exactly 312 rows.',
            'Four test events only; per-event deltas are descriptive, not significant.',
        ],
    )
    save_results(OUT_JSON, payload)

    intro = [
        'Scope: event-balanced MAPE on clean checkpoints of events 321-324, tiers '
        '500/1000/1500/2000 (312 rows), identical for every row below. Both models '
        'are fit with frozen hyperparameters (JointNet hidden 8, seed 1729, gain 1.0; '
        'ridge strength 0.01, gain 0.5) — no tuning was performed, so every '
        '321-324 number is descriptive.',
        '',
        'Protocol A (table below): validation features are the frozen full-history '
        'features for every variant (reference pool = all 315 training events); only '
        'which labeled rows enter the supervised fit — and the compression/masks fit '
        'on those rows — changes. `n_supervised cells (sum w)` is '
        '`np.count_nonzero(prep.weights) (prep.weights.sum())` after any scaling.',
        '',
        f'Frozen baselines reproduced by `validate()`: nn_pca MAPE {_pct(FROZEN_REFERENCE["nn_pca"]["mape"])}, '
        f'ridge {_pct(FROZEN_REFERENCE["ridge"]["mape"])}.',
    ]
    sections = [
        (None, intro),
        ('Ranked results — Protocol A', ranked_table(results)),
        ('Per-event MAPE — Protocol A', per_event_table(results)),
    ] + paired_sections(results) + [
        ('Protocol B — restricted analog reference pool [311..320]',
         ['**These rows are NOT matched to Protocol A**: the reference pool change '
          'rebuilds the validation case features as well. Never mix this table with '
          'the Protocol A table.', '']
         + ranked_table(results_b) + [''] + per_event_table(results_b)),
        ('Supervision retained per scope',
         _table(['variant', 'events', 'cells', 'sum w', 'scaled cells'],
                [[v, s['n_supervised_events'], s['n_supervised_cells'],
                  f"{s['weight_sum']:.4f}",
                  s.get('n_cells_scaled', '—')]
                 for v, s in sorted(scopes.items())])),
        ('Caveats', [
            '- `post_rule_only` supervises ids 311-320 only — 10 events, of which 8 '
            '(311, 314-320) start on/after the 2026-05-26 rule change; 312/313 are '
            'pre-rule events that the id range of the spec includes.',
            '- Only 8-10 events of supervision (vs 315) in the restricted scopes; '
            'compression/masks are fit on the restricted rows only.',
            '- The frozen training context holds cases for 310 of the 315 training '
            'events (ids 1, 2, 3, 4, 10 have none) — identical for every Protocol A '
            'variant including the full_315 baseline, so scope comparisons stay matched.',
            '- Recency-weighted variants scale weights without renormalizing total '
            'mass (fixed decay/strength acts relatively stronger) and keep '
            'full-bank-fitted compression/masks — only loss weights change.',
            '- `sum w < 1` everywhere: proxy labels carry quality weight < 1.',
            "- Protocol B's restricted reference pool logged case-build failures while "
            'rebuilding validation context (tiny tiers 1/10 lost all references, plus '
            'stale-score checkpoints); the scored focal set nonetheless stayed at '
            'exactly 312 rows, but the features differ from Protocol A so the tables '
            'must never be mixed.',
            '- Four test events only; deltas are descriptive, do not overclaim.',
        ]),
    ]
    write_report(OUT_REPORT, 'Training-data scope sweep (new post-rule events 321-324)',
                 sections)
    print('done.', flush=True)


if __name__ == '__main__':
    main()
