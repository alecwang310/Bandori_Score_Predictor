"""Score the Fresh production model on held-out event 325 (both row sets).

Fresh production_model is trained on the 12 strictly-post-rule events
(311, 314-324), fit_cutoff = 325 start. Predictions come from its own runtime
(fresh_predictor.Predictor) driven at each checkpoint; labels are stripped from
the input inside the runtime (restrict(..., labels=False)), and 325 never
entered its fitting.

Self-check: its external 24-point MAPE should reproduce ~7.99% from
reports/fresh_event325_external_20261019 (same points, same cutoffs).

Run:
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/sweep_stack_20261019/fresh_on_325.py
"""
import csv
import json
import time
from pathlib import Path

import numpy as np

from bandori import sweep_harness as H
from bandori.role_analog import build_cases
from collect_bandori_data import timestamp
from fresh_predictor.runtime import Predictor

FOCAL = (500, 1000, 1500, 2000)
OUT = Path('runs/sweep_stack_20261019/fresh_on_325.json')
MODEL = 'runs/fresh_post_rule_v1_20261019/production_model'


def mape(points):
    errs = [p[1] / p[2] - 1 for p in points]
    return dict(rows=len(errs), mape=float(np.mean(np.abs(errs))), bias=float(np.mean(errs)))


def main():
    bank = H.load_bank()
    data = bank.data
    # live input: the live download; runtime strips labels before use
    live = H.Dataset.load('bandori_cn_data', bank.config.tiers) if hasattr(H, 'Dataset') else None
    if live is None:
        from bandori.dataset import Dataset
        live = Dataset.load('bandori_cn_data', bank.config.tiers)
    pred = Predictor(MODEL)
    print(f'fresh model: supervised {len(pred.meta["supervised_event_ids"])} events, '
          f'fit_cutoff {pred.meta["fit_cutoff"]}, coefficients {pred.meta["coefficient_count"]}', flush=True)

    refs = list(bank.train_ids) + [321, 322, 323, 324]
    cases, failures, _ = build_cases(data, bank.config, [325], refs)
    cutoffs = sorted({int(c.cutoff) for c in cases})
    ext = list(csv.DictReader(open('reports/fresh_event325_external_20261019/comparison.csv', encoding='utf-8-sig')))
    ext_pts = {(timestamp(r['issued_at']), int(r['tier'])) for r in ext if r['fresh_input'] == 'True'}
    ext_cutoffs = sorted({k for k, _ in ext_pts})
    missing_ext = [t for t in ext_cutoffs if t not in cutoffs]
    print(f'standing cutoffs: {len(cutoffs)}, external cutoffs: {len(ext_cutoffs)}, not in standing: {missing_ext}', flush=True)

    t0 = time.time()
    values = {}   # (cutoff, tier) -> final
    errors = []
    for t in cutoffs:
        try:
            out = pred.predict(live, 325, float(t))
        except ValueError as e:
            errors.append((t, str(e))); continue
        for r in out['predictions']:
            if r['status'] == 'ok':
                values[(t, int(r['tier']))] = float(r['final'])
    print(f'predicted {len(values)} cells in {time.time()-t0:.1f}s; cutoff errors: {len(errors)}', flush=True)
    for t, e in errors[:5]:
        print('   cutoff error', t, e, flush=True)

    idx = {(int(c.cutoff), c.tier): i for i, c in enumerate(cases)}
    A = [i for k, i in idx.items() if k in ext_pts]
    B = [i for i, c in enumerate(cases) if c.tier in FOCAL]

    def pick(ii):
        out = []
        for i in ii:
            c = cases[i]
            v = values.get((int(c.cutoff), c.tier))
            if v is None:
                continue
            out.append((c, v, data.labels[(325, c.tier)]))
        return out

    known = {'stack_postrule30': dict(external_24pts=0.0453, standing_164rows=0.0774),
             'stack_post20': dict(external_24pts=0.0435, standing_164rows=0.0748),
             'stack_all_pre325': dict(external_24pts=0.0614, standing_164rows=0.0885),
             'ridge_full_refit': dict(external_24pts=0.0431, standing_164rows=0.0879),
             'baseline': dict(external_24pts=0.0865, standing_164rows=0.1473)}
    result = dict(model='Fresh production_model (12 strictly post-rule events, 120 coefs, strength 0.01)',
        supervised_events=pred.meta['supervised_event_ids'], fit_cutoff=pred.meta['fit_cutoff'],
        known_comparison=known, sets={})
    for set_name, ii in (('external_24pts', A), ('standing_164rows', B)):
        pts = pick(ii)
        res = mape(pts)
        res['coverage'] = f'{len(pts)}/{len(ii)}'
        result['sets'][set_name] = res
        print(f'--- {set_name} (coverage {res["coverage"]})', flush=True)
        print(f'  fresh_production  MAPE {100*res["mape"]:6.2f}%  bias {100*res["bias"]:+6.2f}%', flush=True)
        for k, v in known.items():
            print(f'  {k:22s} MAPE {100*v[set_name]:6.2f}%  (prior runs)', flush=True)
    OUT.write_text(json.dumps(result, indent=1, default=float))
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
