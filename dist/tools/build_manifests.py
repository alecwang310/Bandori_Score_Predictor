#!/usr/bin/env python3
"""(Re)generate dist/models/fresh/manifest.json and dist/models/ridge/manifest.json.

The stack manifest is produced by export_stack_package.py together with its
package. This tool only records static provenance plus the sha256 of the
package assets; it never touches model weights. Run from anywhere:

    .venv/bin/python dist/tools/build_manifests.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'dist' / 'models'
FIT_CUTOFF = 1790647200.0   # start of event 325


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assets_of(package_dir):
    return {p.name: sha(p) for p in sorted(package_dir.iterdir())
            if p.is_file() and p.name != 'manifest.json'}


def fresh_manifest():
    package = MODELS / 'fresh'
    meta = json.loads((package / 'model.json').read_text())
    assert meta['fit_cutoff'] == FIT_CUTOFF
    assert meta['supervised_event_ids'] == [311, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324]
    return dict(
        name='fresh', kind=meta['kind'], api_version=1, fit_cutoff=meta['fit_cutoff'],
        supervised_events=len(meta['supervised_event_ids']),
        assets=assets_of(package),
        training_provenance=dict(
            recipe='fresh_predictor research fit (120 joint partially-shared coefficients, '
                   'strength 0.01); production assets runs/fresh_post_rule_v1_20261019/'
                   'production_model, packaged for distribution (model.json '
                   'exported_from_model_sha256)',
            scope='12 strictly-post-rule supervised events (311, 314-324; 312/313 start pre-rule); '
                  '319-event curve-reference bank through 324 for completion/pace priors',
            supervised_event_ids=meta['supervised_event_ids'],
            reference_events=len(meta['reference_event_ids']),
            fit_cutoff_basis='start of event 325 = 1790647200; labels of 321-324 verified available '
                             'before 325; events 325/326 excluded from fitting',
            sigma='calibrated from prequential forecasts on events 317-320 only (at most four '
                  'events per tier, same-tier horizon cells, no cross-tier borrowing)',
            weighting='730-day feature time decay; 3x weight for post-rule events; '
                      'reference events are never used as supervised labels',
            validation=dict(scope='held-out event 325 (descriptive single holdout)',
                            external_24pts_mape=0.07991951604650417,
                            standing_164rows_mape=0.10634931606797905)),
        limitations=[
            'Fits only 12 strictly-post-rule events (311, 314-324); rejects events that started '
            'before the 2026-05-26 rule change',
            'Events 325/326 are excluded from fitting (fit_cutoff = start of event 325)',
            'Held-out 325: 7.99% external-24 / 10.63% standing-164 MAPE — a single inspected '
            'holdout, not a fresh independent test',
            'T1/T10 have no supervised labels; their forecasts fall back to priors',
            'Sigma uses at most four 317-320 events per tier: small proxy-label sample, contains '
            'systematic bias, no independent coverage or normality guarantee',
            'Reference/analog pool stops at event 324'])


def ridge_manifest():
    package = MODELS / 'ridge'
    deployment = json.loads((package / 'deployment.json').read_text())
    assert deployment['kind'] == 'bandori_cn_ridge_deployment_v1'
    assert deployment['gain'] == 0.5
    assert deployment['fit_cutoff'] == FIT_CUTOFF
    assert len(deployment['training_event_ids']) == 319
    return dict(
        name='ridge', kind=deployment['kind'], api_version=1, fit_cutoff=deployment['fit_cutoff'],
        supervised_events=len(deployment['training_event_ids']),
        assets=assets_of(package),
        training_provenance=dict(
            recipe='JointRidge full refit (bandori/advanced_statistics) over 319 events through '
                   '324 with PCA inputs (rich compression + 24h observer summaries + rule '
                   'metadata), shrinkage strength 0.01, 730-day decay and 3x post-rule weighting',
            scope='319 events through 324 supervised and used as the analog/reference pool; '
                  'events 325/326 excluded from fitting',
            strength=0.01,
            strength_selection='only searched parameter; chosen by strict chronological forward '
                               'selection on events 317-320 without reading 321-324 outcomes',
            multiplier='1 + gain*delta with gain 0.5 (1+gain·delta rule)',
            sigma='sigma.json: out-of-fold residuals from the previously inspected 12 post-rule '
                  'events (time-held-out; horizon/scenario cells; T1/T10 borrow T20)',
            distribution_source='packaged as old/dist_fresh/long_event_model; same family as '
                                'runs/rule_aware_3x_20261018/full_refit/models; '
                                'source_model_sha256=' + deployment['source_model_sha256'],
            fit_cutoff_basis='start of event 325 = 1790647200',
            validation=dict(scope='held-out event 325 (descriptive single holdout)',
                            external_24pts_mape=0.04307469034481867,
                            standing_164rows_mape=0.08790822825824854)),
        limitations=[
            'Trained on 319 events through 324; events 325/326 are excluded from fitting',
            'Held-out 325: 4.31% external-24 / 8.79% standing-164 MAPE — descriptive, previously '
            'inspected holdout, not a fresh independent test',
            'Strength was selected on 317-320 only; the model design reused earlier experiments, '
            'so historical checks are not fully independent of design iteration',
            'Sigma comes from 12 post-rule events with closing-tracker proxy labels: systematic '
            'bias, no 68% coverage guarantee, not specially calibrated for extra-long events; '
            'T1/T10 explicitly borrow T20',
            'No clock_v2 variant is shipped; the frozen daily_v1 feature layout is the deployment'])


def main():
    for builder in (fresh_manifest, ridge_manifest):
        manifest = builder()
        path = MODELS / manifest['name'] / 'manifest.json'
        path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, sort_keys=True) + '\n')
        print('wrote', path.relative_to(ROOT), f"({manifest['supervised_events']} supervised events, "
              f"{len(manifest['assets'])} assets)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
