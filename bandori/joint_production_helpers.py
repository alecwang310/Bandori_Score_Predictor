"""LOO tuning grid and out-of-fold passes for the tuned joint production model."""
from .sweep_harness import fit_jointnet, predict as corr_predict, prepare_fit, score, supervised_slice
from .regime_suite import finish_context


GRID = tuple((h, e, lr, d)
             for h in (4, 8, 16)
             for e in (120, 240, 480)
             for lr in (0.003, 0.01, 0.03)
             for d in (0.001, 0.003, 0.01))
GAINS = (0.5, 0.75, 1.0, 1.25)
SEEDS = tuple(1729 + k for k in range(8))


def _fold(bank, scope, held):
    """Fitted compression/masks + held-out context for one LOO fold."""
    fit_ids = [e for e in scope if e != held]
    prep = prepare_fit(bank, fit_ids)
    hold = supervised_slice(bank.ctx_all, [held])
    hold, _ = finish_context(hold, prep.compression, prep.masks)
    return prep, hold


def loo_pass1(bank, scope):
    """Grid over configs x gains; score each by LOO focal MAPE on events 311-320."""
    rows = {}
    for held in scope:
        if held not in range(311, 321):
            continue  # scope includes older recent events; tune only on post-rule ids
        prep, hold = _fold(bank, scope, held)
        X = hold['matrices']['pca']
        for hidden, epochs, lr, decay in GRID:
            net = fit_jointnet(prep.matrix['pca'], prep.targets, prep.weights, prep.bins,
                               bank.config, hidden=hidden, seed=1729,
                               epochs=epochs, learning_rate=lr, decay=decay)
            delta = net.predict(X)
            for gain in GAINS:
                vals = corr_predict(hold['cases'], hold['index'], delta, gain, 'nn')
                res = score(hold['cases'], vals, bank.data)
                key = (hidden, epochs, lr, decay, gain)
                acc = rows.setdefault(key, dict(mape=[], bias=[]))
                acc['mape'].append(res['mape'])
                acc['bias'].append(res['bias'])
    out = []
    for (hidden, epochs, lr, decay, gain), acc in rows.items():
        out.append(dict(hidden=hidden, epochs=epochs, lr=lr, decay=decay, gain=gain,
                        mape=float(sum(acc['mape']) / len(acc['mape'])),
                        bias=float(sum(acc['bias']) / len(acc['bias'])),
                        folds=len(acc['mape'])))
    return out


def loo_pass2(bank, scope, best):
    """Out-of-fold per-case rows for the winning config, all tiers (sigma input)."""
    rows = []
    for held in scope:
        if held not in range(311, 321):
            continue
        prep, hold = _fold(bank, scope, held)
        net = fit_jointnet(prep.matrix['pca'], prep.targets, prep.weights, prep.bins,
                           bank.config, hidden=best['hidden'], seed=1729,
                           epochs=best['epochs'], learning_rate=best['lr'], decay=best['decay'])
        delta = net.predict(hold['matrices']['pca'])
        vals = corr_predict(hold['cases'], hold['index'], delta, best['gain'], 'nn')
        for c, v in zip(hold['cases'], vals):
            truth = bank.data.labels.get((c.event_id, c.tier))
            if truth is None or truth <= 0 or v <= 0:
                continue
            age = (c.cutoff - c.observed_at) / 3600
            rows.append(dict(event_id=c.event_id, tier=c.tier, ridge=float(v), actual=float(truth),
                             hours_remaining=c.remaining_hours,
                             scenario='gap_24h' if age > 12 else 'gap_6h' if age > 2 else 'clean',
                             event_end=bank.data.events[c.event_id].end, checkpoint=c.cutoff))
        res = score(hold['cases'], vals, bank.data)
        print(f'  LOO OOF {held}: MAPE {100*res["mape"]:.2f}%', flush=True)
    return rows
