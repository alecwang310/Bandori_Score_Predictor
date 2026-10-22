# Bandori CN score predictor

A prediction-only Python program for the Chinese server. It includes frozen model
weights, the historical reference bank needed for analog forecasts, and live API
retrieval. Python 3.10 or newer is required; NumPy is the only external dependency.

```sh
python3 -m pip install -r requirements.txt
python3 predict.py
```

Press Enter to select the latest CN event, then Enter again for its most recent
08:00 forecast time. To use current observations directly:

```sh
python3 predict.py --event latest --as-of now
```

The program fetches public CN data from Bestdori, with HHWX as a fallback. It
caches responses in `cache/` and saves forecasts in `output/`. Those directories
are created locally and excluded from Git. `--no-save` only prints the forecast.
No account or API key is required.

The default Chinese bulletin reports two separate forecasts: the previous equal
average of Ridge and Stack, followed by recent rule/state Ridge. It does not
average those two configurations. To select one explicitly:

```sh
python3 predict.py --event latest --as-of now --model ridge
python3 predict.py --event latest --as-of now --model average
```

`--model previous_ridge` and `--model stack_mf` expose the average's components.
`--report full` shows every available tier. JSON output always includes all tiers.
Unzoned times are China time. An issue time limits what observations may be used;
the prediction target is always the score at event close. Scores normally must
be no more than one hour old.

For a previously saved input snapshot:

```sh
python3 predict.py --event 326 --as-of "2026-10-17 20:00" --snapshot input.json.gz
```

Use `--save-snapshot input.json.gz` to retain the inputs to an online forecast.
`--data /path/to/bandori_cn_data` reads an existing completed download. This
package does not download an entire historical dataset or train models.

Uncertainty uses the frozen calibration for each model and tier. The notation
`1.23(45) * 10^6` means `(1.23 +/- 0.45) * 10^6`. These are approximate historical
RMS error scales, including bias; they do not guarantee normal errors or coverage.
Uncalibrated tiers and unsupported horizons are marked in the bulletin.

Keep the `models/` directory beside `predict.py`. The shared historical bank is
required at prediction time; it is included once for all models. No research
reports, training code, experiment tools or training command are included here.
