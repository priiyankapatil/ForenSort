# ForenSort

CNN-based latent fingerprint quality assessment (research prototype, advisory only).
Not for identification or forensic conclusions. See the project plan PDF for full scope.

## Status
- [x] Phase 1 scaffold + dataset decision template (`reports/dataset_decision.md`)
- [x] Phase 2 dataset inspection (`src/audit_dataset.py`)
- [x] Phase 3 loader / baseline
- [x] Phase 4 training (baseline trained on Colab; best validation macro-F1 0.582, QWK 0.813)
- [x] Phase 5 evaluation (test accuracy 0.745, QWK 0.822, MAE 0.29)
- [x] Phase 6 Streamlit app (`app/streamlit_app.py`)
- [x] Phase 6b model improvement experiment (`configs/ordinal.yaml`) — tested; baseline retained
- [x] Phase 7 OpenLQM comparison (1,563 valid results; 2 OpenLQM timeouts)

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Training should run on Google Colab (no local GPU).

## Local review app

After placing the trained checkpoint at `outputs/checkpoints/baseline_best.pt`:

```
streamlit run app/streamlit_app.py
```

The app accepts PNG, JPG, BMP, TIFF, and NIST `.lffs` transactions. For LFFS
files it extracts the embedded image automatically and can show the embedded
expert quality map for reference. It displays a regional quality overlay and
is advisory only; it does not identify people or support forensic conclusions.
It also reports model confidence and warns about low-contrast or uncertain
inputs. Confidence is a model score, not a guarantee that a prediction is
correct.
Multiple files can be uploaded together. The app sorts them by an advisory
image score based on average regional quality and usable-region coverage, then
lets you inspect each prediction individually. High/Medium/Low categories are
review-order suggestions, not forensic conclusions.

### Example test

The app was manually tested with a held-out NIST LFFS sample. The model
processed a 647 x 937 image and achieved 89.9% regional agreement with its
embedded expert map, with a 0.10 average quality-level error. This is one
example, not a replacement for the aggregate test results below.

The aggregate held-out test results were 74.5% accuracy, 0.581 macro-F1,
0.822 quadratic weighted kappa, and 0.29 mean absolute quality-level error.
Stress tests also covered blank, black, noisy, tiny, wide, and very large
inputs; they completed without crashes. Synthetic inputs are not meaningful
quality assessments and should be treated as out-of-distribution cases.

### Optional model improvement experiment

The baseline is kept unchanged. The bundled experiment keeps the successful
baseline class weighting and adds an ordinal distance penalty, because quality
levels 0-5 are ordered. This avoids the failed aggressive weighting experiment
in `configs/high_quality.yaml` and does not overwrite baseline outputs.

```
python src/train.py configs/ordinal.yaml
python src/test.py configs/ordinal.yaml outputs_ordinal/checkpoints/ordinal_best.pt
```

Keep the experiment only if its test results improve the high-quality classes
without causing an unacceptable drop in overall QWK or MAE.

The ordinal experiment was evaluated on the held-out test split:

| Model | Accuracy | Macro-F1 | QWK | MAE |
|---|---:|---:|---:|---:|
| Baseline | 0.745 | 0.581 | 0.822 | 0.29 |
| Ordinal | 0.743 | 0.577 | 0.820 | 0.29 |

The baseline remains the selected model because the ordinal loss did not
improve the aggregate test metrics.

### OpenLQM comparison

OpenLQM is used only as an external benchmark; it is not used as a training
label. After installing the official Windows release, compare both systems on
the same held-out test images:

```
python src/compare_openlqm.py configs/baseline.yaml outputs/checkpoints/baseline_best.pt
```

The script writes `outputs/openlqm_comparison/comparison.csv` and
`outputs/openlqm_comparison/summary.json`. OpenLQM's OQ score and ForenSort's
image score use different scales, so the comparison uses Spearman rank
correlations rather than treating the raw scores as interchangeable.

On the 1,563 images with valid OpenLQM output, the rank correlations were:

| Comparison | Spearman correlation |
|---|---:|
| OpenLQM OQ vs expert mean quality | 0.396 |
| ForenSort image score vs expert mean quality | 0.836 |
| OpenLQM OQ vs ForenSort image score | 0.471 |

Two images exceeded the 30-second OpenLQM timeout and were recorded as
failures rather than stopping the benchmark. This comparison is specific to
the NIST SD 302 test split and is not evidence that either system is
universally better.

## Data
Request NIST SD 302 via https://www.nist.gov/itl/iad/btg/nist-special-database-302 and place files under `data/raw/`.
Raw data and manifests are git-ignored; never publish restricted images.

## Audit
```
python src/audit_dataset.py data/raw/<subset> [--group-regex "<regex>"] [--write]
```
