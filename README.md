# ForenSort

ForenSort is a research prototype that estimates **regional quality in latent
fingerprint images** using a convolutional neural network. It predicts a
quality level for each image region, then aggregates those predictions into an
advisory image score that can help prioritize images for human review.

> **Important:** ForenSort is advisory research software. It does not identify
> people, match fingerprints, or provide forensic conclusions. Its predictions
> must not replace qualified human examination.

## What the project does

ForenSort:

1. Reads a grayscale latent fingerprint image.
2. Divides the image into regions.
3. Predicts a quality level from **0 to 5** for each region.
4. Calculates average quality, usable-region coverage, confidence, and an
   advisory image score.
5. Displays a regional quality map and lets users review multiple images.

The model is trained on expert regional quality annotations from NIST SD 302h.
The repository contains the complete model architecture, data pipeline,
training code, evaluation code, and Streamlit interface.

## Web application

The live website has **not been deployed yet**.

To run the application locally:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Then open the local address shown by Streamlit, usually:

<http://localhost:8501>

When a public deployment is available, its URL should be added here.

## Model

The model is defined in [`src/model.py`](src/model.py). It uses:

- A pretrained ResNet-18 encoder adapted for one-channel grayscale images.
- A lightweight multi-scale decoder.
- A six-class regional quality head.
- Class-weighted cross-entropy to handle imbalanced quality levels.
- Optional ordinal loss because quality levels are ordered.

The trained checkpoint is intentionally not included in this repository because
it is large and depends on restricted training data. Place a compatible
checkpoint at:

```text
outputs/checkpoints/baseline_best.pt
```

## Results

### Held-out NIST test split

The test split was kept separate from training and model selection:

| Metric | Result |
|---|---:|
| Accuracy | 0.745 |
| Macro-F1 | 0.581 |
| Quadratic weighted kappa | 0.822 |
| Mean absolute quality-level error | 0.289 |

Performance is strongest on common quality levels. Rare high-quality regions
are more difficult for the current model, so these results should be treated as
baseline research results rather than production guarantees.

### Independent CSFD-10k check

An independent evaluation on 100 evenly spaced CSFD-10k images produced:

- Spearman correlation with MOS: **0.607**
- Pearson correlation with MOS: **0.641**

This indicates a moderate relationship between ForenSort's advisory scores and
the independent human-rated quality scores. A larger evaluation should be run
before drawing stronger conclusions.

## Repository structure

```text
app/                 Streamlit review application
configs/             Training configurations
notebooks/           Colab training notebook
outputs/figures/     Example visualizations
outputs/metrics/     Evaluation metrics and edge-case results
reports/             Dataset and project reports
scripts/             Packaging utilities
src/model.py         Neural network architecture and losses
src/dataset.py       Dataset loading and preprocessing
src/train.py         Training entry point
src/test.py          Held-out test evaluation
src/inference.py     Inference helpers used by the app
src/evaluate.py      Metric calculation
src/evaluate_csfd10k.py  Independent CSFD-10k evaluation
```

## Training and evaluation

Training is intended to run on Google Colab or another machine with a suitable
GPU:

```powershell
python src/train.py configs/baseline.yaml
python src/test.py configs/baseline.yaml outputs/checkpoints/baseline_best.pt
```

The optional ordinal experiment can be run with:

```powershell
python src/train.py configs/ordinal.yaml
python src/test.py configs/ordinal.yaml outputs_ordinal/checkpoints/ordinal_best.pt
```

For the independent CSFD-10k check:

```powershell
python src/evaluate_csfd10k.py `
  external_data/csfd10k/CSFD-images.zip `
  external_data/csfd10k/im_names.txt `
  external_data/csfd10k/mos.txt `
  outputs/checkpoints/baseline_best.pt
```

Use `--all` only when a complete evaluation is intended.

## Data and privacy

The repository does **not** include restricted fingerprint images, raw dataset
archives, `.lffs` files, or trained weights. Obtain the data through the
appropriate official channels and place it in the local paths expected by the
scripts. Do not upload restricted images or personal data to public
repositories.

## Limitations

- This is a research prototype, not a validated forensic product.
- The independent CSFD-10k run currently uses a 100-image sample.
- High-quality classes are underrepresented and have lower F1 scores.
- The advisory score is not interchangeable with a standardized quality scale.
- Predictions can be unreliable for unusual, blank, noisy, or out-of-distribution
  images.

## License and dataset terms

Dataset access, usage, and redistribution are subject to the terms of the
respective dataset providers. Check those terms before downloading or sharing
any data.
