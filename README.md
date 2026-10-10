# Fingerprint + Iris Biometric Authentication

A research and experimentation project for **deep-learning-based fingerprint and iris authentication**. It trains CNN-based embedding models, evaluates each modality independently, and investigates two ways of combining fingerprint and iris evidence. The code is organized so that training, embedding extraction, authentication protocols, fusion, metrics, and plotting can be run and inspected separately.

> **Research caveat:** the available fingerprint and iris datasets contain independent subject populations. The fusion experiment pairs held-out subjects into deterministic *synthetic virtual identities*. These results are useful for testing the implemented fusion pipeline, but they are **not** results from genuinely paired fingerprint and iris samples belonging to the same people.

## Contents

- [Highlights](#highlights)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Dataset layout](#dataset-layout)
- [Installation and setup](#installation-and-setup)
- [Run the pipeline](#run-the-pipeline)
- [Run authentication experiments](#run-authentication-experiments)
- [Configure the protocol](#configure-the-protocol)
- [Fusion methods](#fusion-methods)
- [Metrics and plots](#metrics-and-plots)
- [Outputs](#outputs)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Further documentation](#further-documentation)

## Highlights

- CNN backbones: **MobileNetV2**, **ResNet18**, and **EfficientNet-B0**.
- Fixed-size, 512-dimensional biometric embeddings by default.
- Subject-disjoint training/evaluation splits, persisted in a split manifest.
- Separate fingerprint and iris enrollment/verification protocols.
- Independent controls for instance-level and subject-level enrollment and verification.
- Synthetic fingerprint–iris fusion using score-level and feature-level methods.
- Evaluation with ROC AUC, EER, FAR, FRR, accuracy at the EER threshold, d-prime, and TAR at configured FAR targets.
- Reproducible experiment artifacts, JSON/CSV metrics, saved genuine/impostor score arrays, and plots.
- Chunked tensor scoring for computationally expensive instance-level fusion.

## Repository structure

```text
Fingerprint-Iris-Authentication/
├── config.py                    # Dataset, training, protocol, fusion and plotting settings
├── main.py                      # Validate data, prepare splits, train models, extract embeddings
├── requirements.txt             # Python dependencies
├── src/
│   ├── datasets.py              # Dataset classes, subject split, split manifest
│   ├── preprocessing.py         # Train/evaluation image transforms
│   ├── models.py                # CNN backbone and embedding/classification heads
│   ├── train.py                 # Training loop, checkpoints and training history
│   ├── embeddings.py            # Checkpoint loading and embedding extraction
│   └── authentication.py        # Enrollment split, templates, validation and similarity scoring
├── experiments/
│   ├── __main__.py              # Enables `python -m experiments`
│   ├── run.py                   # CLI, experiment orchestration and output writing
│   ├── protocols.py             # Standalone authentication protocols and aggregation
│   ├── fusion.py                # Virtual pairing, fusion scoring and fusion baselines
│   ├── metrics.py               # ROC, AUC, EER and operating-point metrics
│   ├── analysis.py              # DET data, PCA, bootstrap intervals and fusion deltas
│   ├── plots.py                 # Metric and diagnostic plots
│   └── io.py                    # Load/save artifacts, metrics and score arrays
├── utils/
│   ├── data_validator.py        # Dataset structure and image validation
│   ├── device.py                # CPU/CUDA selection and device information
│   ├── random_utils.py          # Random seeds and deterministic/performance flags
│   └── cleanup.py               # Removes generated Python bytecode caches
├── docs/
│   └── DOCUMENTATION.md         # Detailed architecture and implementation guide
├── experiment_fingerprint.ipynb # Fingerprint exploration/experiment notebook
├── experiment_iris.ipynb        # Iris exploration/experiment notebook
└── experiment_fused.ipynb       # Fusion exploration/experiment notebook
```

`datasets/` and `results/` are runtime data/output directories and may not be included in a source archive. The program creates output subdirectories when needed. The actual dataset files must be placed in the expected layout described below.

## Requirements

- Python version compatible with the installed PyTorch and torchvision builds.
- PyTorch and torchvision (the supplied requirements pin CUDA 13.0 builds).
- A CUDA-capable NVIDIA GPU is recommended for training, but the code selects CPU if CUDA is unavailable.
- Enough storage for the source datasets, model checkpoints, embeddings, and experiment score arrays.

The exact pinned PyTorch versions in `requirements.txt` are intentional project dependencies. Install a wheel compatible with your Python version and platform if those particular wheels are unavailable for your environment.

## Dataset layout

The paths are controlled by `config.py` and default to `datasets/fingerprints/` and `datasets/iris/`.

Fingerprint images are expected at:

```text
datasets/fingerprints/<subject_id>/<finger>/<subject_id>_<finger>_<impression>.bmp
```

Example:

```text
datasets/fingerprints/004/L0/004_L0_0.bmp
datasets/fingerprints/004/L0/004_L0_1.bmp
...
datasets/fingerprints/004/R3/004_R3_4.bmp
```

There are eight finger classes per subject: `L0`–`L3` and `R0`–`R3`. Subject IDs are zero-padded to three digits in paths.

Iris images are expected at:

```text
datasets/iris/<subject_id>/<eye>/<subject_id>_<eye>_<impression:02d>.jpg
```

Example:

```text
datasets/iris/004/L/004_L_00.jpg
datasets/iris/004/L/004_L_01.jpg
...
datasets/iris/004/R/004_R_09.jpg
```

The default configuration expects 500 fingerprint subjects with 8 fingers and 5 impressions per finger, and 1,000 iris subjects with left/right eyes and 10 impressions per eye. Image extensions, counts, and expected source dimensions are defined in `config.py`. `main.py` runs structural and image-integrity checks before training.

## Default evaluation protocol

The classifier learns a separate class for each subject+finger (fingerprint) or subject+eye (iris). Enrollment and verification use disjoint impressions:

| Modality | Evaluation subjects | Biometric units per subject | Impressions per unit | Enrollment | Verification |
|---|---:|---:|---:|---:|---:|
| Fingerprint | 250 | 8 fingers | 5 | 2 per finger | 3 per finger |
| Iris | 500 | 2 eyes | 10 | 4 per eye | 6 per eye |

With the default instance-level standalone settings, that is 2,000 fingerprint enrollment templates and 6,000 fingerprint verification queries; iris has 1,000 enrollment templates and 6,000 verification queries. Subject-level pooling changes the unit of authentication and produces one enrollment template per subject with 3 fingerprint or 6 iris verification templates per subject.

The default headline fusion experiment pairs 250 held-out fingerprint subjects with 250 held-out iris subjects. With instance-level fusion enrollment and verification, the cross-product is 16 fused enrollment templates and 288 multimodal query pairs per virtual identity. Actual counts depend on the four granularity switches in `config.py`.

## Installation and setup

From the repository root:

```bash
# Create an environment (optional if you already use a suitable shared environment).
python3 -m venv .venv
source .venv/bin/activate

# Upgrade the installer and install the project dependencies.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate`. If the pinned CUDA PyTorch wheels cannot be resolved, install a PyTorch/torchvision pair compatible with your Python, OS, and CUDA runtime using the official PyTorch installation instructions, then install the remaining dependencies from `requirements.txt` without replacing that compatible pair.

Before running the pipeline:

1. Place the datasets in the directory layout above.
2. Review `config.py`, especially the dataset counts, train/evaluation split sizes, training parameters, and authentication granularity switches.
3. Ensure sufficient disk space under `results/`.

## Run the pipeline

### Train all configured models and extract embeddings

```bash
python3 main.py
```

This entry point seeds the random number generators, validates both datasets, creates or loads the subject-wise split, prepares training and evaluation datasets, selects CPU/CUDA, trains the configured models, and extracts embeddings for downstream authentication experiments. Model checkpoints, histories, the subject split, and embeddings are saved under `results/`.

The configured models are trained for both modalities, so a full run can be computationally expensive. Pretrained torchvision weights may need to be downloaded on the first run unless already cached.

### Run experiments using existing artifacts

The experiment CLI evaluates existing embedding artifacts; it does **not** retrain the CNNs.

```bash
# Fingerprint only
python3 -m experiments --modality fingerprint

# Iris only
python3 -m experiments --modality iris

# Synthetic fusion
python3 -m experiments --modality fusion

# Run fingerprint, iris, and fusion experiments
python3 -m experiments --modality all
```

Choose a subset of architectures with `--models`:

```bash
python3 -m experiments --modality fusion --models mobilenet_v2,resnet18
```

Skip plot generation with `--no-plots`:

```bash
python3 -m experiments --modality fusion --no-plots
```

Bootstrap confidence intervals are disabled by default. Enable them with, for example:

```bash
python3 -m experiments --modality fusion --bootstrap-iterations 200 --bootstrap-seed 37
```

The full virtual-population diagnostic is an explicitly separate, non-headline path and can be requested with `--include-nonheldout-diagnostic`. Its output must not be presented as a held-out evaluation result because it may include subjects used during training.

## Configure the protocol

All primary settings are in `config.py`. The four granularity switches are independent:

```python
STANDALONE_VIRTUAL_ENROLLMENT = True
STANDALONE_VIRTUAL_VERIFICATION = True

FUSION_VIRTUAL_ENROLLMENT = False
FUSION_VIRTUAL_VERIFICATION = True
```

- `True`: retain separate biometric instances (each finger or eye).
- `False`: pool biometric instances to the subject level by averaging embeddings.

The defaults above intentionally configure standalone experiments at instance level and fusion at subject-level enrollment with instance-level verification. Set both switches for a workflow to `True` for instance-level enrollment and verification, or both to `False` for subject-level enrollment and verification. Mixed settings are supported as ablations; document the exact settings alongside any reported result.

The current default fusion settings are a mixed-granularity protocol. Do not describe results as fully subject-level fusion unless both fusion switches are `False`.

Other important settings include:

| Setting | Purpose |
|---|---|
| `RANDOM_SEED` | Seed for deterministic splitting and random operations |
| `MODEL_NAMES` | CNN architectures evaluated by the pipeline |
| `IMAGE_SIZE` | Square input resolution after resize/pad |
| `EMBEDDING_DIM` | Output embedding dimension (default 512) |
| `BATCH_SIZE`, `NUM_EPOCHS` | Training batch size and epochs |
| `LEARNING_RATE`, `WEIGHT_DECAY` | Optimizer hyperparameters |
| `USE_AMP`, `AMP_DTYPE` | Mixed-precision training configuration |
| `TRAIN_DETERMINISTIC`, `ENABLE_TF32` | Reproducibility/performance controls |
| `FUSION_FINGERPRINT_WEIGHT` | Fingerprint contribution to score-level fusion; iris weight is its complement |
| `FUSION_SCORE_BATCH_SIZE` | Maximum number of cross-modal query pairs scored in one fusion batch |
| `EXPERIMENT_PAIRS` | Number of held-out source-subject pairs in headline fusion |
| `FAR_TARGETS` | FAR targets for TAR reporting |
| `PCA_MAX_POINTS`, `SCORE_HISTOGRAM_BINS` | Plot/analysis limits |

See [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) for every configuration section, algorithms, data flow, metric definitions, output formats, and implementation details.

## Fusion methods

- **Score-level fusion:** compute fingerprint and iris cosine similarities separately and combine them with `FUSION_FINGERPRINT_WEIGHT` and `1 - FUSION_FINGERPRINT_WEIGHT`.
- **Feature-level fusion:** fit a per-dimension standardizer using enrollment templates only for each modality, standardize the corresponding query/template embeddings, apply square-root modality weights, concatenate the feature vectors, L2-normalize, and compute cosine similarity. Dimensions with near-zero enrollment standard deviation use unit scale to avoid amplifying numerical noise.

Feature-level fusion is not merely square-root-weighted concatenation of the original normalized embeddings: the enrollment-only dimension-wise standardization is what makes its scoring space distinct from score-level fusion. The transform is fitted on enrollment templates only to avoid fitting it to verification samples. See the technical documentation for equations and trial construction.

Fusion is computed in vectorized, chunked tensor batches controlled by `FUSION_SCORE_BATCH_SIZE`. CUDA is used for the scoring stage when available. The final score arrays and metric calculations are handled using NumPy/CPU operations, so a full instance-level run can still be memory-intensive.

## Metrics and plots

Metrics include:

- ROC AUC
- Equal Error Rate (EER) and its threshold
- FAR and FRR at the EER threshold
- Accuracy at the EER threshold
- Genuine/impostor score mean, standard deviation, and median
- d-prime
- Genuine and impostor trial counts
- TAR at configured FAR targets

Accuracy is threshold-dependent and is influenced by the number of genuine and impostor scores. Interpret it together with EER, FAR/FRR, and AUC, not as a standalone measure.

Plots include genuine/impostor score distributions, ROC curves, DET curves, FAR/FRR versus threshold, PCA projections, training history, EER/accuracy comparisons, and fusion improvement comparisons. ROC plots use conventional linear FAR/TAR axes with a zoomed high-performing region. DET plots use normal-deviate probability axes and a low-error inset. FAR/FRR plots use a logarithmic error-rate axis with a display floor; the display floor does not change the stored metric values.

## Outputs

Typical generated paths include:

```text
results/
├── subject_split.json
├── models/                         # Model checkpoints
├── embeddings/                     # Saved train/evaluation embeddings
└── experiments/
    ├── fingerprint/
    │   ├── metrics.json
    │   ├── metrics.csv
    │   ├── scores/
    │   └── plots/
    ├── iris/
    │   ├── metrics.json
    │   ├── metrics.csv
    │   ├── scores/
    │   └── plots/
    └── fusion/
        ├── metrics.json
        ├── metrics.csv
        ├── fusion_delta.json
        ├── fusion_delta.csv
        ├── pairing.json
        ├── manifest.json
        ├── scores/
        └── plots/
```

Exact artifact filenames are generated from modality/model names. Fusion comparison plots include grouped score-level versus feature-level bars for each fingerprint–iris architecture pair. `pairing.json` records the deterministic synthetic pairing and prominently documents that the identities are synthetic.

## Reproducibility and limitations

- Subject splits are persisted in `results/subject_split.json`; the loader validates the saved split before using it.
- The default seed is `37`. Keep the split manifest and configuration with reported results.
- Fusion pairs independent fingerprint and iris evaluation subjects. It is not genuine paired multimodal evaluation.
- Cartesian products reuse the same modality query samples in multiple trials. Trial scores are therefore not all statistically independent; avoid interpreting raw trial count as an independent-sample count.
- Enrollment/verification granularity changes the authentication unit and can change the results substantially. Compare systems only after stating their protocol settings.
- The local data, pretrained weights, checkpoints, and experiment outputs are not included in this source archive.

## Further documentation

Read [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) for the detailed code architecture, tensor/data flow, training and inference behavior, standalone and fusion trial semantics, configuration reference, metrics, plot conventions, and output schemas.
