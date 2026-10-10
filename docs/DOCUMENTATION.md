# Fingerprint + Iris Biometric Authentication — Technical Documentation

This document describes the implementation in this repository. It is intended to be sufficient for a developer or researcher to understand the project structure, data assumptions, execution paths, model lifecycle, authentication protocol, fusion algorithms, configuration, metrics, and outputs without needing to infer behavior from filenames alone.

## 1. Purpose and scope

The project evaluates deep-learning representations for fingerprint and iris authentication. CNNs are trained as modality-specific classifiers; the trained feature pathways are then used to extract fixed-size embeddings. Authentication is evaluated independently for fingerprint and iris and jointly through a synthetic fingerprint–iris fusion experiment.

The project supports three torchvision CNN backbones:

- `mobilenet_v2`
- `resnet18`
- `efficientnet_b0`

The default embedding dimension is 512. The project evaluates authentication using cosine similarity and reports threshold-free and threshold-dependent metrics.

**Important scope limitation:** the fingerprint and iris datasets do not contain matched samples from the same physical subjects. Fusion pairs held-out subjects from the two independent populations into deterministic *synthetic virtual identities*. The resulting fusion experiments test the implemented pairing/scoring protocol; they do not establish real-world multimodal performance on genuine same-person fingerprint/iris pairs.

## 2. Architecture at a glance

```text
                         config.py
                 (paths, datasets, models,
                training, protocol, plotting)
                            │
                            ▼
                      main.py entrypoint
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
    utils.data_validator             utils.random_utils
             │                             │
             └──────────────┬──────────────┘
                            ▼
                    src/datasets.py
            subject-disjoint train/eval split
                            │
                            ▼
                 src/preprocessing.py
                            │
                            ▼
                     src/models.py
           CNN backbone → embedding → classifier
                            │
                            ▼
                      src/train.py
          checkpoints + histories under results/
                            │
                            ▼
                   src/embeddings.py
              saved embedding artifacts/metadata
                            │
                            ▼
                   python -m experiments
                            │
          ┌─────────────────┴──────────────────┐
          ▼                                    ▼
  experiments/protocols.py             experiments/fusion.py
  standalone protocols                  synthetic multimodal trials
          │                                    │
          └─────────────────┬──────────────────┘
                            ▼
                   experiments/metrics.py
            ROC/AUC, EER, FAR/FRR, accuracy, d'
                            │
                            ▼
               experiments/analysis.py + plots.py
            bootstrap, PCA, DET/ROC/FAR-FRR, comparisons
                            │
                            ▼
                  experiments/io.py + run.py
                   JSON / CSV / NPZ / PNG outputs
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Central source of paths, dataset counts, transforms, training settings, protocol switches, fusion weights, plotting sizes, and CLI defaults. Validates key invariants at import time. |
| `main.py` | Seeds randomness, validates datasets, prepares datasets, chooses a device, trains all configured modality/model combinations, and extracts embeddings. Cache cleanup runs in `finally`. |
| `src/datasets.py` | Generates/loads the subject split, validates split integrity, parses the expected directory/file naming scheme, and returns PyTorch datasets. |
| `src/preprocessing.py` | Defines train-time augmentation and deterministic evaluation transforms. |
| `src/models.py` | Wraps torchvision CNN backbones with a learned embedding layer and classification head. |
| `src/train.py` | DataLoader creation, epoch loop, optimization, mixed precision/performance options, checkpointing, and history output. |
| `src/embeddings.py` | Restores checkpoints and writes embeddings with aligned metadata for later evaluation. |
| `src/authentication.py` | Defines modality protocols, validates embedding artifacts, splits enrollment/verification impressions, creates templates, and scores templates. |
| `experiments/protocols.py` | Applies configurable instance-/subject-level aggregation and computes standalone genuine/impostor score arrays. |
| `experiments/fusion.py` | Pairs held-out subjects into synthetic identities, constructs cross-modal trials, scores score-level and feature-level fusion, and computes modality baselines. |
| `experiments/metrics.py` | Computes ROC, AUC, EER, operating thresholds, FAR/FRR, accuracy, score statistics, d-prime, and TAR-at-FAR. |
| `experiments/analysis.py` | DET curve data, PCA projection, bootstrap confidence intervals, and fusion-improvement calculations. |
| `experiments/plots.py` | Generates score distributions, ROC, DET, FAR/FRR, PCA, training-history, metric-comparison, and fusion-improvement plots. |
| `experiments/io.py` | Artifact paths and JSON/CSV/NPZ serialization. |
| `utils/data_validator.py` | Validates dataset structure and image integrity before the full training pipeline. |
| `utils/device.py` | Chooses CUDA when available, otherwise CPU, and prints device details. |
| `utils/random_utils.py` | Seeds Python, NumPy, and PyTorch and configures deterministic kernels/TF32 behavior. |
| `utils/cleanup.py` | Removes `__pycache__` directories at the end of the main pipeline. |

## 3. Repository and runtime directories

The checked-in source contains `config.py`, `main.py`, `src/`, `experiments/`, `utils/`, `docs/`, `requirements.txt`, and three notebooks. The `datasets/` and `results/` directories are runtime data/artifact locations and may be omitted from source archives because they can be large.

By default, `config.py` resolves paths relative to the repository root:

```text
datasets/
├── fingerprints/
└── iris/

results/
├── subject_split.json
├── models/
├── embeddings/
└── experiments/
    ├── fingerprint/
    ├── iris/
    └── fusion/
```

`PROJECT_ROOT` is computed from the location of `config.py`; the program does not depend on the caller's current working directory for these project paths.

## 4. Dataset assumptions and naming

### 4.1 Fingerprint data

Default configuration:

| Property | Value |
|---|---:|
| Subjects | 500 |
| Fingers per subject | 8 |
| Impressions per finger | 5 |
| Training subjects | 250 |
| Held-out evaluation subjects | 250 |
| Training classes | 250 × 8 = 2,000 |
| All configured classes | 500 × 8 = 4,000 |
| File extension | `.bmp` |
| Expected source dimensions (validator) | 328 × 356 |

Expected paths use a three-digit subject ID and finger names `L0`, `L1`, `L2`, `L3`, `R0`, `R1`, `R2`, `R3`:

```text
datasets/fingerprints/004/L0/004_L0_0.bmp
datasets/fingerprints/004/L0/004_L0_1.bmp
...
datasets/fingerprints/004/R3/004_R3_4.bmp
```

A fingerprint classifier class is **subject + finger**, e.g. `004_L0`. The impression number identifies a sample within that class; it is not a separate class label.

### 4.2 Iris data

Default configuration:

| Property | Value |
|---|---:|
| Subjects | 1,000 |
| Eyes per subject | 2 (`L`, `R`) |
| Impressions per eye | 10 |
| Training subjects | 500 |
| Held-out evaluation subjects | 500 |
| Training classes | 500 × 2 = 1,000 |
| All configured classes | 1,000 × 2 = 2,000 |
| Constructed file extension | `.jpg` (the dataset code constructs this exact filename) |
| Expected source dimensions (validator) | 640 × 480 |

Expected paths:

```text
datasets/iris/004/L/004_L_00.jpg
datasets/iris/004/L/004_L_01.jpg
...
datasets/iris/004/R/004_R_09.jpg
```

An iris classifier class is **subject + eye**, e.g. `004_L`. The eye remains a distinct training class; it is not merged with the opposite eye during training.

### 4.3 Metadata keys

Dataset samples retain separate fields for:

- `subject`: source subject ID;
- `class_key`: subject+finger or subject+eye class;
- `sample_key`: class key plus impression index;
- `impression`: index within the class;
- `path`: source file path;
- `class_id`: integer classifier target for training samples.

The sample key preserves traceability but does not define a classification class.

## 5. Subject-wise split and leakage control

`src/datasets.py` generates a split independently for fingerprint and iris using `RANDOM_SEED`. Subject IDs are shuffled, partitioned into the configured training/evaluation counts, and sorted within each partition. The manifest is saved to `results/subject_split.json` and reused on subsequent runs when valid.

Validation checks that:

1. the configured train and evaluation counts sum to the total number of subjects;
2. train and evaluation subject sets do not overlap;
3. the union covers every expected subject;
4. the counts match the configuration.

The split is subject-disjoint within each modality: no source subject is used to train a model and then evaluated by that modality's held-out protocol. Fusion then uses the held-out subject lists only, as recorded in its pairing/manifest files.

## 6. Image preprocessing

`src/preprocessing.py` separates stochastic training transforms from deterministic evaluation transforms. Both modalities are converted to a consistent three-channel representation and normalized using ImageNet mean and standard deviation.

### 6.1 Geometry and channels

`ResizeAndPad` scales the longest image dimension to `IMAGE_SIZE` while preserving aspect ratio, then centers the image on a square canvas filled with `PAD_FILL`. `EnsureRGB` converts grayscale/palette images to RGB so torchvision backbones always receive three channels. The default `IMAGE_SIZE` is 356.

### 6.2 Fingerprint training transform

The current pipeline applies local contrast enhancement (`EnhanceFingerprintStructure`, using OpenCV CLAHE), resize/pad, random rotation, random affine translation/scale, RGB conversion, tensor conversion, and ImageNet normalization. The default rotation is ±10 degrees, affine translation is `(0.05, 0.05)`, and scale is `(0.95, 1.05)`.

### 6.3 Iris training transform

The iris pipeline applies local contrast enhancement (`EnhanceIrisTexture`, using CLAHE), resize/pad, brightness/contrast jitter, RGB conversion, tensor conversion, and ImageNet normalization. Default brightness and contrast jitter values are `0.1`.

### 6.4 Evaluation transform

Evaluation uses deterministic transforms: image geometry/channel normalization without stochastic augmentation. This ensures enrollment and verification embeddings are generated using a stable transform.

## 7. CNN and embedding model

`src/models.py` defines `BiometricEmbeddingModel`. The forward path is conceptually:

```text
RGB image
   ↓
CNN backbone (MobileNetV2 / ResNet18 / EfficientNet-B0)
   ↓
feature pooling / flattening
   ↓
Linear(feature_dim, EMBEDDING_DIM)
   ↓
BatchNorm1d(EMBEDDING_DIM)
   ↓
ReLU
   ↓
embedding vector (512 dimensions by default)
   ↓
Linear(EMBEDDING_DIM, number_of_training_classes)
   ↓
class logits used for training
```

The torchvision backbone can initialize from pretrained ImageNet weights. The classifier head is modality-specific because the number of classes differs between fingerprint and iris. The embedding layer learns a compact representation optimized through the classification objective. During authentication, the classifier logits are not used as similarity scores; the embedding vector is extracted and compared by cosine similarity.

The same architecture family is trained independently for fingerprint and iris. Do not assume that weights are shared across modalities.

## 8. Training pipeline

`main.py` executes the following sequence:

1. `utils.random_utils.set_seed()` configures Python, NumPy, and PyTorch random seeds and deterministic/performance flags.
2. `utils.data_validator.validate_all()` checks expected directory structure, file counts/names, image integrity, and sample properties. If validation fails, the pipeline returns before training.
3. `src.datasets.prepare_datasets()` loads/generates the subject split and creates fingerprint/iris train and evaluation datasets.
4. `utils.device.get_device()` selects CUDA if available, otherwise CPU; device information is printed.
5. `src.train.train_all_models()` trains each configured architecture for each modality.
6. `src.embeddings.extract_all_embeddings()` loads the trained checkpoints and saves embeddings and aligned metadata for the train/evaluation datasets.
7. `utils.cleanup.cache_cleanup()` runs in the `finally` block to remove Python bytecode cache folders.

Training options are centralized in `config.py`. The source includes configurable batch size, epochs, learning rate, weight decay, DataLoader workers, AMP and AMP dtype, save-best-only behavior, deterministic mode, TF32, channels-last, prefetch factor, `torch.compile`, and optimizer-state saving. Exact behavior depends on those values and hardware; benchmark results should record the configuration and device.

The training loop records histories and writes checkpoints/history artifacts below `results/`. `src/embeddings.py` restores a checkpoint, runs the model in inference mode, extracts the embedding vector, and saves a serialized artifact with embeddings plus subject/class/sample/impression/path metadata. Experiment evaluation relies on these artifacts rather than re-running CNN inference for every metric or plot.

## 9. Embedding artifacts and expected fields

An embedding artifact is a serialized PyTorch object loaded by `experiments/io.py`. The core fields used by evaluation are:

| Field | Meaning |
|---|---|
| `embeddings` | 2-D tensor `[number_of_samples, embedding_dimension]` |
| `subjects` | Source subject IDs aligned with the rows |
| `class_keys` | Subject+finger or subject+eye key per row |
| `sample_keys` | Unique class+impression traceability key |
| `impressions` | Impression index within a class |
| `paths` | Original image paths |

Evaluation validates alignment, non-empty embeddings, class sample counts, impression indices, and sample-key consistency. A legacy artifact without `sample_keys` can have them derived from class keys and impression numbers.

The experiment pipeline loads evaluation embeddings and restricts the authentication protocol to held-out subjects from the split manifest. Embeddings for training subjects may exist, but they are not part of the headline held-out evaluation.

## 10. Enrollment and verification protocol

Enrollment and verification use different impressions from each biometric class. The configured counts are:

| Modality | Impressions/class | Enrollment | Verification |
|---|---:|---:|---:|
| Fingerprint | 5 | 2 (indices 0–1) | 3 (indices 2–4) |
| Iris | 10 | 4 (indices 0–3) | 6 (indices 4–9) |

`src/authentication.py` validates the complete embedding artifact and then splits each class by impression index. Enrollment embeddings are averaged and normalized to create templates. Verification embeddings are retained for scoring or aggregated according to the protocol switches.

### 10.1 Instance-level template construction

When enrollment is instance-level, each fingerprint finger or iris eye gets a separate template:

- Fingerprint: average the two enrollment embeddings within each subject+finger class, giving 8 templates per evaluation subject.
- Iris: average the four enrollment embeddings within each subject+eye class, giving 2 templates per evaluation subject.

For the configured evaluation populations this yields 2,000 fingerprint templates and 1,000 iris templates only if all 500 iris evaluation subjects are used. In the standalone protocol, all held-out subjects for the modality are evaluated; fusion may use only the first `EXPERIMENT_PAIRS` paired subjects.

### 10.2 Subject-level enrollment

When enrollment is subject-level, embeddings are L2-normalized and averaged across all classes belonging to a subject, then the resulting mean is normalized. Fingerprint enrollment combines the subject's eight fingers; iris enrollment combines both eyes. This changes the authentication unit from a particular finger/eye to the subject's pooled biometric representation.

### 10.3 Instance-level verification

Every held-out impression remains an individual query. For standalone instance-level evaluation, the matching finger/eye template is genuine; templates associated with other authentication units are impostors.

### 10.4 Subject-level verification

Verification can be pooled across the subject's instances. The implementation groups the same verification impression index across the subject's fingers/eyes and averages those embeddings into a subject-level query. Thus fingerprint produces three verification templates per subject and iris produces six. This preserves multiple verification observations while making each query represent the subject-level scope.

## 11. Authentication granularity switches

Four independent Boolean values in `config.py` control enrollment and verification granularity:

```python
STANDALONE_VIRTUAL_ENROLLMENT = True
STANDALONE_VIRTUAL_VERIFICATION = True
FUSION_VIRTUAL_ENROLLMENT = False
FUSION_VIRTUAL_VERIFICATION = True
```

The current values shown are the defaults in the supplied implementation.

| Value | Enrollment interpretation | Verification interpretation |
|---|---|---|
| `True` | Keep individual finger/eye templates separate | Keep individual finger/eye queries separate |
| `False` | Pool enrollment instances into one subject template | Pool each verification impression across the subject's instances |

Enrollment and verification are controlled independently. The four combinations are useful for protocol ablations, but mixed settings should be named explicitly in any report. The `VIRTUAL` word in these switch names is historical/configuration terminology; for standalone protocols it does not mean the source identity itself is synthetic. In fusion, virtual identity specifically refers to the synthetic cross-dataset pairing described below.

## 12. Standalone experiment path

The CLI entry point is `experiments/__main__.py`, which calls `experiments.run.main()`.

For `--modality fingerprint` or `--modality iris`, `run_modality()`:

1. loads the model's evaluation embedding artifact;
2. calls `evaluate_single_modality()` with the modality protocol and configured granularity;
3. calculates metrics from genuine and impostor scores;
4. writes score arrays to NPZ and metrics to JSON/CSV;
5. optionally generates score distribution, ROC, DET, FAR/FRR, and PCA plots;
6. loads training history and generates a training comparison plot;
7. writes a comparison of EER and accuracy at the EER threshold across selected models.

`--models` accepts a comma-separated subset of `mobilenet_v2`, `resnet18`, and `efficientnet_b0`. The experiment runner evaluates existing artifacts; it does not train models. Run `python3 main.py` first if the required checkpoints and embedding artifacts do not exist.

## 13. Synthetic multimodal fusion

### 13.1 Why virtual identities are used

The source fingerprint and iris datasets have different subject IDs and are not paired observations of the same people. The fusion implementation therefore pairs held-out fingerprint and iris subjects deterministically. Each virtual identity contains one fingerprint subject and one iris subject; the mapping is written to `results/experiments/fusion/pairing.json` and the run manifest.

The default `EXPERIMENT_PAIRS = 250` means the headline experiment uses 250 pairs, limited by `VIRTUAL_PAIRING_LIMIT`. The held-out lists are sorted independently and paired by position. This is a deterministic synthetic construction, not a claim of real cross-modal identity correspondence.

### 13.2 Building modality-specific fusion data

`evaluate_fusion_pair()` maps each selected source subject to a virtual ID, filters the embedding artifacts to the paired held-out subjects, and independently builds fingerprint and iris enrollment templates and verification queries using `FUSION_VIRTUAL_ENROLLMENT` and `FUSION_VIRTUAL_VERIFICATION`.

The fingerprint and iris queries are then combined as a Cartesian product within each virtual identity. Integer index arrays are used for the scoring path to avoid repeatedly looking up string keys in the inner loop.

### 13.3 Query-pair counts

With instance-level enrollment and verification:

- Fingerprint: 8 finger classes × 3 verification impressions = 24 queries per virtual identity.
- Iris: 2 eye classes × 6 verification impressions = 12 queries per virtual identity.
- Cartesian product: 24 × 12 = 288 multimodal query pairs per virtual identity.
- At 250 virtual identities: 72,000 multimodal query pairs.

With subject-level enrollment and verification:

- Fingerprint: 1 enrollment template and 3 subject-level verification queries.
- Iris: 1 enrollment template and 6 subject-level verification queries.
- Cartesian product: 3 × 6 = 18 multimodal query pairs per virtual identity, or 4,500 for 250 identities.

Mixed-granularity settings are supported. Query counts and genuine-template matching rules can therefore differ from the two all-instance/all-subject examples above. The actual configuration is saved to the fusion manifest.

### 13.4 Fused enrollment templates

For each virtual identity, the implementation forms a Cartesian product of its fingerprint and iris enrollment templates:

- Instance-level enrollment: 8 × 2 = 16 fused templates per virtual identity; for 250 virtual identities, 4,000 fused templates.
- Subject-level enrollment: 1 × 1 = 1 fused template per virtual identity; for 250 virtual identities, 250 fused templates.

A fused template records one fingerprint enrollment template and one iris enrollment template. A genuine multimodal match requires the relevant fingerprint and iris components to match the query's authentication unit under the selected granularity rules.

### 13.5 Genuine and impostor score construction

The system first calculates modality-specific query-to-template cosine-similarity matrices. Each cross-modal query pair indexes a fingerprint query row and an iris query row. The fusion scoring stage combines those scores against the fused enrollment templates. The genuine mask identifies the enrolled fused template(s) corresponding to the target virtual identity and biometric instance(s); all other comparisons form impostor scores.

Because the same modality query can appear in multiple Cartesian query pairs, resulting scores are correlated. The nominal number of scores is not the same as the number of statistically independent biometric observations.

### 13.6 Memory/performance behavior

Instance-level fusion can produce a large all-to-all comparison set. The scoring implementation processes cross-modal query pairs in batches set by `FUSION_SCORE_BATCH_SIZE` (default 1024), using vectorized PyTorch tensor gathers/operations rather than nested Python-level loops. CUDA is used for the scoring stage if available; otherwise it uses CPU tensors. The score distributions, ROC/EER computations, plotting, and serialization include CPU/NumPy work, so chunked scoring bounds intermediate tensor work but does not make the entire experiment constant-memory. Reducing the batch size lowers per-batch intermediate memory at the cost of more batches; it does not reduce the total number of trials.

## 14. Score-level fusion

For fingerprint similarity score \(S_{fp}\), iris similarity score \(S_{iris}\), and fingerprint weight \(w\):

\[
S_{fusion}=wS_{fp}+(1-w)S_{iris}
\]

`FUSION_FINGERPRINT_WEIGHT` defaults to `0.5`; the iris weight is the complement. The two modality similarities are computed independently using cosine similarity against their modality templates. The resulting fused scores are then evaluated like other authentication scores.

This is a score-space operation: the embeddings are not concatenated to form the score-level fusion representation.

## 15. Feature-level fusion

Feature-level fusion uses a different scoring space from score-level fusion:

1. Stack the modality's enrollment templates and fit a per-dimension mean and population standard deviation using enrollment templates only.
2. Use unit scale for dimensions whose standard deviation is at most `1e-6`; this avoids amplifying near-constant dimensions.
3. Standardize each enrollment template and corresponding verification embedding with that modality's fitted mean and scale.
4. Multiply standardized fingerprint features by \(\sqrt{w}\) and standardized iris features by \(\sqrt{1-w}\).
5. Concatenate the weighted fingerprint and iris vectors.
6. L2-normalize the concatenated vectors and compute cosine similarity between each fused query and fused enrollment template.

The standardizer is fitted only from enrollment templates. Verification samples and verification labels are not used to fit its mean/scale. This avoids fitting the transformation to the held-out verification distribution.

Why the standardization matters: if the original independently normalized modality vectors were simply concatenated with square-root weights, the cosine similarity of the concatenation would equal the weighted sum of the individual cosine similarities (assuming the usual unit-normalized vectors). That would make feature-level and score-level scores algebraically identical. The enrollment-only dimension-wise standardization changes relative coordinate scales and therefore makes the current feature-level scoring function genuinely distinct. Metrics can still happen to be close, but equality is no longer required by construction.

The feature-level PCA plot uses the feature-level fused enrollment representation produced by this pipeline.

## 16. Fusion baselines and interpretation

The fusion experiment also reports `fingerprint_baseline` and `iris_baseline` scores alongside `score_level` and `feature_level`. These baselines help compare fusion against the modality-specific systems used within the same fusion protocol.

Comparisons must state the enrollment and verification granularity. A subject-level pooled fingerprint baseline is not the same system as a standalone single-finger protocol. A change in EER can come from multi-instance pooling, multimodal fusion, or both. To isolate the effect of fusion, compare like-for-like protocols and explain the synthetic pairing caveat.

`fusion_delta_records()` computes improvement against the stronger same-pair baseline as implemented in `experiments/analysis.py`; the output CSV/JSON and plots are diagnostic summaries, not a substitute for examining the underlying metric rows.

## 17. Metric definitions

Metrics are calculated from one array of genuine similarity scores and one array of impostor similarity scores. Larger cosine similarity means a stronger match; scores at or above the decision threshold are accepted.

### 17.1 ROC and AUC

The ROC curve plots false acceptance rate against true acceptance rate over score thresholds. In this implementation the false-positive rate corresponds to FAR and true-positive rate corresponds to TAR. AUC summarizes discrimination over the full threshold range.

### 17.2 FAR and FRR

- **FAR (False Acceptance Rate):** fraction of impostor scores accepted at a threshold.
- **FRR (False Rejection Rate):** fraction of genuine scores rejected at a threshold.
- **TAR (True Acceptance Rate):** one minus FRR.

The threshold increases/decreases the trade-off between false accepts and false rejects. `FAR_TARGETS` defaults to `(1e-2, 1e-3, 1e-4)`; TAR is reported at thresholds chosen for those impostor FAR targets where the finite sample permits.

### 17.3 EER

Equal Error Rate is the point where FAR and FRR are equal or approximately equal. The implementation derives the ROC rates and thresholds and estimates EER from the crossing. It also records the corresponding decision threshold and evaluates FAR/FRR at that threshold.

### 17.4 Accuracy at EER

Accuracy is computed at the EER threshold:

\[
\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

Here genuine scores at or above threshold are true accepts, while impostor scores below threshold are true rejects. Accuracy depends on the relative counts of genuine and impostor comparisons, so it must be read alongside EER, FAR, FRR, and AUC. It is not the primary biometric metric by itself.

### 17.5 Score statistics and d-prime

The report includes mean, sample standard deviation, and median for genuine and impostor scores. d-prime is calculated from the difference in means divided by the pooled standard deviation used by the implementation. Degenerate zero pooled variance is handled with a finite value for serialization.

### 17.6 Bootstrap intervals

Bootstrap intervals are optional and disabled by default (`EXPERIMENT_DEFAULT_BOOTSTRAP_ITERATIONS = 0`). The CLI accepts `--bootstrap-iterations` and `--bootstrap-seed`. The analysis code caps sampled genuine/impostor arrays using configured limits to avoid unbounded resampling cost. Because Cartesian fusion scores are correlated, bootstrap intervals should be interpreted with care; the resampling scheme does not make the source trials independent.

## 18. Plot conventions

`experiments/plots.py` contains all plotting functions. The current plotting conventions are:

- **Score distributions:** density histograms for genuine and impostor similarity scores.
- **ROC:** linear FAR/TAR axes from 0 to 1, random-classifier diagonal, and a zoomed low-FAR/high-TAR inset to expose small deviations from the ideal corner.
- **DET:** false acceptance and false rejection probabilities shown on normal-deviate/probit axes, with standard percentage tick labels and a low-error inset. Endpoints are clipped for display because the normal inverse CDF is infinite at zero and one.
- **FAR/FRR:** cosine threshold on a linear x-axis; logarithmic error-rate y-axis with a display floor of `1e-3`. The floor only affects plotted values, not metric calculations.
- **PCA:** deterministic two-dimensional projection, with point count limited by `PCA_MAX_POINTS`.
- **Comparison bars:** EER/accuracy fusion comparisons group score-level and feature-level bars for every fingerprint-model × iris-model pair.
- **Fusion improvement bars:** EER/AUC improvement plots group score-level and feature-level bars within each model-pair category.
- **Training history:** training loss and epoch duration for the selected model histories.

Plot labels and scale choices are designed for comparison and legibility; a plotted floor, clipped endpoint, inset, or axis transform does not change the saved scores or metric values.

## 19. Configuration reference

All settings below are declared in `config.py`. The file itself is authoritative; this table describes the supplied implementation's intent and defaults.

### 19.1 Paths

| Setting | Default/meaning |
|---|---|
| `PROJECT_ROOT` | Directory containing `config.py` |
| `DATASET_DIR` | `<PROJECT_ROOT>/datasets` |
| `FINGERPRINT_DIR` | `<DATASET_DIR>/fingerprints` |
| `IRIS_DIR` | `<DATASET_DIR>/iris` |
| `RESULTS_DIR` | `<PROJECT_ROOT>/results` |
| `MODELS_DIR` | `results/models` |
| `EMBEDDINGS_DIR` | `results/embeddings` |
| `METRICS_DIR` | `results/metrics` (reserved path constant; experiment outputs are organized under `results/experiments`) |

### 19.2 Dataset and identity settings

| Setting | Default |
|---|---:|
| `FINGERPRINT_NUM_SUBJECTS` | 500 |
| `FINGERPRINT_NUM_FINGERS_PER_SUBJECT` | 8 |
| `FINGERPRINT_IMPRESSIONS_PER_FINGER` | 5 |
| `FINGERPRINT_TRAIN_SUBJECTS` / `FINGERPRINT_EVAL_SUBJECTS` | 250 / 250 |
| `FINGERPRINT_NUM_CLASSES` | total subjects × fingers |
| `FINGERPRINT_TRAIN_CLASSES` | training subjects × fingers |
| `IRIS_NUM_SUBJECTS` | 1,000 |
| `IRIS_EYES` | `("L", "R")` |
| `IRIS_IMPRESSIONS_PER_EYE` | 10 |
| `IRIS_TRAIN_SUBJECTS` / `IRIS_EVAL_SUBJECTS` | 500 / 500 |
| `IRIS_NUM_CLASSES` | total subjects × eyes |
| `IRIS_TRAIN_CLASSES` | training subjects × eyes |
| `FINGERPRINT_EXTENSIONS` | `.bmp` |
| `IRIS_EXTENSIONS` | `.jpg`, `.jpeg` |
| `FINGERPRINT_IMAGE_WIDTH` / `HEIGHT` | 328 / 356, for validation |
| `IRIS_IMAGE_WIDTH` / `HEIGHT` | 640 / 480, for validation |

### 19.3 Reproducibility, model, and embedding settings

| Setting | Default | Purpose |
|---|---|---|
| `RANDOM_SEED` | 37 | Random seed |
| `MODEL_NAMES` | three supported architectures | Architectures trained/evaluated |
| `MODALITIES` | fingerprint, iris | Supported standalone modalities |
| `IMAGE_SIZE` | 356 | Resize/pad target side length |
| `EMBEDDING_DIM` | 512 | Embedding layer output dimension |
| `IMAGENET_MEAN` / `IMAGENET_STD` | standard ImageNet values | RGB normalization |

### 19.4 Authentication and fusion settings

| Setting | Default | Purpose |
|---|---|---|
| `FINGERPRINT_ENROLL_IMPRESSIONS` | 2 | Enrollment impressions per finger class |
| `FINGERPRINT_VERIFY_IMPRESSIONS` | derived as 3 | Verification impressions per finger class |
| `IRIS_ENROLL_IMPRESSIONS` | 4 | Enrollment impressions per eye class |
| `IRIS_VERIFY_IMPRESSIONS` | derived as 6 | Verification impressions per eye class |
| `VIRTUAL_PAIRING_LIMIT` | minimum modality subject count (500) | Maximum possible source-subject pairings |
| `EXPERIMENT_PAIRS` | 250 | Headline fusion pairing count |
| `FUSION_FINGERPRINT_WEIGHT` | 0.5 | Fingerprint score/feature contribution; iris weight is complement |
| `FUSION_SCORE_BATCH_SIZE` | 1024 | Query-pair batch size for fusion scoring |
| `STANDALONE_VIRTUAL_ENROLLMENT` | `True` | Standalone enrollment granularity |
| `STANDALONE_VIRTUAL_VERIFICATION` | `True` | Standalone verification granularity |
| `FUSION_VIRTUAL_ENROLLMENT` | `False` | Fusion enrollment granularity |
| `FUSION_VIRTUAL_VERIFICATION` | `True` | Fusion verification granularity |

The config validates that train/evaluation subject counts cover each dataset, `EXPERIMENT_PAIRS` is positive and within the pairing limit, fusion weight is in `[0, 1]`, score batch size is positive, granularity switches are Boolean, and enrollment/verification impression counts are positive.

### 19.5 Training settings

| Setting | Default | Purpose |
|---|---|---|
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 30 | Epoch count |
| `LEARNING_RATE` | `1e-4` | Optimizer learning rate |
| `WEIGHT_DECAY` | `1e-4` | Optimizer weight decay |
| `NUM_WORKERS` | `os.cpu_count()` | DataLoader worker count |
| `USE_AMP` | `True` | Automatic mixed precision switch |
| `SAVE_BEST_ONLY` | `True` | Checkpoint selection behavior |
| `TRAIN_DETERMINISTIC` | `True` | Deterministic training configuration |
| `ENABLE_TF32` | `True` | TF32 acceleration where supported |
| `USE_CHANNELS_LAST` | `False` | Channels-last memory format switch |
| `PREFETCH_FACTOR` | 2 | DataLoader prefetch setting |
| `AMP_DTYPE` | `"float16"` | AMP dtype selection |
| `USE_TORCH_COMPILE` | `False` | `torch.compile` switch |
| `SAVE_OPTIMIZER_STATE` | `True` | Whether optimizer state is checkpointed |

### 19.6 Analysis and plot settings

| Setting | Default | Purpose |
|---|---|---|
| `FAR_TARGETS` | `(1e-2, 1e-3, 1e-4)` | Target FAR values for TAR reporting |
| `BOOTSTRAP_DEFAULT_ITERATIONS` | 0 | Default bootstrap resamples (disabled) |
| `BOOTSTRAP_DEFAULT_CONFIDENCE` | 0.95 | Default confidence level |
| `BOOTSTRAP_DEFAULT_SEED` | `RANDOM_SEED` | Bootstrap seed |
| `BOOTSTRAP_MAX_GENUINE_SAMPLES` | 2,000 | Cap for bootstrap genuine sampling |
| `BOOTSTRAP_MAX_IMPOSTOR_SAMPLES` | 100,000 | Cap for bootstrap impostor sampling |
| `PCA_MAX_POINTS` | 5,000 | Maximum PCA points |
| `PLOT_DPI` | 160 | Saved plot resolution |
| `SCORE_HISTOGRAM_BINS` | 100 | Histogram bins |
| `PLOT_*_FIGSIZE` | defined tuples | Per-plot dimensions |
| `PLOT_FUSION_DELTA_HEIGHT` | 5 | Fusion improvement plot height |
| `PLOT_FUSION_DELTA_MIN_WIDTH` | 10 | Minimum fusion improvement plot width |
| `PLOT_SCATTER_SIZE` | 8 | PCA scatter point size |
| `EXPERIMENT_DEFAULT_MODALITY` | `"all"` | CLI default modality |
| `EXPERIMENT_DEFAULT_BOOTSTRAP_ITERATIONS` | 0 | CLI bootstrap default |
| `EXPERIMENT_DEFAULT_BOOTSTRAP_SEED` | 37 | CLI bootstrap seed |

## 20. Command-line reference

Run commands from the repository root.

```bash
python3 main.py
python3 -m experiments --modality fingerprint
python3 -m experiments --modality iris
python3 -m experiments --modality fusion
python3 -m experiments --modality all
```

Options implemented by `experiments.run`:

| Option | Meaning |
|---|---|
| `--modality {fingerprint,iris,fusion,all}` | Select one or more experiment families |
| `--models model1,model2` | Evaluate a comma-separated subset of supported architectures |
| `--no-plots` | Skip plot generation; metrics/scores are still written |
| `--bootstrap-iterations N` | Number of bootstrap resamples; 0 disables bootstrap intervals |
| `--bootstrap-seed N` | Base seed for bootstrap resampling |
| `--include-nonheldout-diagnostic` | Also run the separately labeled diagnostic path after fusion |

The non-held-out diagnostic is intentionally separate from headline results and may include training subjects. Do not merge those records into the main held-out performance claims.

## 21. Output files and schemas

### 21.1 Standalone outputs

Under `results/experiments/fingerprint/` and `results/experiments/iris/`, the runner writes:

- `metrics.json` and `metrics.csv`: one row per evaluated architecture;
- `scores/<model>.npz`: genuine and impostor score arrays;
- `bootstrap_intervals.json` when bootstrap is enabled;
- `manifest.json`: modality, models, protocol, bootstrap settings, and granularity switches;
- `plots/`: distributions, ROC, DET, FAR/FRR, PCA, training, and comparisons.

### 21.2 Fusion outputs

Under `results/experiments/fusion/`, the runner writes:

- `metrics.json` and `metrics.csv`: one row for each model pair and method (`score_level`, `feature_level`, `fingerprint_baseline`, `iris_baseline`);
- `scores/<fp_model>__<iris_model>__<method>.npz`: genuine and impostor score arrays;
- `pairing.json`: mapping from virtual IDs to source fingerprint/iris subjects and the synthetic-identity warning;
- `manifest.json`: selected model set, pairing counts, protocol switches, and bootstrap configuration;
- `fusion_delta.json` and `fusion_delta.csv`: improvement summaries against same-pair baselines;
- `bootstrap_intervals.json` when enabled;
- `plots/`: per-pair/method diagnostics and grouped comparisons.

Metric rows contain model/protocol identifiers, cohort and trial counts, and the fields returned by `MetricResult.to_dict()`. Large per-threshold arrays (`thresholds`, `far`, `frr`) are removed from the aggregate metric rows and stored/visualized through the relevant analysis paths; the FAR/FRR plot receives them directly during the run.

The saved NPZ score files are the source material for reproducing score distributions and many of the plots without re-running model training. Keep `config.py`, the split manifest, fusion pairing map, and experiment manifests with any reported results.

## 22. Reproducibility and interpretation checklist

Before reporting a result, record:

1. the dataset version and exact file layout;
2. the saved subject split and random seed;
3. architecture and embedding dimension;
4. training hyperparameters, compute device, and mixed-precision settings;
5. the four enrollment/verification granularity switches;
6. fusion weight and fusion method;
7. number of synthetic pairs and the synthetic-identity caveat;
8. genuine/impostor trial counts and metric definitions;
9. whether bootstrap intervals were enabled;
10. whether results came from the held-out headline experiment or the non-held-out diagnostic.

Do not compare scores without checking protocol consistency. A change from finger/eye templates to subject-pooled templates changes the task and can improve performance independently of multimodal fusion. The independent source datasets also mean that the current fusion results cannot establish true same-person cross-modal complementarity.

## 23. Testing and notebooks

The supplied archive contains the three experiment notebooks, but does not contain a `tests/` directory. If tests are added or available in another working copy, run the project's test suite from the repository root using the command defined by that test setup. Do not assume tests are present in this archive.

The notebooks (`experiment_fingerprint.ipynb`, `experiment_iris.ipynb`, and `experiment_fused.ipynb`) are exploratory/presentation entry points. The reproducible CLI and `experiments/` modules are the canonical implementation for full experiment runs; notebooks should be treated as consumers of project artifacts unless their cells explicitly perform a separate computation.

## 24. Limitations and non-goals

- **Synthetic multimodal pairing:** source subjects are independently sampled and not genuine same-person pairs.
- **Correlated Cartesian trials:** a verification embedding may be reused in many query pairs, so score counts are not independent observation counts.
- **Granularity dependence:** subject pooling and instance-level authentication represent different tasks.
- **Accuracy dependence:** accuracy depends on threshold and genuine/impostor balance; prioritize EER, FAR/FRR, and AUC for biometric interpretation.
- **Resource requirements:** full training across six modality/model combinations and instance-level fusion can be expensive in runtime, RAM, and storage. Chunked scoring limits intermediate fusion tensors but all resulting score arrays and metric operations still consume resources.
- **Not production-ready:** the code is an experimental research implementation. It does not provide liveness detection, secure template storage, biometric privacy guarantees, calibration for deployment, or production access-control hardening.

## 25. End-to-end quick reference

```bash
# 1. Put datasets in datasets/fingerprints and datasets/iris.
# 2. Review config.py and the four protocol switches.
# 3. Validate, train, and extract embeddings.
python3 main.py

# 4. Evaluate each modality independently.
python3 -m experiments --modality fingerprint
python3 -m experiments --modality iris

# 5. Evaluate synthetic multimodal fusion.
python3 -m experiments --modality fusion

# 6. Inspect results/experiments/{fingerprint,iris,fusion}/.
```

For a report, describe the fusion experiment as synthetic virtual fusion, include the granularity settings, and distinguish score-level from the enrollment-standardized feature-level method.
