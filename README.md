# Fingerprint + Iris Biometric Authentication

A deep-learning-based multimodal biometric authentication project using CNN embeddings for fingerprint and iris recognition. The project supports MobileNetV2, ResNet18, and EfficientNet-B0, standalone authentication experiments, synthetic fingerprint+iris fusion, and configurable instance-level vs subject-level authentication protocols.

## Project structure

```text
Fingerprint-Iris-Authentication/
├── config.py                    # Central user-editable configuration
├── main.py                      # Training + embedding extraction pipeline
├── requirements.txt
├── src/
│   ├── datasets.py              # Dataset loading and subject splits
│   ├── models.py                # CNN backbone dispatch
│   ├── preprocessing.py
│   ├── train.py
│   ├── embeddings.py
│   └── authentication.py        # Enrollment, verification, templates, scoring
├── experiments/
│   ├── __main__.py
│   ├── run.py                   # Experiment orchestration
│   ├── protocols.py             # Configurable authentication protocols
│   ├── fusion.py                # Synthetic FP+iris fusion
│   ├── metrics.py               # AUC, EER, FAR/FRR, accuracy, etc.
│   ├── analysis.py
│   ├── plots.py
│   └── io.py
├── tests/
│   └── test_experiments.py
├── ui/
│   ├── app.py                    # Flask/Jinja manual authentication UI
│   ├── templates/
│   └── static/
├── datasets/
├── results/
├── experiment_fingerprint.ipynb
├── experiment_iris.ipynb
├── experiment_fused.ipynb
└── DOCUMENTATION.md
```

## Quick start

```bash
# Full training + embedding extraction
python3 main.py

# Standalone fingerprint
python3 -m experiments --modality fingerprint

# Standalone iris
python3 -m experiments --modality iris

# Synthetic fusion
python3 -m experiments --modality fusion

# All experiments
python3 -m experiments --modality all
```

Use `--no-plots` to skip plot generation.

## Pipeline

The project has three major stages:

1. **Training** — CNNs learn discriminative biometric representations from labeled fingerprint/iris classes.
2. **Embedding extraction** — the classifier backbone produces fixed-size embeddings.
3. **Authentication** — enrollment embeddings form templates; held-out embeddings are verification queries; cosine similarity produces genuine and impostor scores.

Training classes remain modality-specific:

- Fingerprint: `subject + finger`
- Iris: `subject + eye`

The authentication protocol can then operate either at this instance level or after pooling instances to the subject level.

## Enrollment and verification

The configured datasets contain:

### Fingerprint

- 500 subjects
- 8 fingers per subject
- 5 impressions per finger
- 2 enrollment impressions
- 3 verification impressions

At instance level:

- 2 samples from each finger are averaged into one finger template.
- 8 fingers × 250 evaluation subjects = **2,000 enrollment templates**.
- 3 verification samples × 8 fingers × 250 subjects = **6,000 verification queries**.

At subject level:

- All enrollment embeddings from the subject's 8 fingers are averaged into one subject template.
- Each verification impression is averaged across the 8 fingers, producing 3 subject-level verification templates per subject.

### Iris

- 1,000 subjects
- 2 eyes per subject
- 10 impressions per eye
- 4 enrollment impressions
- 6 verification impressions

At instance level:

- 4 samples from each eye are averaged into one eye template.
- 2 eyes × 250 evaluation subjects = **500 enrollment templates**.
- 6 verification samples × 2 eyes × 250 subjects = **3,000 verification queries**.

At subject level:

- All enrollment embeddings from both eyes are averaged into one subject template.
- Each verification impression is averaged across both eyes, producing 6 subject-level verification templates per subject.

## Authentication granularity configuration

The following four switches in `config.py` control enrollment and verification independently:

```python
STANDALONE_VIRTUAL_ENROLLMENT = True
STANDALONE_VIRTUAL_VERIFICATION = True

FUSION_VIRTUAL_ENROLLMENT = False
FUSION_VIRTUAL_VERIFICATION = False
```

`True` means **instance-level** operation: individual fingers/eyes remain separate.

`False` means **subject-level** operation: the corresponding fingerprint fingers or iris eyes are pooled by averaging.

The settings are independent so mixed-granularity protocols can also be evaluated for ablation studies.

### Typical standalone configurations

**Instance-level standalone:**

```python
STANDALONE_VIRTUAL_ENROLLMENT = True
STANDALONE_VIRTUAL_VERIFICATION = True
```

This is the standard finger-specific/eye-specific authentication protocol.

**Subject-level standalone:**

```python
STANDALONE_VIRTUAL_ENROLLMENT = False
STANDALONE_VIRTUAL_VERIFICATION = False
```

This measures authentication after multi-finger/multi-eye aggregation.

## Synthetic fusion

The fingerprint and iris evaluation cohorts are independent and do not represent the same physical people. Fusion therefore creates deterministic **synthetic virtual identities** by pairing held-out fingerprint and iris subjects positionally.

For example:

```text
Virtual ID 0 = fingerprint evaluation subject A + iris evaluation subject B
Virtual ID 1 = fingerprint evaluation subject C + iris evaluation subject D
...
```

This is a synthetic fusion experiment, not a claim that the two source subjects are the same person.

### Instance-level fusion

With:

```python
FUSION_VIRTUAL_ENROLLMENT = True
FUSION_VIRTUAL_VERIFICATION = True
```

each finger and eye remains separate.

Enrollment:

- 8 fingerprint finger templates
- 2 iris eye templates
- Cartesian product = **16 fused enrollment templates per virtual subject**

Verification:

- 8 fingers × 3 samples = 24 fingerprint queries
- 2 eyes × 6 samples = 12 iris queries
- Cartesian product = **288 multimodal query pairs per virtual subject**

With 250 virtual subjects:

- **72,000 multimodal verification query pairs**

Each query pair is scored against the appropriate combined enrollment templates. A fused template is genuine only when both its fingerprint and iris components match the query's authentication unit.

**Performance note:** Instance-level fusion can produce a large all-to-all score set (72,000 query pairs against 4,000 fused enrollment templates for the 250-subject configuration). The implementation therefore constructs fusion scores in vectorized tensor batches controlled by `FUSION_SCORE_BATCH_SIZE` (default `1024`) instead of using nested Python loops. CUDA is used automatically for this tensor-scoring stage when available; metric calculation and plotting remain CPU/NumPy based.

### Subject-level fusion

With:

```python
FUSION_VIRTUAL_ENROLLMENT = False
FUSION_VIRTUAL_VERIFICATION = False
```

each virtual subject has:

- 1 fingerprint enrollment template from all 8 fingers
- 1 iris enrollment template from both eyes
- 3 fingerprint verification templates
- 6 iris verification templates

The verification Cartesian product is therefore:

\[
3 	imes 6 = 18
\]

multimodal query pairs per virtual subject, or **4,500** for 250 virtual subjects.

### Important interpretation

The fusion baseline and standalone baseline can represent different authentication granularities. In particular, a subject-level fusion fingerprint baseline should not be directly compared with a single-finger standalone EER as if they were the same system. Subject-level aggregation itself can substantially improve performance.

## Fusion methods

Two fusion methods are evaluated:

- **Score-level fusion** — weighted combination of fingerprint and iris cosine similarities.
- **Feature-level fusion** — weighted concatenation of the two normalized embeddings followed by cosine similarity.

The fingerprint weight is controlled by:

```python
FUSION_FINGERPRINT_WEIGHT = 0.5
```

The iris weight is `1 - FUSION_FINGERPRINT_WEIGHT`.

## Metrics

Each experiment saves:

- AUC
- EER
- EER threshold
- FAR at EER
- FRR at EER
- **Accuracy at EER threshold**
- Genuine mean/std/median
- Impostor mean/std/median
- d-prime
- Genuine trial count
- Impostor trial count
- TAR at configured FAR targets

Accuracy is defined at the EER operating threshold:

\[
Accuracy = rac{TP + TN}{TP + TN + FP + FN}
\]

where similarity scores at or above the threshold are accepted.

Because biometric experiments can have many more impostor scores than genuine scores, accuracy should be interpreted together with EER, FAR, FRR, and AUC rather than as the sole performance measure.

## Outputs

For each modality:

```text
results/experiments/<modality>/
├── metrics.json
├── metrics.csv
├── bootstrap_intervals.json
├── scores/
└── plots/
    ├── distributions/
    ├── roc/
    ├── det/
    ├── far_frr/
    ├── pca/
    └── comparisons/
        ├── eer_comparison.png
        └── accuracy_comparison.png
```

Fusion additionally produces:

The fusion accuracy comparison is saved at `results/experiments/fusion/plots/comparisons/fusion_accuracy_comparison.png` and includes all fingerprint-model × iris-model combinations, with separate score-level and feature-level bars.

```text
results/experiments/fusion/
├── metrics.json
├── metrics.csv
├── fusion_delta.json
├── fusion_delta.csv
├── pairing.json
├── bootstrap_intervals.json
├── scores/
└── plots/
    ├── det/
    ├── distributions/
    ├── far_frr/
    ├── pca/
    ├── roc/
    └── comparisons/
        ├── fusion_accuracy_comparison.png
        ├── fusion_eer_improvement.png
        └── fusion_auc_improvement.png
```

The notebooks are presentation layers over these experiment artifacts.

## Reproducibility

The subject split and experiment pairing are deterministic under the configured seed. The headline fusion experiment uses `EXPERIMENT_PAIRS` independently held-out fingerprint/iris subject pairs, with a maximum controlled by `VIRTUAL_PAIRING_LIMIT`.

## Local authentication UI

The project includes a Flask/Jinja demonstration UI for running the trained embedding pipeline manually. It supports selecting a modality and trained model, enrolling a user by uploading one or more samples under a supplied subject ID, and then identifying or verifying uploaded samples. When a claimed subject ID is supplied, the UI reports the claimed score, strongest non-claimed score, the saved EER threshold for that model, and the resulting genuine/impostor decision. Enrollments are kept in memory only and are cleared when the process exits or with the **Clear all enrollments** action.

Start it with:

```bash
python3 -m ui.app
```

Then open `http://127.0.0.1:5000/`. The UI uses the same deterministic evaluation preprocessing and trained checkpoints as the experiment pipeline.

## Testing

Run:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover deterministic virtual pairing, trial construction, subject-level verification aggregation, configuration types, and metric reporting.

## Important limitation

The headline fusion experiment uses **synthetic virtual identities** because the available fingerprint and iris datasets have independent subject populations. The fusion results therefore demonstrate the behavior of the implemented multimodal fusion protocol, but they should not be described as performance measured on genuinely paired fingerprint-and-iris samples from the same physical individuals.
