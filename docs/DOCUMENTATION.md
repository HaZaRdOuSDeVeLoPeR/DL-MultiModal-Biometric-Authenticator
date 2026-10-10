# Fingerprint + Iris Biometric Authentication — Technical Documentation

## 1. Overview

This project implements a deep-learning-based fingerprint and iris authentication system. It trains CNN backbones to produce biometric embeddings and evaluates authentication using cosine similarity.

Supported architectures:

- MobileNetV2
- ResNet18
- EfficientNet-B0

The project supports both standalone modality evaluation and synthetic fingerprint+iris fusion.

The important design principle is that **training identity semantics and authentication granularity are separate concerns**. The models are trained using subject+finger or subject+eye classes, while the post-training authentication protocol can retain those biometric instances or pool them to subject level.

---

## 2. End-to-end flow

```text
Dataset
   ↓
Deterministic subject split
   ↓
CNN training
   ↓
Embedding extraction
   ↓
Enrollment / verification split
   ↓
Template construction
   ↓
Cosine similarity
   ↓
Genuine + impostor scores
   ↓
AUC / EER / FAR / FRR / Accuracy / d-prime
```

Fusion adds:

```text
Fingerprint embeddings ─┐
                        ├── score-level / feature-level fusion
Iris embeddings ────────┘
```

---

## 3. Dataset and subject split

Fingerprint:

- 500 total subjects
- 250 training subjects
- 250 held-out evaluation subjects
- 8 fingers per subject
- 5 impressions per finger

Iris:

- 1,000 total subjects
- 500 training subjects
- 500 held-out evaluation subjects
- 2 eyes per subject
- 10 impressions per eye

Training and evaluation subjects are disjoint.

The fusion experiment cannot use true same-person multimodal pairs because the fingerprint and iris evaluation populations are independent. It therefore constructs deterministic synthetic virtual identities by positionally pairing held-out subjects.

---

## 4. Training identity semantics

### Fingerprint

Each classifier class represents:

```text
subject + finger
```

For example:

```text
004_L0
004_L1
...
004_R3
```

The eight fingers of a subject are therefore eight classifier classes.

### Iris

Each classifier class represents:

```text
subject + eye
```

For example:

```text
004_L
004_R
```

The two eyes are two classifier classes.

This is intentional: the CNN learns to distinguish impressions belonging to the same biometric source.

---

## 5. Enrollment and verification split

### Fingerprint

```text
5 impressions/finger

0, 1 → enrollment
2, 3, 4 → verification
```

Two enrollment embeddings are averaged to produce one template for each enrollment unit.

### Iris

```text
10 impressions/eye

0, 1, 2, 3 → enrollment
4, 5, 6, 7, 8, 9 → verification
```

Four enrollment embeddings are averaged to produce one template for each enrollment unit.

Enrollment and verification samples are disjoint.

---

## 6. Authentication granularity

Four central configuration switches control whether biometric instances are retained or pooled:

```python
STANDALONE_VIRTUAL_ENROLLMENT = True
STANDALONE_VIRTUAL_VERIFICATION = True

FUSION_VIRTUAL_ENROLLMENT = False
FUSION_VIRTUAL_VERIFICATION = False
```

The term "virtual" refers to the configurable authentication unit; it does not imply that different physical people share a biometric.

### `True`: instance level

Fingerprint:

```text
Subject
 ├── Finger 1 → template
 ├── Finger 2 → template
 ...
 └── Finger 8 → template
```

Iris:

```text
Subject
 ├── Left eye → template
 └── Right eye → template
```

### `False`: subject level

Fingerprint enrollment embeddings from all eight fingers are pooled into one subject template.

Iris enrollment embeddings from both eyes are pooled into one subject template.

For verification, each held-out impression is pooled across the subject's biometric instances, producing multiple subject-level verification templates.

---

## 7. Standalone protocol

### Instance-level standalone

Recommended setting:

```python
STANDALONE_VIRTUAL_ENROLLMENT = True
STANDALONE_VIRTUAL_VERIFICATION = True
```

Fingerprint:

- 2,000 enrollment templates for 250 evaluation subjects
- 6,000 verification queries

Iris:

- 500 enrollment templates
- 3,000 verification queries

Each verification query is compared against all enrollment templates. The matching biometric instance is genuine; other templates are impostors.

### Subject-level standalone

```python
STANDALONE_VIRTUAL_ENROLLMENT = False
STANDALONE_VIRTUAL_VERIFICATION = False
```

Fingerprint:

- 1 subject-level enrollment template/subject
- 3 subject-level verification templates/subject

Iris:

- 1 subject-level enrollment template/subject
- 6 subject-level verification templates/subject

This protocol measures the effect of multi-instance aggregation.

---

## 8. Synthetic fusion

### Virtual pairing

The held-out evaluation subjects are sorted independently and paired by position:

```text
virtual 0 = fingerprint_eval[0] + iris_eval[0]
virtual 1 = fingerprint_eval[1] + iris_eval[1]
...
```

With the default `EXPERIMENT_PAIRS = 250`, there are 250 virtual pairings.

These are **not real multimodal people**. The pairing is an experimental construction required because the source datasets are unpaired.

---

## 9. Fusion enrollment

### Instance-level fusion enrollment

```python
FUSION_VIRTUAL_ENROLLMENT = True
```

Fingerprint:

```text
8 fingers × 2 enrollment impressions
       ↓
8 fingerprint templates
```

Iris:

```text
2 eyes × 4 enrollment impressions
       ↓
2 iris templates
```

Cartesian product:

```text
8 × 2 = 16 fused enrollment templates / virtual subject
```

### Subject-level fusion enrollment

```python
FUSION_VIRTUAL_ENROLLMENT = False
```

Fingerprint:

```text
8 fingers × 2 enrollment impressions
       ↓
1 subject-level FP template
```

Iris:

```text
2 eyes × 4 enrollment impressions
       ↓
1 subject-level iris template
```

Cartesian product:

```text
1 × 1 = 1 fused enrollment template / virtual subject
```

---

## 10. Fusion verification

### Instance-level verification

```python
FUSION_VIRTUAL_VERIFICATION = True
```

Fingerprint:

```text
8 fingers × 3 verification samples = 24 queries
```

Iris:

```text
2 eyes × 6 verification samples = 12 queries
```

Cartesian product:

```text
24 × 12 = 288 multimodal query pairs / virtual subject
```

At 250 virtual subjects:

```text
250 × 288 = 72,000 query pairs
```

### Subject-level verification

```python
FUSION_VIRTUAL_VERIFICATION = False
```

Fingerprint:

```text
3 subject-level verification templates
```

Iris:

```text
6 subject-level verification templates
```

Cartesian product:

```text
3 × 6 = 18 multimodal query pairs / virtual subject
```

At 250 virtual subjects:

```text
250 × 18 = 4,500 query pairs
```

---

## 11. Fusion scoring

Instance-level fusion uses vectorized, chunked tensor scoring controlled by `FUSION_SCORE_BATCH_SIZE` (default `1024`) to avoid expensive Python-level Cartesian loops. CUDA is used automatically for the tensor-scoring stage when available; ROC/EER metric calculation remains CPU/NumPy based.

### Score-level fusion

For a fingerprint score \(S_{fp}\) and iris score \(S_{iris}\):

\[
S_{fusion} =
w_{fp}S_{fp} +
(1-w_{fp})S_{iris}
\]

where:

```python
FUSION_FINGERPRINT_WEIGHT = 0.5
```

by default.

### Feature-level fusion

The fingerprint and iris embeddings are scaled using the square roots of their fusion weights, concatenated, normalized, and compared using cosine similarity.

This provides a feature-space fusion alternative to score-level fusion.

---

## 12. Genuine and impostor trials

For instance-level fusion, a fused enrollment template contains one fingerprint biometric instance and one iris biometric instance.

A verification pair is genuine when **both modality components correspond to the target authentication unit**.

All other fused enrollment templates produce impostor scores.

For subject-level fusion, the subject-level fingerprint and iris templates are the genuine target.

The implementation supports mixed enrollment/verification granularity as an ablation protocol.

---

## 13. Metrics

The experiment reports:

- AUC
- EER
- EER threshold
- FAR at EER
- FRR at EER
- Accuracy at EER threshold
- Genuine mean, standard deviation, median
- Impostor mean, standard deviation, median
- d-prime
- Genuine trial count
- Impostor trial count
- TAR at configured FAR targets

### EER

The Equal Error Rate is the operating point where FAR and FRR are approximately equal.

### Accuracy

Accuracy is computed at the EER threshold:

\[
Accuracy =
rac{TP+TN}{TP+TN+FP+FN}
\]

A similarity score at or above the threshold is accepted.

**Important:** biometric datasets commonly contain far more impostor comparisons than genuine comparisons. Consequently, accuracy can look very high even when the operating characteristics are not excellent. EER, FAR, FRR and AUC should therefore remain the primary biometric metrics.

---

## 14. Plots

Standalone experiments generate:

- score distributions
- ROC curves
- DET curves
- FAR/FRR curves
- PCA projections
- EER comparison
- **accuracy comparison**

Fusion generates:

- ROC curves for score-level and feature-level fusion for every fingerprint × iris architecture pair
- FAR/FRR curves for every fusion method and architecture pair
- PCA projections of the fused feature-level enrollment representations for every architecture pair
- score distributions and DET curves for fusion methods

The fusion accuracy comparison is saved at `results/experiments/fusion/plots/comparisons/fusion_accuracy_comparison.png` and includes all fingerprint-model × iris-model combinations, with separate score-level and feature-level bars.

- DET curves
- score distributions for fusion methods
- EER comparison
- **accuracy comparison**
- fusion EER improvement
- fusion AUC improvement

---

## 15. Interpreting standalone vs fusion baselines

A single-finger standalone fingerprint experiment and a subject-level fusion fingerprint baseline are different systems.

For example:

```text
Single-finger:
1 finger → finger template

Subject-level:
8 fingers → averaged subject template
```

The second system can be substantially more accurate because information from multiple fingers has already been combined.

Therefore, a subject-level fingerprint baseline should not be presented as directly equivalent to a single-finger standalone baseline.

A useful analysis is:

```text
single-finger FP
        ↓
multi-finger FP
        ↓
multi-finger FP + iris
```

This separates the benefit of multi-instance aggregation from the benefit of multimodal fusion.

---

## 16. Reproducibility and configuration

Important configuration values include:

```python
RANDOM_SEED = 37

EXPERIMENT_PAIRS = 250
VIRTUAL_PAIRING_LIMIT = 500

FUSION_FINGERPRINT_WEIGHT = 0.5

STANDALONE_VIRTUAL_ENROLLMENT = True
STANDALONE_VIRTUAL_VERIFICATION = True

FUSION_VIRTUAL_ENROLLMENT = False
FUSION_VIRTUAL_VERIFICATION = False
```

All four granularity switches are validated as booleans.

---

## 17. Experiment outputs

Each experiment writes JSON and CSV metric summaries plus score arrays and plots.

Fusion additionally records the virtual pairing map and the selected protocol configuration in `pairing.json` and `manifest.json`.

This makes each run traceable to its authentication protocol.

---

## 18. Limitations

### Synthetic multimodal identities

Fingerprint and iris subjects are not paired in the source datasets. Fusion therefore uses synthetic virtual identities.

The results should be described as:

> performance of the implemented synthetic multimodal fusion protocol

rather than:

> performance on genuine paired fingerprint-and-iris subjects.

### Correlated trials

Cartesian products reuse the same underlying verification queries across multiple trials. Therefore, the resulting scores are not all statistically independent.

This does not prevent ROC/EER computation, but trial counts should not be described as independent observations.

### Accuracy interpretation

Accuracy is threshold-dependent and affected by the genuine/impostor class balance. It should not replace EER, FAR, FRR, or AUC in biometric evaluation.

---

## 19. Testing

Run:

```bash
python3 -m unittest discover -s tests -v
```

The test suite covers:

- deterministic virtual subject pairing
- invalid pairing configurations
- Cartesian trial construction
- subject-level verification aggregation
- metric accuracy reporting
- authentication granularity configuration


## 20. Local authentication UI

`ui/app.py` provides a local Flask/Jinja presentation layer over the trained embedding models. It loads the same checkpoints from `results/models`, applies the deterministic evaluation preprocessing, averages multiple uploaded enrollment embeddings into one subject template, and compares verification embeddings against all enrolled templates. A claimed identity can be checked at the model's saved EER threshold; without a claim the interface performs open-set identification by selecting the highest similarity score.

The UI stores enrollment templates only in process memory. It is intended for local project demonstration, not production biometric deployment.
