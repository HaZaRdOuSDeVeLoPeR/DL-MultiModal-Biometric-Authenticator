"""
Central project configuration.

User-editable project settings live in this module. Runtime modules import
these values instead of maintaining independent copies of dataset, training,
protocol, or presentation settings.

For detailed documentation, see DOCUMENTATION.md in the project root.

Quick start:
- Edit values below to customize the project behavior
- Run `python3 main.py` for full pipeline or `python3 -m experiments --modality all`
- All settings are validated on import; invalid configurations raise ValueError
"""

from pathlib import Path
import os

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "datasets"
FINGERPRINT_DIR = DATASET_DIR / "fingerprints"
IRIS_DIR = DATASET_DIR / "iris"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
METRICS_DIR = RESULTS_DIR / "metrics"


# ============================================================
# Dataset configuration
# ============================================================

FINGERPRINT_NUM_SUBJECTS = 500
FINGERPRINT_NUM_FINGERS_PER_SUBJECT = 8
FINGERPRINT_IMPRESSIONS_PER_FINGER = 5
FINGERPRINT_TRAIN_SUBJECTS = 250
FINGERPRINT_EVAL_SUBJECTS = 250
FINGERPRINT_NUM_CLASSES = (
    FINGERPRINT_NUM_SUBJECTS * FINGERPRINT_NUM_FINGERS_PER_SUBJECT
)
FINGERPRINT_TRAIN_CLASSES = (
    FINGERPRINT_TRAIN_SUBJECTS * FINGERPRINT_NUM_FINGERS_PER_SUBJECT
)

IRIS_NUM_SUBJECTS = 1000
IRIS_EYES = ("L", "R")
IRIS_NUM_EYES_PER_SUBJECT = len(IRIS_EYES)
IRIS_IMPRESSIONS_PER_EYE = 10
IRIS_TRAIN_SUBJECTS = 500
IRIS_EVAL_SUBJECTS = 500
IRIS_NUM_CLASSES = IRIS_NUM_SUBJECTS * IRIS_NUM_EYES_PER_SUBJECT
IRIS_TRAIN_CLASSES = IRIS_TRAIN_SUBJECTS * IRIS_NUM_EYES_PER_SUBJECT

FINGERPRINT_EXTENSIONS = {".bmp"}
IRIS_EXTENSIONS = {".jpg", ".jpeg"}

# Source image dimensions, retained for documentation and validation.
FINGERPRINT_IMAGE_WIDTH = 328
FINGERPRINT_IMAGE_HEIGHT = 356
IRIS_IMAGE_WIDTH = 640
IRIS_IMAGE_HEIGHT = 480


# ============================================================
# Reproducibility and model selection
# ============================================================

RANDOM_SEED = 37
MODEL_NAMES = ("mobilenet_v2", "resnet18", "efficientnet_b0")
MODALITIES = ("fingerprint", "iris")


# ============================================================
# Model input and embedding configuration
# ============================================================

IMAGE_SIZE = 356
EMBEDDING_DIM = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# Authentication protocol configuration
# ============================================================

FINGERPRINT_ENROLL_IMPRESSIONS = 2
FINGERPRINT_VERIFY_IMPRESSIONS = (
    FINGERPRINT_IMPRESSIONS_PER_FINGER - FINGERPRINT_ENROLL_IMPRESSIONS
)
IRIS_ENROLL_IMPRESSIONS = 4
IRIS_VERIFY_IMPRESSIONS = (
    IRIS_IMPRESSIONS_PER_EYE - IRIS_ENROLL_IMPRESSIONS
)

# Fusion is synthetic side-by-side pairing. The headline experiment pairs a
# configurable number of independently held-out fingerprint and iris subjects
# by sorted-list position; the virtual population is not a genuine cohort.
VIRTUAL_PAIRING_LIMIT = min(FINGERPRINT_NUM_SUBJECTS, IRIS_NUM_SUBJECTS)
EXPERIMENT_PAIRS = 250
FUSION_FINGERPRINT_WEIGHT = 0.5
# Number of cross-modal verification query pairs processed per tensor batch.
FUSION_SCORE_BATCH_SIZE = 1024

# Authentication granularity switches. True keeps biometric instances
# (individual fingers/eyes) separate; False pools them to the subject level.
# Enrollment and verification can be controlled independently. The mixed
# combinations are supported for ablation experiments.
STANDALONE_VIRTUAL_ENROLLMENT = True
STANDALONE_VIRTUAL_VERIFICATION = True
FUSION_VIRTUAL_ENROLLMENT = False
FUSION_VIRTUAL_VERIFICATION = True


# ============================================================
# Training configuration
# ============================================================

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = os.cpu_count()
USE_AMP = True
SAVE_BEST_ONLY = True
TRAIN_DETERMINISTIC = True
ENABLE_TF32 = True
USE_CHANNELS_LAST = False
PREFETCH_FACTOR = 2
AMP_DTYPE = "float16"
USE_TORCH_COMPILE = False
SAVE_OPTIMIZER_STATE = True


# ============================================================
# Training and evaluation transforms
# ============================================================

# These are deliberately centralized because changing augmentation changes the
# learned representation and must affect both the production and benchmark
# dataset paths.
FINGERPRINT_ROTATION_DEGREES = 10
FINGERPRINT_AFFINE_DEGREES = 0
FINGERPRINT_AFFINE_TRANSLATE = (0.05, 0.05)
FINGERPRINT_AFFINE_SCALE = (0.95, 1.05)
IRIS_COLOR_JITTER_BRIGHTNESS = 0.1
IRIS_COLOR_JITTER_CONTRAST = 0.1
PAD_FILL = 0


# ============================================================
# Reusable analysis configuration
# ============================================================

FAR_TARGETS = (1e-2, 1e-3, 1e-4)
BOOTSTRAP_DEFAULT_ITERATIONS = 0
BOOTSTRAP_DEFAULT_CONFIDENCE = 0.95
BOOTSTRAP_DEFAULT_SEED = RANDOM_SEED
BOOTSTRAP_MAX_GENUINE_SAMPLES = 2_000
BOOTSTRAP_MAX_IMPOSTOR_SAMPLES = 100_000
PCA_MAX_POINTS = 5_000


# ============================================================
# Plot configuration
# ============================================================

PLOT_DPI = 160
SCORE_HISTOGRAM_BINS = 100
PLOT_SCORE_FIGSIZE = (8, 5)
PLOT_ROC_FIGSIZE = (7, 6)
PLOT_FAR_FRR_FIGSIZE = (8, 5)
PLOT_DET_FIGSIZE = (7, 6)
PLOT_PCA_FIGSIZE = (8, 6)
PLOT_TRAINING_FIGSIZE = (12, 4.5)
PLOT_METRIC_FIGSIZE = (8, 5)
PLOT_FUSION_DELTA_HEIGHT = 5
PLOT_FUSION_DELTA_MIN_WIDTH = 10
PLOT_SCATTER_SIZE = 8


# ============================================================
# Experiment CLI defaults
# ============================================================

EXPERIMENT_DEFAULT_MODALITY = "all"
EXPERIMENT_DEFAULT_BOOTSTRAP_ITERATIONS = BOOTSTRAP_DEFAULT_ITERATIONS
EXPERIMENT_DEFAULT_BOOTSTRAP_SEED = BOOTSTRAP_DEFAULT_SEED


# ============================================================
# Configuration invariants
# ============================================================

if FINGERPRINT_TRAIN_SUBJECTS + FINGERPRINT_EVAL_SUBJECTS != FINGERPRINT_NUM_SUBJECTS:
    raise ValueError("Fingerprint train/evaluation subjects must cover the dataset.")
if IRIS_TRAIN_SUBJECTS + IRIS_EVAL_SUBJECTS != IRIS_NUM_SUBJECTS:
    raise ValueError("Iris train/evaluation subjects must cover the dataset.")
if EXPERIMENT_PAIRS <= 0 or EXPERIMENT_PAIRS > VIRTUAL_PAIRING_LIMIT:
    raise ValueError(
        "EXPERIMENT_PAIRS must be positive and no greater than VIRTUAL_PAIRING_LIMIT."
    )

if not 0.0 <= FUSION_FINGERPRINT_WEIGHT <= 1.0:
    raise ValueError("FUSION_FINGERPRINT_WEIGHT must be between 0 and 1.")
if FUSION_SCORE_BATCH_SIZE <= 0:
    raise ValueError("FUSION_SCORE_BATCH_SIZE must be positive.")

for _name in (
    "STANDALONE_VIRTUAL_ENROLLMENT",
    "STANDALONE_VIRTUAL_VERIFICATION",
    "FUSION_VIRTUAL_ENROLLMENT",
    "FUSION_VIRTUAL_VERIFICATION",
):
    if not isinstance(globals()[_name], bool):
        raise ValueError(f"{_name} must be True or False.")
if FINGERPRINT_ENROLL_IMPRESSIONS <= 0 or FINGERPRINT_VERIFY_IMPRESSIONS <= 0:
    raise ValueError("Fingerprint enrollment and verification counts must be positive.")
if IRIS_ENROLL_IMPRESSIONS <= 0 or IRIS_VERIFY_IMPRESSIONS <= 0:
    raise ValueError("Iris enrollment and verification counts must be positive.")
