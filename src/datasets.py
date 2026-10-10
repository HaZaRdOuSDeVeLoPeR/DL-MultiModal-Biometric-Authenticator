import json
import random
from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing import (
    get_fingerprint_train_transform,
    get_iris_train_transform,
    get_eval_transform,
)

from config import (
    FINGERPRINT_DIR,
    IRIS_DIR,

    FINGERPRINT_NUM_SUBJECTS,
    FINGERPRINT_NUM_FINGERS_PER_SUBJECT,
    FINGERPRINT_IMPRESSIONS_PER_FINGER,
    FINGERPRINT_TRAIN_SUBJECTS,
    FINGERPRINT_EVAL_SUBJECTS,

    IRIS_NUM_SUBJECTS,
    IRIS_EYES,
    IRIS_IMPRESSIONS_PER_EYE,
    IRIS_TRAIN_SUBJECTS,
    IRIS_EVAL_SUBJECTS,

    FINGERPRINT_EXTENSIONS,
    IRIS_EXTENSIONS,

    RANDOM_SEED,
    RESULTS_DIR
)


# ============================================================
# Types
# ============================================================

SplitName = Literal["train", "eval"]


def make_sample_key(class_key: str, impression: int) -> str:
    """Build a traceability key without changing the classifier label.

    ``class_key`` identifies the biometric class (subject + finger for
    fingerprints, or subject + eye for iris).  ``impression`` identifies one
    sample within that class and is deliberately not part of ``class_id``.
    """
    return f"{class_key}_{int(impression):02d}"


# ============================================================
# Split manifest
# ============================================================

SPLIT_FILE = RESULTS_DIR / "subject_split.json"


def validate_split_configuration() -> None:
    """
    Validate that the configured train/evaluation splits
    account for every subject in each modality.
    """

    fingerprint_total = (FINGERPRINT_TRAIN_SUBJECTS + FINGERPRINT_EVAL_SUBJECTS)

    if fingerprint_total != FINGERPRINT_NUM_SUBJECTS:
        raise ValueError(
            "Fingerprint subject split configuration is invalid: "
            f"{fingerprint_total} subjects allocated, "
            f"but {FINGERPRINT_NUM_SUBJECTS} are available."
        )

    iris_total = (IRIS_TRAIN_SUBJECTS + IRIS_EVAL_SUBJECTS)

    if iris_total != IRIS_NUM_SUBJECTS:
        raise ValueError(
            "Iris subject split configuration is invalid: "
            f"{iris_total} subjects allocated, "
            f"but {IRIS_NUM_SUBJECTS} are available."
        )


def generate_subject_split(seed: int = RANDOM_SEED) -> dict:
    """
    Generate a reproducible subject-wise train/evaluation split.

    The same subject can never appear in both splits within
    a modality.

    All biometric samples belonging to a training subject
    remain available for training.
    """

    validate_split_configuration()

    rng = random.Random(seed)

    # --------------------------------------------------------
    # Fingerprint subjects
    # --------------------------------------------------------

    fingerprint_subjects = list(
        range(FINGERPRINT_NUM_SUBJECTS)
    )

    rng.shuffle(fingerprint_subjects)

    fingerprint_train = sorted(
        fingerprint_subjects[:FINGERPRINT_TRAIN_SUBJECTS]
    )

    fingerprint_eval = sorted(
        fingerprint_subjects[FINGERPRINT_TRAIN_SUBJECTS:]
    )

    # --------------------------------------------------------
    # Iris subjects
    # --------------------------------------------------------

    iris_subjects = list(
        range(IRIS_NUM_SUBJECTS)
    )

    rng.shuffle(iris_subjects)

    iris_train = sorted(
        iris_subjects[:IRIS_TRAIN_SUBJECTS]
    )

    iris_eval = sorted(
        iris_subjects[IRIS_TRAIN_SUBJECTS:]
    )

    return {
        "seed": seed,

        "fingerprint": {
            "train_subjects": fingerprint_train,
            "eval_subjects": fingerprint_eval
        },

        "iris": {
            "train_subjects": iris_train,
            "eval_subjects": iris_eval
        },
    }


def save_subject_split(manifest: dict) -> None:
    """
    Save the subject-wise split manifest to disk.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SPLIT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )


def load_subject_split() -> dict | None:
    """
    Load an existing subject split if available.
    """

    if not SPLIT_FILE.exists():
        return None

    with SPLIT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_subject_split(manifest: dict) -> None:
    """
    Validate the subject-wise split manifest.

    Ensures:
        - correct number of subjects
        - no train/evaluation overlap
        - complete subject coverage
    """

    # --------------------------------------------------------
    # Fingerprint
    # --------------------------------------------------------

    fingerprint_train = set(manifest["fingerprint"]["train_subjects"])
    fingerprint_eval = set(manifest["fingerprint"]["eval_subjects"])

    if fingerprint_train & fingerprint_eval:
        raise ValueError(
            "Fingerprint train/evaluation subject overlap detected."
        )

    fingerprint_all = (fingerprint_train | fingerprint_eval)

    expected_fingerprint = set(
        range(FINGERPRINT_NUM_SUBJECTS)
    )

    if fingerprint_all != expected_fingerprint:
        raise ValueError(
            "Fingerprint subject split does not "
            "cover all subjects."
        )

    if len(fingerprint_train) != FINGERPRINT_TRAIN_SUBJECTS:
        raise ValueError(
            "Fingerprint training subject count is incorrect."
        )

    if len(fingerprint_eval) != FINGERPRINT_EVAL_SUBJECTS:
        raise ValueError(
            "Fingerprint evaluation subject count is incorrect."
        )

    # --------------------------------------------------------
    # Iris
    # --------------------------------------------------------

    iris_train = set(manifest["iris"]["train_subjects"])
    iris_eval = set(manifest["iris"]["eval_subjects"])

    if iris_train & iris_eval:
        raise ValueError(
            "Iris train/evaluation subject overlap detected."
        )

    iris_all = (iris_train | iris_eval)

    expected_iris = set(
        range(IRIS_NUM_SUBJECTS)
    )

    if iris_all != expected_iris:
        raise ValueError(
            "Iris subject split does not "
            "cover all subjects."
        )

    if len(iris_train) != IRIS_TRAIN_SUBJECTS:
        raise ValueError(
            "Iris training subject count is incorrect."
        )

    if len(iris_eval) != IRIS_EVAL_SUBJECTS:
        raise ValueError(
            "Iris evaluation subject count is incorrect."
        )


def get_subject_split(seed: int = RANDOM_SEED) -> dict:
    """
    Load the existing subject split if it matches the current
    configuration. Otherwise generate a new one.
    """

    manifest = load_subject_split()

    if manifest is not None:

        stored_seed = manifest.get("seed")

        if stored_seed == seed:

            try:
                validate_subject_split(manifest)
                return manifest

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                pass

    manifest = generate_subject_split(seed)

    validate_subject_split(manifest)
    save_subject_split(manifest)

    return manifest


# ============================================================
# Fingerprint dataset
# ============================================================

class FingerprintDataset(Dataset):
    """
    Dataset for fingerprint biometric classes.

    A class is defined by:

        subject + finger

    Therefore each subject contributes
    FINGERPRINT_NUM_FINGERS_PER_SUBJECT classes.

    All impressions belonging to a training subject are used.
    Evaluation subjects are completely disjoint from training.
    """

    def __init__(self,split: SplitName,manifest: dict):
        self.split = split
        self.manifest = manifest

        self.transform = (
            get_fingerprint_train_transform()
            if split == "train"
            else get_eval_transform()
        )

        subjects = manifest["fingerprint"][f"{split}_subjects"]

        self.samples: list[tuple[int, int, int, Path]] = []
        self.class_map: dict[tuple[int, int], int] = {}

        class_id = 0

        for subject in sorted(subjects):

            subject_key = f"{subject:03d}"

            for finger_index in range(FINGERPRINT_NUM_FINGERS_PER_SUBJECT):

                if self.split == "train":
                    self.class_map[(subject, finger_index)] = class_id
                    class_id += 1

                finger = (
                    f"L{finger_index}"
                    if finger_index < 4
                    else f"R{finger_index - 4}"
                )

                finger_dir = FINGERPRINT_DIR / subject_key / finger

                for impression in range(
                    FINGERPRINT_IMPRESSIONS_PER_FINGER
                ):

                    path = finger_dir / f"{subject_key}_{finger}_{impression}.bmp"

                    if path.suffix.lower() not in FINGERPRINT_EXTENSIONS:
                        raise ValueError(
                            f"Unexpected fingerprint extension: {path}"
                        )

                    if not path.exists():
                        raise FileNotFoundError(
                            f"Fingerprint image not found: {path}"
                        )

                    self.samples.append(
                        (subject, finger_index, impression, path)
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:

        (subject, finger_index, impression, path) = self.samples[index]

        with Image.open(path) as image:
            image = image.copy()

        image = self.transform(image)

        finger = (
            f"L{finger_index}"
            if finger_index < 4
            else f"R{finger_index - 4}"
        )

        class_key = f"{subject:03d}_{finger}"
        result = {
            "image": image,
            "subject": subject,
            "finger": finger,
            "finger_index": finger_index,
            # Training identity: subject + finger.  Impression is sample
            # metadata and must not create a separate classifier class.
            "class_key": class_key,
            "sample_key": make_sample_key(class_key, impression),
            "impression": impression,
            "path": str(path),
        }

        if self.split == "train":
            result["class_id"] = self.class_map[(subject, finger_index)]

        return result


# ============================================================
# Iris dataset
# ============================================================

class IrisDataset(Dataset):
    """
    Dataset for iris biometric classes.

    A class is defined by:

        subject + eye

    Therefore every subject contributes two biometric classes:

        subject + L
        subject + R

    All impressions belonging to a training subject are used.
    Evaluation subjects are completely disjoint from training.
    """

    def __init__(self, split: SplitName, manifest: dict):
        self.split = split
        self.manifest = manifest

        self.transform = (
            get_iris_train_transform()
            if split == "train"
            else get_eval_transform()
        )

        subjects = manifest["iris"][f"{split}_subjects"]

        self.samples: list[tuple[int, str, int, Path]] = []
        self.class_map: dict[tuple[int, str], int] = {}

        class_id = 0

        for subject in sorted(subjects):

            subject_key = f"{subject:03d}"

            for eye in IRIS_EYES:

                if self.split == "train":
                    self.class_map[(subject, eye)] = class_id
                    class_id += 1

                eye_dir = (IRIS_DIR / subject_key / eye)

                for impression in range(IRIS_IMPRESSIONS_PER_EYE):

                    path = eye_dir / f"{subject_key}_{eye}_{impression:02d}.jpg"

                    if path.suffix.lower() not in IRIS_EXTENSIONS:
                        raise ValueError(
                            f"Unexpected iris extension: {path}"
                        )

                    if not path.exists():
                        raise FileNotFoundError(
                            f"Iris image not found: {path}"
                        )

                    self.samples.append(
                        (subject, eye, impression, path)
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:

        (subject, eye, impression, path) = self.samples[index]

        with Image.open(path) as image:
            image = image.copy()

        image = self.transform(image)

        class_key = f"{subject:03d}_{eye}"
        result = {
            "image": image,
            "subject": subject,
            "eye": eye,
            # Training identity: subject + eye.  Impression is sample
            # metadata and must not create a separate classifier class.
            "class_key": class_key,
            "sample_key": make_sample_key(class_key, impression),
            "impression": impression,
            "path": str(path)
        }

        if self.split == "train":
            result["class_id"] = self.class_map[(subject, eye)]

        return result


# ============================================================
# Dataset preparation
# ============================================================

def prepare_datasets(seed: int = RANDOM_SEED) -> dict:
    """
    Prepare subject-disjoint training and evaluation datasets.

    Returns:
        fingerprint_train
        fingerprint_eval
        iris_train
        iris_eval
    """

    manifest = get_subject_split(seed)
    validate_subject_split(manifest)

    datasets = {
        "fingerprint_train": FingerprintDataset(split="train", manifest=manifest),
        "fingerprint_eval": FingerprintDataset(split="eval", manifest=manifest),
        "iris_train": IrisDataset(split="train", manifest=manifest),
        "iris_eval": IrisDataset(split="eval", manifest=manifest)
    }

    return datasets