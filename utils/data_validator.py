from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import (
    FINGERPRINT_DIR,
    IRIS_DIR,

    FINGERPRINT_NUM_SUBJECTS,
    FINGERPRINT_NUM_CLASSES,
    FINGERPRINT_NUM_FINGERS_PER_SUBJECT,
    FINGERPRINT_IMPRESSIONS_PER_FINGER,
    FINGERPRINT_TRAIN_SUBJECTS,
    FINGERPRINT_EVAL_SUBJECTS,
    FINGERPRINT_EXTENSIONS,
    FINGERPRINT_IMAGE_WIDTH,
    FINGERPRINT_IMAGE_HEIGHT,

    IRIS_NUM_SUBJECTS,
    IRIS_NUM_CLASSES,
    IRIS_EYES,
    IRIS_IMPRESSIONS_PER_EYE,
    IRIS_TRAIN_SUBJECTS,
    IRIS_EVAL_SUBJECTS,
    IRIS_EXTENSIONS,
    IRIS_IMAGE_WIDTH,
    IRIS_IMAGE_HEIGHT,
)


# ============================================================
# Validation result
# ============================================================

@dataclass
class ValidationResult:
    name: str
    passed: bool
    errors: list[str]
    warnings: list[str]

    def print_report(self) -> None:
        status = "PASSED" if self.passed else "FAILED"

        print("\n")
        print("=" * 70)
        print(f"{self.name}: {status}")
        print("=" * 70)

        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  [ERROR] {error}")

        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  [WARNING] {warning}")

        if not self.errors and not self.warnings:
            print("  No problems detected.")


# ============================================================
# Generic helpers
# ============================================================

def _numeric_ids(directory: Path) -> set[int]:
    """
    Return numeric directory names as integers.
    """

    ids: set[int] = set()

    if not directory.exists():
        return ids

    for item in directory.iterdir():
        if item.is_dir() and item.name.isdigit():
            ids.add(int(item.name))

    return ids


def _image_files(
    directory: Path,
    extensions: set[str],
) -> list[Path]:
    """
    Return image files directly inside a directory.
    """

    if not directory.exists():
        return []

    return sorted(
        file
        for file in directory.iterdir()
        if (
            file.is_file()
            and file.suffix.lower() in extensions
        )
    )


def _validate_image_file(file: Path) -> str | None:
    """
    Verify that an image can be opened and decoded.

    Returns:
        None if valid.
        Error message otherwise.
    """

    try:
        with Image.open(file) as image:
            image.verify()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        return f"{file}: {error}"

    return None


def _get_image_properties(
    file: Path,
) -> tuple[tuple[int, int], str, str]:
    """
    Return image dimensions, mode and format.
    """

    with Image.open(file) as image:

        if image.format is None:
            raise ValueError(
                f"Could not determine image format: {file}"
            )

        return (
            image.size,
            image.mode,
            image.format,
        )


# ============================================================
# Fingerprint validation
# ============================================================

def validate_fingerprints() -> ValidationResult:
    """
    Validate the complete CASIA-FingerprintV5 dataset.

    Expected structure:

        fingerprints/
        ├── 000/
        │   ├── L0/
        │   │   ├── 000_L0_0.bmp
        │   │   ├── ...
        │   │   └── 000_L0_4.bmp
        │   ├── L1/
        │   ├── L2/
        │   ├── L3/
        │   ├── R0/
        │   ├── R1/
        │   ├── R2/
        │   └── R3/
        ├── 001/
        └── ...

    Each subject must contain:

        8 fingers × 5 impressions = 40 images.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not FINGERPRINT_DIR.exists():
        return ValidationResult(
            name="Fingerprint Dataset",
            passed=False,
            errors=[
                f"Directory does not exist: "
                f"{FINGERPRINT_DIR}"
            ],
            warnings=[],
        )

    subject_ids = _numeric_ids(FINGERPRINT_DIR)

    expected_subject_ids = set(
        range(FINGERPRINT_NUM_SUBJECTS)
    )

    missing_subjects = (
        expected_subject_ids - subject_ids
    )

    extra_subjects = (
        subject_ids - expected_subject_ids
    )

    if missing_subjects:
        errors.append(
            "Missing fingerprint subjects: "
            f"{sorted(missing_subjects)}"
        )

    if extra_subjects:
        warnings.append(
            "Unexpected fingerprint subjects: "
            f"{sorted(extra_subjects)}"
        )

    finger_names = (
        [
            f"L{i}"
            for i in range(4)
        ]
        +
        [
            f"R{i}"
            for i in range(4)
        ]
    )

    total_images = 0

    for subject_id in sorted(expected_subject_ids):

        subject_dir = (
            FINGERPRINT_DIR
            / f"{subject_id:03d}"
        )

        if not subject_dir.exists():
            continue

        # ----------------------------------------------------
        # Check finger directories
        # ----------------------------------------------------

        actual_fingers = {
            item.name
            for item in subject_dir.iterdir()
            if item.is_dir()
        }

        expected_fingers = set(finger_names)

        missing_fingers = (
            expected_fingers - actual_fingers
        )

        extra_fingers = (
            actual_fingers - expected_fingers
        )

        if missing_fingers:
            errors.append(
                f"Fingerprint subject "
                f"{subject_id:03d}: "
                f"missing finger directories "
                f"{sorted(missing_fingers)}"
            )

        if extra_fingers:
            warnings.append(
                f"Fingerprint subject "
                f"{subject_id:03d}: "
                f"unexpected directories "
                f"{sorted(extra_fingers)}"
            )

        # ----------------------------------------------------
        # Check each finger
        # ----------------------------------------------------

        for finger in finger_names:

            finger_dir = (
                subject_dir / finger
            )

            if not finger_dir.exists():
                continue

            files = _image_files(
                finger_dir,
                FINGERPRINT_EXTENSIONS,
            )

            total_images += len(files)

            # ------------------------------------------------
            # Check impression count
            # ------------------------------------------------

            if (
                len(files)
                != FINGERPRINT_IMPRESSIONS_PER_FINGER
            ):
                errors.append(
                    f"Fingerprint "
                    f"{subject_id:03d}/{finger}: "
                    f"expected "
                    f"{FINGERPRINT_IMPRESSIONS_PER_FINGER} "
                    f"images, found {len(files)}"
                )

            # ------------------------------------------------
            # Check filenames
            # ------------------------------------------------

            expected_names = {
                (
                    f"{subject_id:03d}_"
                    f"{finger}_"
                    f"{impression}.bmp"
                )
                for impression in range(
                    FINGERPRINT_IMPRESSIONS_PER_FINGER
                )
            }

            actual_names = {
                file.name
                for file in files
            }

            missing_files = (
                expected_names - actual_names
            )

            if missing_files:
                errors.append(
                    f"Fingerprint "
                    f"{subject_id:03d}/{finger}: "
                    f"missing files "
                    f"{sorted(missing_files)}"
                )

            unexpected_files = (
                actual_names - expected_names
            )

            if unexpected_files:
                warnings.append(
                    f"Fingerprint "
                    f"{subject_id:03d}/{finger}: "
                    f"unexpected files "
                    f"{sorted(unexpected_files)}"
                )

    expected_total = (
        FINGERPRINT_NUM_SUBJECTS
        * FINGERPRINT_NUM_FINGERS_PER_SUBJECT
        * FINGERPRINT_IMPRESSIONS_PER_FINGER
    )

    print("\n")
    print("=" * 70)
    print("FINGERPRINT DATASET")
    print("=" * 70)

    print(
        f"Subjects found       : "
        f"{len(subject_ids)}"
    )

    print(
        f"Total BMP images     : "
        f"{total_images}"
    )

    print(
        f"Expected images      : "
        f"{expected_total}"
    )

    print(
        f"Images / subject     : "
        f"{FINGERPRINT_NUM_FINGERS_PER_SUBJECT * FINGERPRINT_IMPRESSIONS_PER_FINGER}"
    )

    print(
        f"Fingers / subject    : "
        f"{FINGERPRINT_NUM_FINGERS_PER_SUBJECT}"
    )

    print(
        f"Impressions / finger : "
        f"{FINGERPRINT_IMPRESSIONS_PER_FINGER}"
    )

    if total_images != expected_total:
        errors.append(
            "Fingerprint image count mismatch: "
            f"expected {expected_total}, "
            f"found {total_images}"
        )

    return ValidationResult(
        name="Fingerprint Dataset",
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ============================================================
# Iris validation
# ============================================================

def validate_iris() -> ValidationResult:
    """
    Validate the complete CASIA-Iris-Thousand dataset.

    Expected structure:

        iris/
        ├── 000/
        │   ├── L/
        │   │   ├── 000_L_00.jpg
        │   │   └── ...
        │   └── R/
        │       ├── 000_R_00.jpg
        │       └── ...
        ├── 001/
        └── ...

    Each subject must contain:

        2 eyes × 10 impressions = 20 images.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not IRIS_DIR.exists():
        return ValidationResult(
            name="Iris Dataset",
            passed=False,
            errors=[
                f"Directory does not exist: "
                f"{IRIS_DIR}"
            ],
            warnings=[],
        )

    subject_ids = _numeric_ids(IRIS_DIR)

    expected_subject_ids = set(
        range(IRIS_NUM_SUBJECTS)
    )

    missing_subjects = (
        expected_subject_ids - subject_ids
    )

    extra_subjects = (
        subject_ids - expected_subject_ids
    )

    if missing_subjects:
        errors.append(
            "Missing iris subjects: "
            f"{sorted(missing_subjects)}"
        )

    if extra_subjects:
        warnings.append(
            "Unexpected iris subjects: "
            f"{sorted(extra_subjects)}"
        )

    total_images = 0

    for subject_id in sorted(expected_subject_ids):

        subject_dir = (
            IRIS_DIR
            / f"{subject_id:03d}"
        )

        if not subject_dir.exists():
            continue

        # ----------------------------------------------------
        # Check eye directories
        # ----------------------------------------------------

        actual_eyes = {
            item.name
            for item in subject_dir.iterdir()
            if item.is_dir()
        }

        expected_eyes = set(IRIS_EYES)

        missing_eyes = (
            expected_eyes - actual_eyes
        )

        extra_eyes = (
            actual_eyes - expected_eyes
        )

        if missing_eyes:
            errors.append(
                f"Iris subject "
                f"{subject_id:03d}: "
                f"missing eye directories "
                f"{sorted(missing_eyes)}"
            )

        if extra_eyes:
            warnings.append(
                f"Iris subject "
                f"{subject_id:03d}: "
                f"unexpected directories "
                f"{sorted(extra_eyes)}"
            )

        # ----------------------------------------------------
        # Check each eye
        # ----------------------------------------------------

        for eye in IRIS_EYES:

            eye_dir = (
                subject_dir / eye
            )

            if not eye_dir.exists():
                continue

            files = _image_files(
                eye_dir,
                IRIS_EXTENSIONS,
            )

            total_images += len(files)

            # ------------------------------------------------
            # Check impression count
            # ------------------------------------------------

            if (
                len(files)
                != IRIS_IMPRESSIONS_PER_EYE
            ):
                errors.append(
                    f"Iris "
                    f"{subject_id:03d}/{eye}: "
                    f"expected "
                    f"{IRIS_IMPRESSIONS_PER_EYE} "
                    f"images, found {len(files)}"
                )

            # ------------------------------------------------
            # Check filenames
            # ------------------------------------------------

            expected_names = {
                (
                    f"{subject_id:03d}_"
                    f"{eye}_"
                    f"{impression:02d}.jpg"
                )
                for impression in range(
                    IRIS_IMPRESSIONS_PER_EYE
                )
            }

            actual_names = {
                file.name
                for file in files
            }

            missing_files = (
                expected_names - actual_names
            )

            if missing_files:
                errors.append(
                    f"Iris "
                    f"{subject_id:03d}/{eye}: "
                    f"missing files "
                    f"{sorted(missing_files)}"
                )

            unexpected_files = (
                actual_names - expected_names
            )

            if unexpected_files:
                warnings.append(
                    f"Iris "
                    f"{subject_id:03d}/{eye}: "
                    f"unexpected files "
                    f"{sorted(unexpected_files)}"
                )

    expected_total = (
        IRIS_NUM_SUBJECTS
        * len(IRIS_EYES)
        * IRIS_IMPRESSIONS_PER_EYE
    )

    print("\n")
    print("=" * 70)
    print("IRIS DATASET")
    print("=" * 70)

    print(
        f"Subjects found       : "
        f"{len(subject_ids)}"
    )

    print(
        f"Total JPEG images    : "
        f"{total_images}"
    )

    print(
        f"Expected images      : "
        f"{expected_total}"
    )

    print(
        f"Images / subject     : "
        f"{len(IRIS_EYES) * IRIS_IMPRESSIONS_PER_EYE}"
    )

    print(
        f"Eyes / subject       : "
        f"{len(IRIS_EYES)}"
    )

    print(
        f"Impressions / eye    : "
        f"{IRIS_IMPRESSIONS_PER_EYE}"
    )

    if total_images != expected_total:
        errors.append(
            "Iris image count mismatch: "
            f"expected {expected_total}, "
            f"found {total_images}"
        )

    return ValidationResult(
        name="Iris Dataset",
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ============================================================
# Image integrity
# ============================================================

def validate_image_integrity() -> ValidationResult:
    """
    Verify that every biometric image can be opened.
    """

    errors: list[str] = []
    warnings: list[str] = []

    print("\n")
    print("=" * 70)
    print("IMAGE INTEGRITY CHECK")
    print("=" * 70)

    all_files: list[Path] = []

    # --------------------------------------------------------
    # Fingerprints
    # --------------------------------------------------------

    for subject_id in range(
        FINGERPRINT_NUM_SUBJECTS
    ):

        subject_dir = (
            FINGERPRINT_DIR
            / f"{subject_id:03d}"
        )

        if not subject_dir.exists():
            continue

        for finger_dir in subject_dir.iterdir():

            if not finger_dir.is_dir():
                continue

            all_files.extend(
                _image_files(
                    finger_dir,
                    FINGERPRINT_EXTENSIONS,
                )
            )

    # --------------------------------------------------------
    # Iris
    # --------------------------------------------------------

    for subject_id in range(
        IRIS_NUM_SUBJECTS
    ):

        subject_dir = (
            IRIS_DIR
            / f"{subject_id:03d}"
        )

        if not subject_dir.exists():
            continue

        for eye in IRIS_EYES:

            eye_dir = (
                subject_dir / eye
            )

            if not eye_dir.exists():
                continue

            all_files.extend(
                _image_files(
                    eye_dir,
                    IRIS_EXTENSIONS,
                )
            )

    total_files = len(all_files)
    valid_files = 0

    for index, file in enumerate(
        all_files,
        start=1,
    ):

        error = _validate_image_file(file)

        if error is not None:
            errors.append(error)

        else:
            valid_files += 1

        if (
            index % 500 == 0
            or index == total_files
        ):
            print(
                f"\rChecked "
                f"{index}/{total_files} images",
                end="",
                flush=True,
            )

    print()

    print(
        f"Valid images   : "
        f"{valid_files}"
    )

    print(
        f"Invalid images : "
        f"{len(errors)}"
    )

    return ValidationResult(
        name="Image Integrity",
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ============================================================
# Image property inspection
# ============================================================

def inspect_image_properties(
    samples_per_dataset: int = 3,
) -> None:
    """
    Inspect representative images from both datasets.
    """

    print("\n")
    print("=" * 70)
    print("IMAGE PROPERTY INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Fingerprints
    # --------------------------------------------------------

    print("\nFingerprint samples:")
    print("-" * 70)

    fingerprint_samples: list[Path] = []

    for subject_id in range(
        FINGERPRINT_NUM_SUBJECTS
    ):

        subject_dir = (
            FINGERPRINT_DIR
            / f"{subject_id:03d}"
        )

        if not subject_dir.exists():
            continue

        for finger_dir in sorted(
            subject_dir.iterdir()
        ):

            if not finger_dir.is_dir():
                continue

            fingerprint_samples.extend(
                _image_files(
                    finger_dir,
                    FINGERPRINT_EXTENSIONS,
                )
            )

            if (
                len(fingerprint_samples)
                >= samples_per_dataset
            ):
                break

        if (
            len(fingerprint_samples)
            >= samples_per_dataset
        ):
            break

    for file in fingerprint_samples[
        :samples_per_dataset
    ]:

        try:
            size, mode, image_format = (
                _get_image_properties(file)
            )

            print(
                f"  {file.name:<25} "
                f"size={size[0]}x{size[1]}, "
                f"mode={mode}, "
                f"format={image_format}"
            )

            expected_size = (
                FINGERPRINT_IMAGE_WIDTH,
                FINGERPRINT_IMAGE_HEIGHT,
            )

            if size != expected_size:
                print(
                    "    [WARNING] Unexpected "
                    f"dimensions. Expected "
                    f"{expected_size[0]}x"
                    f"{expected_size[1]}"
                )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:

            print(
                f"  [ERROR] {file}: {error}"
            )

    # --------------------------------------------------------
    # Iris
    # --------------------------------------------------------

    print("\nIris samples:")
    print("-" * 70)

    iris_samples: list[Path] = []

    for subject_id in range(
        IRIS_NUM_SUBJECTS
    ):

        subject_dir = (
            IRIS_DIR
            / f"{subject_id:03d}"
        )

        if not subject_dir.exists():
            continue

        for eye in IRIS_EYES:

            eye_dir = (
                subject_dir / eye
            )

            if not eye_dir.exists():
                continue

            iris_samples.extend(
                _image_files(
                    eye_dir,
                    IRIS_EXTENSIONS,
                )
            )

            if (
                len(iris_samples)
                >= samples_per_dataset
            ):
                break

        if (
            len(iris_samples)
            >= samples_per_dataset
        ):
            break

    for file in iris_samples[
        :samples_per_dataset
    ]:

        try:
            size, mode, image_format = (
                _get_image_properties(file)
            )

            print(
                f"  {file.name:<25} "
                f"size={size[0]}x{size[1]}, "
                f"mode={mode}, "
                f"format={image_format}"
            )

            expected_size = (
                IRIS_IMAGE_WIDTH,
                IRIS_IMAGE_HEIGHT,
            )

            if size != expected_size:
                print(
                    "    [WARNING] Unexpected "
                    f"dimensions. Expected "
                    f"{expected_size[0]}x"
                    f"{expected_size[1]}"
                )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:

            print(
                f"  [ERROR] {file}: {error}"
            )

    print("=" * 70)


# ============================================================
# Dataset statistics
# ============================================================

def print_dataset_statistics() -> None:
    """
    Print expected dataset statistics.
    """

    fingerprint_images = (
        FINGERPRINT_NUM_SUBJECTS
        * FINGERPRINT_NUM_FINGERS_PER_SUBJECT
        * FINGERPRINT_IMPRESSIONS_PER_FINGER
    )

    iris_images = (
        IRIS_NUM_SUBJECTS
        * len(IRIS_EYES)
        * IRIS_IMPRESSIONS_PER_EYE
    )

    print("\n")
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    print("\nFingerprint:")
    print(
        f"  Subjects             : "
        f"{FINGERPRINT_NUM_SUBJECTS}"
    )
    print(
        f"  Fingers / subject    : "
        f"{FINGERPRINT_NUM_FINGERS_PER_SUBJECT}"
    )
    print(
        f"  Impressions / finger : "
        f"{FINGERPRINT_IMPRESSIONS_PER_FINGER}"
    )
    print(
        f"  Biometric classes    : "
        f"{FINGERPRINT_NUM_CLASSES}"
    )
    print(
        f"  Images / subject     : "
        f"{FINGERPRINT_NUM_FINGERS_PER_SUBJECT * FINGERPRINT_IMPRESSIONS_PER_FINGER}"
    )
    print(
        f"  Total images         : "
        f"{fingerprint_images}"
    )

    print("\nIris:")
    print(
        f"  Subjects             : "
        f"{IRIS_NUM_SUBJECTS}"
    )
    print(
        f"  Eyes / subject       : "
        f"{len(IRIS_EYES)}"
    )
    print(
        f"  Impressions / eye    : "
        f"{IRIS_IMPRESSIONS_PER_EYE}"
    )
    print(
        f"  Biometric classes    : "
        f"{IRIS_NUM_CLASSES}"
    )
    print(
        f"  Images / subject     : "
        f"{len(IRIS_EYES) * IRIS_IMPRESSIONS_PER_EYE}"
    )
    print(
        f"  Total images         : "
        f"{iris_images}"
    )

    print("\nSubject-wise split:")
    print(
        f"  Fingerprint train    : "
        f"{FINGERPRINT_TRAIN_SUBJECTS}"
    )
    print(
        f"  Fingerprint eval     : "
        f"{FINGERPRINT_EVAL_SUBJECTS}"
    )
    print(
        f"  Iris train           : "
        f"{IRIS_TRAIN_SUBJECTS}"
    )
    print(
        f"  Iris eval            : "
        f"{IRIS_EVAL_SUBJECTS}"
    )

    print("=" * 70)


# ============================================================
# Complete validation
# ============================================================

def validate_all() -> bool:
    """
    Run all raw dataset validation checks.
    """

    print_dataset_statistics()

    fingerprint_result = (
        validate_fingerprints()
    )

    iris_result = (
        validate_iris()
    )

    fingerprint_result.print_report()
    iris_result.print_report()

    if (
        fingerprint_result.passed
        and iris_result.passed
    ):

        integrity_result = (
            validate_image_integrity()
        )

        integrity_result.print_report()

        if integrity_result.passed:
            inspect_image_properties()

    else:

        integrity_result = ValidationResult(
            name="Image Integrity",
            passed=False,
            errors=[
                "Skipped because structural "
                "validation failed."
            ],
            warnings=[],
        )

    all_passed = (
        fingerprint_result.passed
        and iris_result.passed
        and integrity_result.passed
    )

    print("\n")
    print("=" * 70)

    if all_passed:
        print(
            "DATASET VALIDATION PASSED"
        )

    else:
        print(
            "DATASET VALIDATION FAILED"
        )

    print("=" * 70)

    return all_passed


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    validate_all()