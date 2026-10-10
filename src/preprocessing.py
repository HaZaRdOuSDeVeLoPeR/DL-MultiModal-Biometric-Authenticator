from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    FINGERPRINT_ROTATION_DEGREES,
    FINGERPRINT_AFFINE_DEGREES,
    FINGERPRINT_AFFINE_TRANSLATE,
    FINGERPRINT_AFFINE_SCALE,
    IRIS_COLOR_JITTER_BRIGHTNESS,
    IRIS_COLOR_JITTER_CONTRAST,
    PAD_FILL,
)

class ResizeAndPad:
    """
    Resize an image while preserving its aspect ratio, then
    pad it to IMAGE_SIZE x IMAGE_SIZE.
    """

    def __init__(self, size: int = IMAGE_SIZE, fill: int = PAD_FILL):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid image dimensions: {image.size}"
            )

        # Scale image so the longest dimension becomes `size`.
        scale = self.size / max(width, height)

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        image = image.resize(
            (new_width, new_height),
            resample=Image.Resampling.BILINEAR,
        )

        # Create square canvas.
        padded = Image.new(
            mode=image.mode,
            size=(self.size, self.size),
            color=self.fill,
        )

        # Center the resized image.
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2

        padded.paste(image, (left, top))

        return padded


class EnsureRGB:
    """
    Convert every biometric image to 3-channel RGB.

    The fingerprint and iris datasets contain L, P, and RGB
    source images. Converting all images to RGB gives the
    ImageNet-pretrained CNNs a consistent 3-channel input.

    For grayscale/palette images, this does not introduce
    color information; the three channels contain equivalent
    intensity information.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")

# --- Custom Biometric Transformers ---

class EnhanceFingerprintStructure:
    """
    Applies Local Contrast Enhancement (CLAHE) to handle pressure variances 
    specifically for the CASIA Fingerprint V5 dataset.
    """
    def __call__(self, image: Image.Image) -> Image.Image:
        # Convert PIL to Grayscale Numpy Array
        img_np = np.array(image.convert("L"))
        
        # Apply CLAHE to sharpen ridge structures
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_np)
        
        return Image.fromarray(enhanced)


class EnhanceIrisTexture:
    """
    Improves pattern visibility in CASIA Iris Thousand by mitigating 
    near-infrared specular reflections and evening out illumination.
    """
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image.convert("L"))
        
        # Equalize histogram locally to handle lighting variations across the eye
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        enhanced = clahe.apply(img_np)
        
        return Image.fromarray(enhanced)


# --- Preprocessing Pipelines ---

def get_fingerprint_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            EnhanceFingerprintStructure(),  # 1. Structural Repair First
            ResizeAndPad(),                 # 2. Geometry Standardization (Use LANCZOS inside)
            transforms.RandomRotation(
                degrees=FINGERPRINT_ROTATION_DEGREES,
                fill=PAD_FILL,
            ),
            transforms.RandomAffine(
                degrees=FINGERPRINT_AFFINE_DEGREES,
                translate=FINGERPRINT_AFFINE_TRANSLATE,
                scale=FINGERPRINT_AFFINE_SCALE,
                fill=PAD_FILL,
            ),
            EnsureRGB(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_iris_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            EnhanceIrisTexture(),          # 1. Texture Boost First
            ResizeAndPad(),                # 2. Geometric Scale
            transforms.ColorJitter(
                brightness=IRIS_COLOR_JITTER_BRIGHTNESS,
                contrast=IRIS_COLOR_JITTER_CONTRAST,
            ),
            EnsureRGB(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform(modality: str = "fingerprint") -> transforms.Compose:
    """
    Deterministic evaluation pipeline. Includes structural enhancements 
    to prevent domain shift during verification.
    """
    enhancer = EnhanceFingerprintStructure() if modality == "fingerprint" else EnhanceIrisTexture()
    
    return transforms.Compose(
        [
            enhancer,
            ResizeAndPad(),
            EnsureRGB(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
