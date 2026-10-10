import os
import random

import numpy as np
import torch

from config import RANDOM_SEED


def set_seed(
    seed: int = RANDOM_SEED,
    *,
    deterministic: bool = True,
    cudnn_benchmark: bool | None = None,
    enable_tf32: bool = True,
) -> None:
    """Seed libraries and configure CUDA performance/reproducibility.

    Deterministic kernels remain the default. Fixed-size GPU training can
    explicitly use ``deterministic=False`` and benchmark mode for throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cudnn_benchmark is None:
        cudnn_benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = enable_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
