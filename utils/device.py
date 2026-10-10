import torch


def get_device() -> torch.device:
    """
    Select the best available compute device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    """Print information about the selected device."""

    print("\n")
    print("=" * 60)
    print("COMPUTE DEVICE")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(
            f"CUDA version: "
            f"{torch.version.cuda}"
        )

        total_memory = (
            torch.cuda.get_device_properties(device)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"VRAM: {total_memory:.2f} GB"
        )

    print("=" * 60)