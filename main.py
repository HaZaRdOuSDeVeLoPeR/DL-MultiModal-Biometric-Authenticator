from utils.data_validator import validate_all
from utils.cleanup import cache_cleanup
from utils.device import get_device, print_device_info
from utils.random_utils import set_seed

from config import ENABLE_TF32, TRAIN_DETERMINISTIC
from src.datasets import prepare_datasets
from src.train import train_all_models
from src.embeddings import extract_all_embeddings


def main() -> None:
    set_seed(
        deterministic=TRAIN_DETERMINISTIC,
        enable_tf32=ENABLE_TF32,
    )

    print("Deep Learning Fingerprint + Iris Authentication")
    print("=" * 60)

    if not validate_all():
        return

    datasets = prepare_datasets()

    print("\n")
    print("=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)
    for name, dataset in datasets.items():
        print(f"{name:<25}: {len(dataset)}")
    print("=" * 60)

    device = get_device()
    print_device_info(device)

    train_results = train_all_models(datasets=datasets, device=device)

    print("\n")
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for name, result in train_results.items():
        print(f"{name:<35}: {result['status']}")
    print("=" * 60)

    extract_all_embeddings(datasets=datasets, device=device)


if __name__ == "__main__":
    try:
        main()
    finally:
        cache_cleanup()
