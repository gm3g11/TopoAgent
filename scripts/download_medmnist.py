#!/usr/bin/env python
"""Download the 11 MedMNIST (224x224) datasets used by TopoBenchmark.

Saves ``<flag>_224.npz`` files to ``MEDMNIST_PATH`` (default ``~/.medmnist``) —
exactly the filenames and location that ``RuleBenchmark/benchmark4/data_loader.py``
expects. The 15 external datasets are not auto-downloadable; see
``docs/benchmark3_datasets.md`` for their official sources.

Usage:
    python scripts/download_medmnist.py
    MEDMNIST_PATH=/data/medmnist python scripts/download_medmnist.py

Requires: medmnist>=3.0.0 (for the 224x224 "MedMNIST+" variants).
"""
import os
from pathlib import Path

# The 11 MedMNIST flags used by TopoBenchmark (lowercase = the npz prefix on disk).
FLAGS = [
    "bloodmnist", "tissuemnist", "pathmnist", "octmnist", "organamnist",
    "retinamnist", "pneumoniamnist", "breastmnist", "dermamnist",
    "organcmnist", "organsmnist",
]


def main():
    root = Path(os.environ.get("MEDMNIST_PATH", os.path.expanduser("~/.medmnist")))
    root.mkdir(parents=True, exist_ok=True)

    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        raise SystemExit("medmnist is required: pip install 'medmnist>=3.0.0'")

    for flag in FLAGS:
        out = root / f"{flag}_224.npz"
        if out.exists():
            print(f"[skip] {out.name} already present")
            continue
        print(f"[download] {flag} (224x224) -> {out}")
        cls = getattr(medmnist, INFO[flag]["python_class"])
        # Instantiating with download=True writes <flag>_224.npz into root.
        cls(split="train", download=True, size=224, root=str(root))

    print(f"\nDone. {len(FLAGS)} MedMNIST datasets are in {root}")
    print("Set MEDMNIST_PATH to this directory when running the benchmark.")


if __name__ == "__main__":
    main()
