#!/usr/bin/env python3
"""Extract 10 sample images from each dataset for visualization/demo purposes.

Datasets:
  My 5 representative picks (one per object type):
    1. SIPaKMeD        (discrete_cells,  RGB, ~1024x1024)
    2. BreakHis        (glands_lumens,   RGB, 700x460)
    3. BrainTumorMRI   (organ_shape,     Gray, 512x512)
    4. ISIC2019        (surface_lesions, RGB, ~1022)
    5. APTOS2019       (vessel_trees,    RGB, ~1050)

  User-requested additional datasets:
    6. BloodMNIST      (discrete_cells,  RGB, 224x224)
    7. OrganAMNIST     (organ_shape,     Gray, 224x224)
    8. OrganCMNIST     (organ_shape,     Gray, 224x224)
    9. DRIVE           (vessel_trees,    RGB, retinal segmentation)

Output: results/sample_images/<dataset_name>/img_<i>_class<label>.png
"""

import sys
import os
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "results" / "sample_images"

# ── DRIVE paths (not in benchmark, loaded directly) ──
DRIVE_ROOT = Path("/afs/crc/group/ball_lab/gmeng_cl/cl_new/datasets/drive/DRIVE")


def save_image(arr, path):
    """Save numpy array as PNG. arr can be float [0,1] or uint8."""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode='L')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr, mode='RGB')
    else:
        img = Image.fromarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


def extract_from_benchmark(dataset_name, n=10, seed=42):
    """Extract n images using the benchmark4 data loader."""
    from RuleBenchmark.benchmark4.data_loader import load_dataset
    from RuleBenchmark.benchmark4.config import DATASETS

    cfg = DATASETS[dataset_name]
    color_mode = cfg['color_mode']

    print(f"\n{'='*60}")
    print(f"  {dataset_name}  (object_type={cfg['object_type']}, color={color_mode})")
    print(f"{'='*60}")

    # Load a small batch (just n images to save memory)
    images, labels, class_names = load_dataset(
        dataset_name, n_samples=n, seed=seed, color_mode=color_mode,
    )

    out_dir = OUTPUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(n, len(images))):
        label = int(labels[i])
        cls = class_names[label] if label < len(class_names) else str(label)
        fname = f"img_{i:02d}_class_{cls}.png"
        save_image(images[i], out_dir / fname)

    print(f"  Saved {min(n, len(images))} images to {out_dir}")
    print(f"  Image shape: {images[0].shape}, dtype: {images[0].dtype}")
    print(f"  Classes represented: {sorted(set(int(l) for l in labels[:n]))}")


def extract_drive(n=10):
    """Extract n images from DRIVE (retinal vessel segmentation dataset)."""
    print(f"\n{'='*60}")
    print(f"  DRIVE  (vessel_trees, RGB, retinal segmentation)")
    print(f"{'='*60}")

    out_dir = OUTPUT_DIR / "DRIVE"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images from training + test
    all_paths = []
    for split, subdir in [("training", "training/images"), ("test", "test/images")]:
        img_dir = DRIVE_ROOT / subdir
        if img_dir.exists():
            for f in sorted(os.listdir(img_dir)):
                if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                    all_paths.append((str(img_dir / f), split))

    # Take first n (evenly from train/test)
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_paths), size=min(n, len(all_paths)), replace=False)
    indices.sort()

    for i, idx in enumerate(indices):
        path, split = all_paths[idx]
        img = Image.open(path).convert('RGB')
        arr = np.array(img)
        fname = f"img_{i:02d}_{split}_{Path(path).stem}.png"
        save_image(arr, out_dir / fname)
        if i == 0:
            print(f"  Image size: {img.size}, shape: {arr.shape}")

    print(f"  Saved {len(indices)} images to {out_dir}")
    print(f"  Total available: {len(all_paths)} (20 training + 20 test)")


def main():
    # Benchmark datasets (use existing data loader)
    benchmark_datasets = [
        'SIPaKMeD',
        'BreakHis',
        'BrainTumorMRI',
        'ISIC2019',
        'APTOS2019',
        'BloodMNIST',
        'OrganAMNIST',
        'OrganCMNIST',
    ]

    for ds in benchmark_datasets:
        try:
            extract_from_benchmark(ds, n=10, seed=42)
        except Exception as e:
            print(f"  ERROR extracting {ds}: {e}")
            import traceback
            traceback.print_exc()

    # DRIVE (not in benchmark, load directly)
    try:
        extract_drive(n=10)
    except Exception as e:
        print(f"  ERROR extracting DRIVE: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  All done! Images saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Summary
    for d in sorted(OUTPUT_DIR.iterdir()):
        if d.is_dir():
            count = len(list(d.glob("*.png")))
            print(f"  {d.name}: {count} images")


if __name__ == '__main__':
    main()
