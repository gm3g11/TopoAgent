#!/usr/bin/env python
"""Run TopoAgent on PHG-Net's 4 datasets to determine optimal descriptors.

Runs TopoAgent (skills_mode) on 10 sample images per dataset:
  - ISIC 2018 (skin lesions, 7-class)
  - Prostate (histopathology, 3-class)
  - CBIS-DDSM (mammography, 2-class)
  - ORIGA/Glaucoma (retinal fundus, 2-class)

Usage:
    python scripts/run_phg_datasets.py --model gpt-4o --n-samples 10
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# Dataset configurations
DATASET_CONFIGS = {
    "ISIC2018": {
        "description": "ISIC 2018 Skin Lesion Classification (7-class)",
        "image_dir": os.path.expanduser("~/data/raw_data/ISIC2018/ISIC2018_Task3_Training_Input"),
        "extensions": [".jpg"],
        "query": "Analyze this dermoscopy skin lesion image using topological features. "
                 "Identify the optimal topological descriptor and parameters for this type of medical image.",
        "object_type": "surface_lesions",
        "n_classes": 7,
        "class_names": ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
    },
    "Prostate": {
        "description": "Prostate Cancer Histopathology (3-class: Gleason 3/4/5)",
        "image_dir": os.path.expanduser("~/data/raw_data/Prostate/rois"),
        "extensions": [".tiff"],
        "query": "Analyze this H&E stained prostate histopathology image using topological features. "
                 "Identify the optimal topological descriptor and parameters for this type of medical image.",
        "object_type": "glands_lumens",
        "n_classes": 3,
        "class_names": ["Gleason3", "Gleason4", "Gleason5"],
    },
    "CBIS_DDSM": {
        "description": "CBIS-DDSM Mammography (2-class: Benign/Malignant)",
        "image_dir": os.path.expanduser("~/data/raw_data/CBIS_DDSM/jpeg"),
        "extensions": [".jpg"],
        "query": "Analyze this mammography breast image using topological features. "
                 "Identify the optimal topological descriptor and parameters for this type of medical image.",
        "object_type": "organ_shape",
        "n_classes": 2,
        "class_names": ["BENIGN", "MALIGNANT"],
    },
    "Glaucoma": {
        "description": "ORIGA Glaucoma Retinal Fundus (2-class)",
        "image_dir": os.path.expanduser("~/data/raw_data/Glaucoma/Glaucoma"),
        "extensions": [".jpg"],
        "query": "Analyze this retinal fundus image for glaucoma using topological features. "
                 "Identify the optimal topological descriptor and parameters for this type of medical image.",
        "object_type": "vessel_trees",
        "n_classes": 2,
        "class_names": ["Normal", "Glaucoma"],
    },
}


def find_sample_images(image_dir: str, extensions: list, n_samples: int, seed: int = 42) -> list:
    """Find and randomly sample image files from a directory."""
    image_dir = Path(image_dir)
    images = []

    for ext in extensions:
        # Search top-level and one level deep (for CBIS-DDSM nested structure)
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*/*{ext}"))

    images = sorted(set(images))  # Deduplicate and sort

    if not images:
        print(f"  WARNING: No images found in {image_dir} with extensions {extensions}")
        return []

    np.random.seed(seed)
    indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
    return [str(images[i]) for i in sorted(indices)]


def run_dataset(agent, dataset_name: str, config: dict, n_samples: int, seed: int = 42):
    """Run TopoAgent on a single dataset."""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} — {config['description']}")
    print(f"{'='*70}")

    # Find sample images
    images = find_sample_images(
        config["image_dir"], config["extensions"], n_samples, seed
    )

    if not images:
        return {"error": "No images found", "dataset": dataset_name}

    print(f"  Found {len(images)} sample images")

    results = []
    descriptors_used = []

    for i, img_path in enumerate(images):
        print(f"\n  [{i+1}/{len(images)}] {Path(img_path).name}")

        start_time = time.time()
        try:
            result = agent.classify(
                image_path=img_path,
                query=config["query"]
            )

            elapsed = time.time() - start_time

            # Extract descriptor recommendation
            descriptor = None
            raw_answer = result.get("raw_answer", "")
            report = result.get("report", {})

            if isinstance(report, dict):
                descriptor = report.get("descriptor")

            if not descriptor and isinstance(raw_answer, dict):
                descriptor = raw_answer.get("descriptor")

            # Also check tools_used for descriptor tools
            tools_used = result.get("tools_used", [])
            descriptor_tools = [t for t in tools_used if t in [
                "persistence_image", "persistence_landscapes", "betti_curves",
                "persistence_silhouette", "persistence_entropy", "persistence_statistics",
                "tropical_coordinates", "template_functions", "atol_descriptor",
                "persistence_codebook", "minkowski_functionals",
                "euler_characteristic_curve", "euler_characteristic_transform",
                "edge_histogram", "lbp_texture"
            ]]

            if not descriptor and descriptor_tools:
                descriptor = descriptor_tools[-1]  # Last descriptor tool used

            if descriptor:
                descriptors_used.append(descriptor)

            print(f"    Descriptor: {descriptor or 'N/A'}")
            print(f"    Confidence: {result.get('confidence', 0):.1f}%")
            print(f"    Tools: {', '.join(tools_used)}")
            print(f"    Time: {elapsed:.1f}s")

            results.append({
                "image": Path(img_path).name,
                "descriptor": descriptor,
                "confidence": result.get("confidence", 0),
                "tools_used": tools_used,
                "rounds_used": result.get("rounds_used", 0),
                "classification": result.get("classification", "Unknown"),
                "time_seconds": elapsed,
                "report": report if isinstance(report, dict) else str(report)[:500],
            })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"    ERROR: {e}")
            results.append({
                "image": Path(img_path).name,
                "error": str(e),
                "time_seconds": elapsed,
            })

        # Small delay to avoid rate limits
        time.sleep(2)

    # Summary
    descriptor_counts = Counter(descriptors_used)
    recommended = descriptor_counts.most_common(1)[0] if descriptor_counts else ("N/A", 0)

    print(f"\n  --- {dataset_name} Summary ---")
    print(f"  Images processed: {len(results)}")
    print(f"  Descriptors used: {dict(descriptor_counts)}")
    print(f"  Most recommended: {recommended[0]} ({recommended[1]}/{len(images)} images)")

    successful = [r for r in results if "error" not in r]
    if successful:
        avg_conf = np.mean([r["confidence"] for r in successful])
        avg_time = np.mean([r["time_seconds"] for r in successful])
        print(f"  Avg confidence: {avg_conf:.1f}%")
        print(f"  Avg time: {avg_time:.1f}s")

    return {
        "dataset": dataset_name,
        "description": config["description"],
        "object_type": config["object_type"],
        "n_samples": len(images),
        "descriptor_counts": dict(descriptor_counts),
        "recommended_descriptor": recommended[0],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run TopoAgent on PHG-Net datasets")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o", help="LLM model")
    parser.add_argument("--n-samples", "-n", type=int, default=10, help="Images per dataset")
    parser.add_argument("--max-rounds", "-r", type=int, default=3, help="Max reasoning rounds")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--datasets", "-d", nargs="+",
                        default=["ISIC2018", "Prostate", "CBIS_DDSM", "Glaucoma"],
                        help="Datasets to evaluate")
    parser.add_argument("--output", "-o", type=str,
                        default="results/phg_descriptor_selection.json",
                        help="Output JSON file")
    parser.add_argument("--skills-mode", action="store_true", default=True,
                        help="Use skills mode (default: True)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"TopoAgent — PHG-Net Dataset Descriptor Selection")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Samples per dataset: {args.n_samples}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Skills mode: {args.skills_mode}")
    print(f"Datasets: {args.datasets}")

    # Create agent
    from topoagent import create_topoagent
    agent = create_topoagent(
        model_name=args.model,
        max_rounds=args.max_rounds,
        skills_mode=args.skills_mode,
        temperature=0.1,
    )

    # Run on each dataset
    all_results = {}
    for dataset_name in args.datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f"WARNING: Unknown dataset '{dataset_name}', skipping")
            continue

        config = DATASET_CONFIGS[dataset_name]
        dataset_result = run_dataset(agent, dataset_name, config, args.n_samples, args.seed)
        all_results[dataset_name] = dataset_result

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — Recommended Descriptors")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<15} {'Recommended Descriptor':<30} {'Object Type':<20}")
    print("-" * 65)
    for name, result in all_results.items():
        desc = result.get("recommended_descriptor", "N/A")
        obj_type = result.get("object_type", "N/A")
        print(f"{name:<15} {desc:<30} {obj_type:<20}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "n_samples_per_dataset": args.n_samples,
        "max_rounds": args.max_rounds,
        "skills_mode": args.skills_mode,
        "datasets": all_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
