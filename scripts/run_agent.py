#!/usr/bin/env python
"""Run TopoAgent on a single image or batch of images.

Usage:
    python scripts/run_agent.py --image path/to/image.png
    python scripts/run_agent.py --image-dir path/to/images/ --output results.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from topoagent import create_topoagent


def run_single_image(
    agent,
    image_path: str,
    query: str = "Classify this medical image using topological features"
) -> Dict[str, Any]:
    """Run agent on a single image.

    Args:
        agent: TopoAgent instance
        image_path: Path to image
        query: Classification query

    Returns:
        Classification result
    """
    print(f"\nProcessing: {image_path}")

    result = agent.classify(
        image_path=image_path,
        query=query
    )

    print(f"  Classification: {result.get('classification', 'Unknown')}")
    print(f"  Confidence: {result.get('confidence', 0):.1f}%")
    print(f"  Rounds: {result.get('rounds_used', 0)}")

    return result


def run_batch(
    agent,
    image_dir: str,
    query: str,
    extensions: List[str] = ['.png', '.jpg', '.jpeg', '.dcm']
) -> List[Dict[str, Any]]:
    """Run agent on a directory of images.

    Args:
        agent: TopoAgent instance
        image_dir: Directory containing images
        query: Classification query
        extensions: Valid file extensions

    Returns:
        List of results
    """
    results = []
    image_dir = Path(image_dir)

    # Find all images
    images = []
    for ext in extensions:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))

    print(f"Found {len(images)} images in {image_dir}")

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}]", end="")
        try:
            result = run_single_image(agent, str(image_path), query)
            result["image_path"] = str(image_path)
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "image_path": str(image_path),
                "error": str(e)
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Run TopoAgent on images")

    # Input options
    parser.add_argument("--image", "-i", type=str, help="Single image path")
    parser.add_argument("--image-dir", "-d", type=str, help="Directory of images")

    # Agent options
    parser.add_argument("--model", "-m", type=str, default="gpt-4o", help="LLM model")
    parser.add_argument("--max-rounds", "-r", type=int, default=3, help="Max rounds")
    parser.add_argument("--query", "-q", type=str,
                       default="Classify this medical image using topological features",
                       help="Classification query")

    # Output options
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.print_help()
        sys.exit(1)

    # Create agent
    print(f"Initializing TopoAgent (model={args.model}, max_rounds={args.max_rounds})")
    agent = create_topoagent(
        model_name=args.model,
        max_rounds=args.max_rounds
    )

    # Run
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        results = [run_single_image(agent, args.image, args.query)]
    else:
        if not os.path.isdir(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            sys.exit(1)
        results = run_batch(agent, args.image_dir, args.query)

    # Save output
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "max_rounds": args.max_rounds,
            "query": args.query,
            "num_images": len(results),
            "results": results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Images processed: {len(results)}")
    successful = [r for r in results if "error" not in r]
    print(f"Successful: {len(successful)}")
    if successful:
        avg_confidence = sum(r.get("confidence", 0) for r in successful) / len(successful)
        avg_rounds = sum(r.get("rounds_used", 0) for r in successful) / len(successful)
        print(f"Avg Confidence: {avg_confidence:.1f}%")
        print(f"Avg Rounds: {avg_rounds:.1f}")


if __name__ == "__main__":
    main()
