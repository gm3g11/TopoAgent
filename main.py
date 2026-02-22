"""TopoAgent Main Entry Point.

This is the main entry point for running TopoAgent.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from topoagent import TopoAgent, create_topoagent
from topoagent.tools import get_all_tools


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TopoAgent: Medical AI Agent for Topological Data Analysis"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="Classify this medical image using topological features",
        help="Classification query/task"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="LLM model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--max-rounds", "-r",
        type=int,
        default=3,
        help="Maximum reasoning rounds (default: 3)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools and exit"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # List tools mode
    if args.list_tools:
        print("\n=== TopoAgent Available Tools ===\n")
        tools = get_all_tools()
        for name, tool in tools.items():
            print(f"  {name}: {tool.description[:80]}...")
        print(f"\nTotal: {len(tools)} tools")
        return

    # Interactive mode
    if args.interactive:
        run_interactive(args.model, args.max_rounds)
        return

    # Single image mode
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)

        print(f"\n=== TopoAgent Analysis ===")
        print(f"Image: {args.image}")
        print(f"Query: {args.query}")
        print(f"Model: {args.model}")
        print(f"Max Rounds: {args.max_rounds}")
        print()

        # Create agent
        agent = create_topoagent(
            model_name=args.model,
            max_rounds=args.max_rounds
        )

        # Run classification
        print("Running analysis...")
        result = agent.classify(
            image_path=args.image,
            query=args.query
        )

        # Display results
        print_result(result)
    else:
        parser.print_help()


def run_interactive(model_name: str, max_rounds: int):
    """Run TopoAgent in interactive mode.

    Args:
        model_name: LLM model name
        max_rounds: Max reasoning rounds
    """
    print("\n=== TopoAgent Interactive Mode ===")
    print("Type 'quit' to exit, 'help' for commands\n")

    agent = create_topoagent(
        model_name=model_name,
        max_rounds=max_rounds
    )

    while True:
        try:
            # Get image path
            image_path = input("Image path (or 'quit'): ").strip()

            if image_path.lower() == 'quit':
                print("Goodbye!")
                break

            if image_path.lower() == 'help':
                print("\nCommands:")
                print("  <image_path>  - Analyze an image")
                print("  tools         - List available tools")
                print("  quit          - Exit")
                continue

            if image_path.lower() == 'tools':
                tools = agent.get_available_tools()
                print(f"\nAvailable tools: {', '.join(tools)}")
                continue

            if not os.path.exists(image_path):
                print(f"Error: File not found: {image_path}")
                continue

            # Get query
            query = input("Query (or press Enter for default): ").strip()
            if not query:
                query = "Classify this medical image using topological features"

            # Run analysis
            print("\nAnalyzing...")
            result = agent.classify(image_path=image_path, query=query)
            print_result(result)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def print_result(result: dict):
    """Print classification result.

    Args:
        result: Classification result dictionary
    """
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    print(f"\nClassification: {result.get('classification', 'Unknown')}")
    print(f"Confidence: {result.get('confidence', 0):.1f}%")
    print(f"Rounds Used: {result.get('rounds_used', 0)}")

    print(f"\nTools Used: {', '.join(result.get('tools_used', []))}")

    if result.get('reasoning_trace'):
        print("\nReasoning Trace:")
        for i, step in enumerate(result['reasoning_trace'], 1):
            print(f"  {i}. {step}")

    if result.get('evidence'):
        print("\nEvidence:")
        for i, ev in enumerate(result['evidence'][:3], 1):
            print(f"  {i}. {str(ev)[:100]}...")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
