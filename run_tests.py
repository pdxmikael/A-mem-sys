import argparse
import sys

from test_suite.runner import main as suite_main


def run():
    parser = argparse.ArgumentParser(description="Convenience wrapper to run the retrieval test suite")
    parser.add_argument("--raw", help="Path to combined raw notes+tests JSON")
    parser.add_argument("--dataset", default="test_suite/examples/dataset.json", help="Path to dataset JSON")
    parser.add_argument("--tests", default="test_suite/examples/tests.json", help="Path to tests JSON")
    parser.add_argument("--persist", default="./memory_db", help="Chroma persist directory")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--use_shaper", action="store_true", help="Enable QueryShaper during retrieval")
    parser.add_argument("--k", type=int, default=10, help="Max candidates to retrieve for metrics")
    args = parser.parse_args()

    # Forward arguments to test_suite.runner.main (which parses sys.argv)
    argv = ["test_suite", "--persist", args.persist, "--model", args.model, "--k", str(args.k)]
    if args.raw:
        argv.extend(["--raw", args.raw])
    else:
        argv.extend(["--dataset", args.dataset, "--tests", args.tests])
    if args.use_shaper:
        argv.append("--use_shaper")

    prev = sys.argv[:]
    try:
        sys.argv = argv
        suite_main()
    finally:
        sys.argv = prev


if __name__ == "__main__":
    run()