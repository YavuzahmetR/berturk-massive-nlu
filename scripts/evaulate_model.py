"""Compatibility entry point for the original misspelled module name."""

from scripts.evaluate_model import load_local_jsonl, run_final_test_evaluation


if __name__ == "__main__":
    run_final_test_evaluation()
