"""Training entry point, preserving the previous public helper names."""

from src.training.train import CHECKPOINT_PATH, compute_class_weights, load_local_jsonl, main


if __name__ == "__main__":
    main()
