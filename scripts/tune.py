"""Optional tuning entry point, preserving previous public helper names."""

from src.training.tune import (
    compute_class_weights_from_counts,
    evaluate_f1,
    load_local_jsonl,
    main,
    precompute_label_counts,
    train_one_trial,
)


if __name__ == "__main__":
    main()
