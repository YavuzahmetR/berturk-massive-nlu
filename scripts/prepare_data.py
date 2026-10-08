"""Prepare the existing MASSIVE train/validation/test data contract."""

from src.data.prepare import run_dataset_preflight_and_manifest


if __name__ == "__main__":
    run_dataset_preflight_and_manifest()
