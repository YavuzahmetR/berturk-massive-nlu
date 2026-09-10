import os
import json
from datasets import load_dataset


def run_dataset_preflight_and_manifest():
    print("[DATASET PREFLIGHT] Amazon MASSIVE tr-TR")

    # 1. Initialize directory structures
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 2. Load dataset partitions
    splits = {
        "train": load_dataset("AmazonScience/massive", "tr-TR", split="train", trust_remote_code=True),
        "validation": load_dataset("AmazonScience/massive", "tr-TR", split="validation", trust_remote_code=True),
        "test": load_dataset("AmazonScience/massive", "tr-TR", split="test", trust_remote_code=True)
    }

    # 3. Verify row counts for data integrity
    expected_counts = {"train": 11514, "validation": 2033, "test": 2974}

    for split_name, dataset_split in splits.items():
        actual_count = len(dataset_split)
        expected_count = expected_counts[split_name]

        if actual_count != expected_count:
            raise ValueError(
                f"[CRITICAL] Split integrity broken for '{split_name}'. "
                f"Expected: {expected_count}, Found: {actual_count}"
            )
        print(f"{split_name.capitalize()} sample count verified: {actual_count}")

    # 4. Check for data leakage across partitions
    train_ids = set(splits["train"]["id"])
    val_ids = set(splits["validation"]["id"])
    test_ids = set(splits["test"]["id"])

    if not train_ids.isdisjoint(val_ids) or not train_ids.isdisjoint(test_ids) or not val_ids.isdisjoint(test_ids):
        raise ValueError("[LEAKAGE DETECTED] Overlapping row IDs found across dataset partitions!")

    # 5. Extract intent metadata directly from official schema
    intent_features = splits["train"].features["intent"].names
    intent_to_id = {name: idx for idx, name in enumerate(intent_features)}
    
    print(f"Total canonical intent count verified: {len(intent_to_id)}")

    # 6. Save raw jsonl cache files for training independence
    for split_name, dataset_split in splits.items():
        output_path = os.path.join(raw_dir, f"{split_name}.jsonl")
        dataset_split.to_json(output_path)

    # 7. Lock down the deterministic dictionary contracts as JSON artifacts
    intent_mapping_path = os.path.join(processed_dir, "intent_to_id.json")
    with open(intent_mapping_path, "w", encoding="utf-8") as f:
        json.dump(intent_to_id, f, indent=4, ensure_ascii=False)
        
    print(f"\n[ARTIFACT LOCKED] Intent mapping saved to: {intent_mapping_path}")
    print("[SUCCESS] Dataset Preflight completed successfully. Ready for training!")


if __name__ == "__main__":
    run_dataset_preflight_and_manifest()
