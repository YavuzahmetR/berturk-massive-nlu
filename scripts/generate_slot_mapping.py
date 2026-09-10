import os
import json

def generate_canonical_slots():
    print("[SLOTS PREFLIGHT] Extracting slot mapping from local Amazon MASSIVE train split")
    
    # 1. paths configuration
    raw_train_path = os.path.join("data", "raw", "train.jsonl")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(raw_train_path):
        raise FileNotFoundError(
            f"[CRITICAL] local train file missing at: {raw_train_path}. "
            "Please run prepare_dataset.py first."
        )

    # 2. Scan dataset records to discover all unique raw slots dynamically
    unique_slots = set()
    
    with open(raw_train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # Extract slots listed inside the explicit slot_method dictionary layer
            if "slot_method" in data and "slot" in data["slot_method"]:
                for slot_name in data["slot_method"]["slot"]:
                    unique_slots.add(slot_name)

    # 3. Construct a standard inside-outside-beginning (BIO) token layout schema
    # "O" means outside any entity. Every active slot gets a B- and I- prefix.
    bio_tags = ["O"]
    for slot in sorted(list(unique_slots)):
        bio_tags.append(f"B-{slot}")
        bio_tags.append(f"I-{slot}")

    # 4. Convert array lists into deterministic numeric registry lookup mappings
    slot_to_id = {tag_name: idx for idx, tag_name in enumerate(bio_tags)}
    
    print(f" Unique raw slots discovered: {len(unique_slots)}")
    print(f" Canonical BIO slot tags count verified: {len(slot_to_id)}")
    
    # 5. Lock down the final artifact contract schema
    slot_mapping_path = os.path.join(processed_dir, "slot_to_id.json")
    with open(slot_mapping_path, "w", encoding="utf-8") as f:
        json.dump(slot_to_id, f, indent=4, ensure_ascii=False)
        
    print(f"[ARTIFACT LOCKED] Slot mapping contract saved to: {slot_mapping_path}")

if __name__ == "__main__":
    generate_canonical_slots()
