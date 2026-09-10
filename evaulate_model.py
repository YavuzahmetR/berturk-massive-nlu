import os
import json
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.modeling.joint_nlu_model import (
    JointTurkishNLUModel,
    MassiveAnnotationParser,
    TokenAlignmentEncoder,
)
from src.data.dataset import MassiveJointNLUDataset, joint_nlu_collate_fn


def load_local_jsonl(file_path: str) -> list:
    """Reads test split records securely from independent local storage cache."""
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def run_final_test_evaluation():
    print("[FINAL EVALUATION] Evaluating on the Locked Official Test Split...\n")

    # 1. Load configuration metadata
    config_path = os.path.join("configs", "joint_bert.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Load frozen data lookups
    with open(os.path.join("data", "processed", "intent_to_id.json"), "r", encoding="utf-8") as f:
        intent_to_id = json.load(f)
    with open(os.path.join("data", "processed", "slot_to_id.json"), "r", encoding="utf-8") as f:
        slot_to_id = json.load(f)

    # 3. Load independent test split
    test_raw = load_local_jsonl(os.path.join("data", "raw", "test.jsonl"))

    # 4. Initialize tokenizer and evaluation workers
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    parser = MassiveAnnotationParser()
    encoder = TokenAlignmentEncoder(slot_to_id=slot_to_id)

    # 5. Build evaluation dataset + dataloader
    test_dataset = MassiveJointNLUDataset(
        raw_examples=test_raw,
        parser=parser,
        encoder=encoder,
        tokenizer=tokenizer,
        intent_to_id=intent_to_id,
        max_length=cfg["model"]["max_length"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=joint_nlu_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # 6. Allocate device and load best checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Computing Hardware Engine Target: {device}")

    model = JointTurkishNLUModel(
        model_name=cfg["model"]["model_name"],
        num_intents=len(intent_to_id),
        num_slots=len(slot_to_id),
        dropout_rate=cfg["model"]["dropout_rate"],
    )

    checkpoint_path = "best_joint_nlu_model.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[CRITICAL] Best validation checkpoint missing at: {checkpoint_path}. Train first."
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 7. Metrics counters
    correct_intents = 0
    total_samples = 0
    total_valid_slot_tokens = 0
    correct_slot_tokens = 0

    print(f"Running inference over {len(test_dataset)} unseen test instances...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intent_label = batch["intent_label"].to(device)

            intent_logits, slot_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # --- Intent accuracy ---
            intent_preds = torch.argmax(intent_logits, dim=-1)
            correct_intents += (intent_preds == intent_label).sum().item()
            total_samples += intent_label.size(0)

            # --- Slot token accuracy (ignoring -100 padding) ---
            slot_preds = torch.argmax(slot_logits, dim=-1)
            flat_labels = slot_labels.view(-1)
            flat_preds = slot_preds.view(-1)
            valid_mask = flat_labels != -100

            correct_slot_tokens += ((flat_preds == flat_labels) & valid_mask).sum().item()
            total_valid_slot_tokens += valid_mask.sum().item()

    # 8. Final metrics
    final_intent_acc = (correct_intents / total_samples) * 100
    final_slot_acc = (correct_slot_tokens / total_valid_slot_tokens) * 100

    print("\n ================= FINAL LOCKED TEST METRICS ================= 📊")
    print(f" Unseen Test Sample Population Covered: {total_samples}")
    print(f" Final Intent Classification Accuracy : {final_intent_acc:.2f}%")
    print(f" Final Slot Labeling Token Accuracy   : {final_slot_acc:.2f}% (Ignoring -100 Padding)")
    print("=================================================================\n")


if __name__ == "__main__":
    run_final_test_evaluation()