import os
import json
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import precision_score as seq_precision
from seqeval.metrics import recall_score as seq_recall
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report as sk_report,
)

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
    print("🔒 [FINAL EVALUATION] Evaluating on the Locked Official Test Split...\n")

    # 1. Load configuration metadata
    config_path = os.path.join("configs", "joint_bert.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Load frozen data lookups
    with open(os.path.join("data", "processed", "intent_to_id.json"), "r", encoding="utf-8") as f:
        intent_to_id = json.load(f)
    with open(os.path.join("data", "processed", "slot_to_id.json"), "r", encoding="utf-8") as f:
        slot_to_id = json.load(f)

    # Build reverse lookups for readable reports
    id_to_intent = {v: k for k, v in intent_to_id.items()}
    id_to_slot = {v: k for k, v in slot_to_id.items()}

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

    checkpoint_path = "checkpoints/best_joint_nlu_model.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[CRITICAL] Best validation checkpoint missing at: {checkpoint_path}. Train first."
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 7. Metric collection containers
    all_intent_true = []
    all_intent_pred = []

    # For seqeval: list of lists of BIO strings per sequence
    all_slot_true_seqs = []
    all_slot_pred_seqs = []

    # For token-level accuracy baseline
    correct_slot_tokens = 0
    total_valid_slot_tokens = 0

    print(f"⏳ Running inference over {len(test_dataset)} unseen test instances...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intent_label = batch["intent_label"].to(device)

            intent_logits, slot_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # --- Intent predictions ---
            intent_preds = torch.argmax(intent_logits, dim=-1)
            all_intent_true.extend(intent_label.cpu().tolist())
            all_intent_pred.extend(intent_preds.cpu().tolist())

            # --- Slot predictions ---
            slot_preds = torch.argmax(slot_logits, dim=-1)

            # Move to CPU for per-sequence decoding
            slot_labels_cpu = slot_labels.cpu().tolist()
            slot_preds_cpu = slot_preds.cpu().tolist()

            for true_seq, pred_seq in zip(slot_labels_cpu, slot_preds_cpu):
                true_bio = []
                pred_bio = []
                for t, p in zip(true_seq, pred_seq):
                    if t == -100:  # ignore padding / special tokens
                        continue
                    true_bio.append(id_to_slot.get(t, "O"))
                    pred_bio.append(id_to_slot.get(p, "O"))

                if true_bio:  # skip empty sequences
                    all_slot_true_seqs.append(true_bio)
                    all_slot_pred_seqs.append(pred_bio)

            # Token-level accuracy (baseline)
            flat_labels = slot_labels.view(-1)
            flat_preds = slot_preds.view(-1)
            valid_mask = flat_labels != -100
            correct_slot_tokens += ((flat_preds == flat_labels) & valid_mask).sum().item()
            total_valid_slot_tokens += valid_mask.sum().item()

    # ================= INTENT METRICS =================
    intent_acc = accuracy_score(all_intent_true, all_intent_pred)
    intent_p, intent_r, intent_f1, _ = precision_recall_fscore_support(
        all_intent_true, all_intent_pred, average="weighted", zero_division=0
    )

    # ================= SLOT METRICS =================
    slot_token_acc = correct_slot_tokens / total_valid_slot_tokens
    slot_p = seq_precision(all_slot_true_seqs, all_slot_pred_seqs, zero_division=0)
    slot_r = seq_recall(all_slot_true_seqs, all_slot_pred_seqs, zero_division=0)
    slot_f1 = seq_f1(all_slot_true_seqs, all_slot_pred_seqs, zero_division=0)

    # ================= PRINT REPORTS =================
    print("\n" + "=" * 70)
    print("📊 FINAL LOCKED TEST METRICS — INTENT")
    print("=" * 70)
    print(f"  Accuracy  : {intent_acc * 100:.2f}%")
    print(f"  Precision : {intent_p * 100:.2f}%  (weighted)")
    print(f"  Recall    : {intent_r * 100:.2f}%  (weighted)")
    print(f"  F1        : {intent_f1 * 100:.2f}%  (weighted)")

    print("\n" + "=" * 70)
    print("🏷️  FINAL LOCKED TEST METRICS — SLOT (entity-level)")
    print("=" * 70)
    print(f"  Precision : {slot_p * 100:.2f}%")
    print(f"  Recall    : {slot_r * 100:.2f}%")
    print(f"  F1        : {slot_f1 * 100:.2f}%")
    print(f"  ── Token-level accuracy (baseline): {slot_token_acc * 100:.2f}%")

    print("\n" + "=" * 70)
    print("📋 DETAILED INTENT CLASSIFICATION REPORT (per-class)")
    print("=" * 70)
    target_names = [id_to_intent[i] for i in sorted(id_to_intent.keys())]
    print(
        sk_report(
            all_intent_true,
            all_intent_pred,
            labels=list(sorted(id_to_intent.keys())),
            target_names=target_names,
            zero_division=0,
            digits=3,
        )
    )

    print("\n" + "=" * 70)
    print("📋 DETAILED SLOT CLASSIFICATION REPORT (per-entity, seqeval)")
    print("=" * 70)
    print(seqeval_report(all_slot_true_seqs, all_slot_pred_seqs, zero_division=0, digits=3))

    print("\n👑 [DONE] Full evaluation complete.")


if __name__ == "__main__":
    run_final_test_evaluation()