import os
import argparse
import json
import copy
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from src.modeling.joint_nlu_model import (
    JointTurkishNLUModel,
    MassiveAnnotationParser,
    TokenAlignmentEncoder,
)
from src.data.dataset import MassiveJointNLUDataset, joint_nlu_collate_fn
from src.modeling.losses import JointNLULoss


CHECKPOINT_PATH = "checkpoints/best_joint_nlu_model.pt"


def load_local_jsonl(file_path: str) -> list:
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def compute_class_weights(dataset, num_classes: int, key: str, pad_idx: int = None, cap: float = None):
    """Eğitim verisinden sınıf frekanslarına göre ağırlık hesaplar."""
    counts = torch.zeros(num_classes, dtype=torch.float)
    for i in range(len(dataset)):
        item = dataset[i]
        labels = item[key] if isinstance(item[key], list) else [item[key]]
        for lbl in labels:
            if pad_idx is not None and lbl == pad_idx:
                continue
            counts[lbl] += 1

    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    if cap is not None:
        weights = torch.clamp(weights, max=cap)
    return weights


def main():
    print("[INFO] Turkish Joint NLU Training — Optimized\n")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/joint_bert.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    print(f"[INFO] Turkish Joint NLU Training — Config: {args.config}\n")

    # 1. Config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["training"]["seed"])

    # 2. Label mappings
    with open(os.path.join("data", "processed", "intent_to_id.json"), "r", encoding="utf-8") as f:
        intent_to_id = json.load(f)
    with open(os.path.join("data", "processed", "slot_to_id.json"), "r", encoding="utf-8") as f:
        slot_to_id = json.load(f)

    # 3. Raw data
    train_raw = load_local_jsonl(os.path.join("data", "raw", "train.jsonl"))
    val_raw = load_local_jsonl(os.path.join("data", "raw", "validation.jsonl"))

    # 4. Core workers
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    parser = MassiveAnnotationParser()
    encoder = TokenAlignmentEncoder(slot_to_id=slot_to_id)

    train_dataset = MassiveJointNLUDataset(
        raw_examples=train_raw, parser=parser, encoder=encoder,
        tokenizer=tokenizer, intent_to_id=intent_to_id,
        max_length=cfg["model"]["max_length"],
    )
    val_dataset = MassiveJointNLUDataset(
        raw_examples=val_raw, parser=parser, encoder=encoder,
        tokenizer=tokenizer, intent_to_id=intent_to_id,
        max_length=cfg["model"]["max_length"],
    )

    # 5. DataLoaders (optimize edilmiş)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=joint_nlu_collate_fn,
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=cfg["training"].get("pin_memory", False),
        persistent_workers=cfg["training"].get("num_workers", 0) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=joint_nlu_collate_fn,
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=cfg["training"].get("pin_memory", False),
        persistent_workers=cfg["training"].get("num_workers", 0) > 0,
    )

    # 6. Device + Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Computing Hardware Engine Target: {device}")

    model = JointTurkishNLUModel(
        model_name=cfg["model"]["model_name"],
        num_intents=len(intent_to_id),
        num_slots=len(slot_to_id),
        dropout_rate=cfg["model"]["dropout_rate"],
    ).to(device)

    # 7. Class weights (opsiyonel)
    intent_weights = None
    slot_weights = None
    cw_mode = cfg["loss"].get("use_class_weights", False)
    cap = cfg["loss"].get("weight_cap", None)

    # "slot" -> sadece slot weightleri
    # True / "both" -> ikisi de
    # False / "none" -> hiçbiri
    use_intent_cw = cw_mode in (True, "both", "intent")
    use_slot_cw = cw_mode in (True, "both", "slot")

    if use_intent_cw or use_slot_cw:
        print(f"[INFO] Class weights mode: {cw_mode}")

    if use_intent_cw:
        intent_weights = compute_class_weights(
            train_dataset, len(intent_to_id), "intent_label", pad_idx=None, cap=cap
        ).to(device)
        print(f"[INFO] Intent weight min/max: {intent_weights.min():.2f}/{intent_weights.max():.2f}")

    if use_slot_cw:
        slot_weights = compute_class_weights(
            train_dataset, len(slot_to_id), "slot_labels",
            pad_idx=cfg["loss"]["slot_pad_idx"], cap=cap,
        ).to(device)
        print(f"[INFO] Slot   weight min/max: {slot_weights.min():.2f}/{slot_weights.max():.2f}")

    # 8. Loss + Optimizer + Scheduler
    loss_fn = JointNLULoss(
        slot_pad_idx=cfg["loss"]["slot_pad_idx"],
        lambda_slot=cfg["loss"]["lambda_slot"],
        intent_weights=intent_weights,
        slot_weights=slot_weights,
        label_smoothing=cfg["loss"].get("label_smoothing", 0.0),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg["training"]["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    # 9. AMP (Mixed Precision)
    use_amp = cfg["training"].get("use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[INFO] Mixed precision (AMP) enabled.")
    else:
        print("[INFO] AMP disabled (CPU or config off).")

    # 10. Early stopping state
    best_val_loss = float("inf")
    patience = cfg["training"].get("early_stopping_patience", None)
    epochs_no_improve = 0
    best_state = None
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    # 11. Training loop
    for epoch in range(cfg["training"]["epochs"]):
        # ----- TRAIN -----
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            slot_labels = batch["slot_labels"].to(device, non_blocking=True)
            intent_label = batch["intent_label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                intent_logits, slot_logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                loss, _, _ = loss_fn(
                    intent_logits=intent_logits, intent_labels=intent_label,
                    slot_logits=slot_logits, slot_labels=slot_labels,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg["training"]["max_grad_norm"]
            )
            # AMP: gradient overflow olduysa optimizer.step() atlanır.
            # Bu durumda scheduler.step()'i de atlamalıyız (warning'i önler).
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()

            if scale_after >= scale_before:
                scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"[TRAIN] Epoch [{epoch+1}/{cfg['training']['epochs']}] -> Train Loss: {avg_train_loss:.4f}")

        # ----- VALIDATION -----
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                slot_labels = batch["slot_labels"].to(device, non_blocking=True)
                intent_label = batch["intent_label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    intent_logits, slot_logits = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    loss, _, _ = loss_fn(
                        intent_logits=intent_logits, intent_labels=intent_label,
                        slot_logits=slot_logits, slot_labels=slot_labels,
                    )
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"[EVAL]  Epoch [{epoch+1}/{cfg['training']['epochs']}] -> Val Loss:   {avg_val_loss:.4f}")

        # ----- CHECKPOINT + EARLY STOPPING -----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, CHECKPOINT_PATH)
            print(f"[CHECKPOINT] Improved -> {CHECKPOINT_PATH}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] No improvement ({epochs_no_improve}/{patience})")

        if patience is not None and epochs_no_improve >= patience:
            print(f"[EARLY-STOP] No improvement for {patience} epochs. Stopping.")
            break

        print("-" * 60)

    print(f"\n[SUCCESS] Best validation loss: {best_val_loss:.4f}")
    print(f"[SUCCESS] Best model saved at: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()