import os
import json
import copy
import yaml
import optuna
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score as sk_f1
from seqeval.metrics import f1_score as seq_f1

from src.modeling.joint_nlu_model import (
    JointTurkishNLUModel,
    MassiveAnnotationParser,
    TokenAlignmentEncoder,
)
from src.data.dataset import MassiveJointNLUDataset, joint_nlu_collate_fn
from src.modeling.losses import JointNLULoss


optuna.logging.set_verbosity(optuna.logging.INFO)


def load_local_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def compute_class_weights_from_counts(counts, cap=None):
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    if cap is not None:
        weights = torch.clamp(weights, max=cap)
    return weights


def precompute_label_counts(dataset, num_intents, num_slots, pad_idx=-100):
    intent_counts = torch.zeros(num_intents, dtype=torch.float)
    slot_counts = torch.zeros(num_slots, dtype=torch.float)
    for i in range(len(dataset)):
        item = dataset[i]
        intent_counts[item["intent_label"]] += 1
        for lbl in item["slot_labels"]:
            if lbl != pad_idx:
                slot_counts[lbl] += 1
    return intent_counts, slot_counts


def evaluate_f1(model, val_loader, id_to_slot, device):
    model.eval()
    all_intent_true, all_intent_pred = [], []
    all_slot_true_seqs, all_slot_pred_seqs = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intent_label = batch["intent_label"].to(device)

            intent_logits, slot_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            all_intent_true.extend(intent_label.cpu().tolist())
            all_intent_pred.extend(torch.argmax(intent_logits, dim=-1).cpu().tolist())

            slot_preds = torch.argmax(slot_logits, dim=-1).cpu().tolist()
            slot_true = slot_labels.cpu().tolist()

            for t_seq, p_seq in zip(slot_true, slot_preds):
                t_bio, p_bio = [], []
                for t, p in zip(t_seq, p_seq):
                    if t == -100:
                        continue
                    t_bio.append(id_to_slot.get(t, "O"))
                    p_bio.append(id_to_slot.get(p, "O"))
                if t_bio:
                    all_slot_true_seqs.append(t_bio)
                    all_slot_pred_seqs.append(p_bio)

    intent_f1 = sk_f1(all_intent_true, all_intent_pred, average="weighted", zero_division=0)
    slot_f1 = seq_f1(all_slot_true_seqs, all_slot_pred_seqs, zero_division=0)
    return intent_f1, slot_f1


def train_one_trial(cfg, trial, train_dataset, val_dataset,
                    id_to_intent, id_to_slot,
                    intent_counts, slot_counts, device):
    # Trial parametreleri (HIZLI ARAMA)
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.05)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)
    cw_mode = trial.suggest_categorical("use_class_weights", ["none", "both"])
    batch_size = 16  # RTX 3050 4GB VRAM icin sabit

    cap = None
    if cw_mode == "both":
        cap = trial.suggest_float("weight_cap", 2.0, 4.0)

    # Windows deadlock'ini onlemek icin num_workers=0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=joint_nlu_collate_fn, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=joint_nlu_collate_fn, num_workers=0, pin_memory=False,
    )

    model = JointTurkishNLUModel(
        model_name=cfg["model"]["model_name"],
        num_intents=len(id_to_intent),
        num_slots=len(id_to_slot),
        dropout_rate=dropout,
    ).to(device)

    intent_weights = None
    slot_weights = None
    if cw_mode == "both":
        intent_weights = compute_class_weights_from_counts(intent_counts, cap).to(device)
        slot_weights = compute_class_weights_from_counts(slot_counts, cap).to(device)

    loss_fn = JointNLULoss(
        slot_pad_idx=-100,
        lambda_slot=cfg["loss"]["lambda_slot"],
        intent_weights=intent_weights,
        slot_weights=slot_weights,
        label_smoothing=label_smoothing,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 4  # 5'ten 4'e dustu, hizlandi
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # AMP KAPALI (RTX 3050'de stabilite icin)
    best_slot_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intent_label = batch["intent_label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            intent_logits, slot_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            loss, _, _ = loss_fn(
                intent_logits=intent_logits, intent_labels=intent_label,
                slot_logits=slot_logits, slot_labels=slot_labels,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        _, slot_f1 = evaluate_f1(model, val_loader, id_to_slot, device)

        trial.report(slot_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if slot_f1 > best_slot_f1:
            best_slot_f1 = slot_f1

        print(f"    [Trial {trial.number}] Epoch {epoch+1}/{epochs} | Slot F1: {slot_f1:.4f}")

    return best_slot_f1


def main():
    with open("configs/joint_bert.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["training"]["seed"])

    with open("data/processed/intent_to_id.json", "r", encoding="utf-8") as f:
        intent_to_id = json.load(f)
    with open("data/processed/slot_to_id.json", "r", encoding="utf-8") as f:
        slot_to_id = json.load(f)

    id_to_intent = {v: k for k, v in intent_to_id.items()}
    id_to_slot = {v: k for k, v in slot_to_id.items()}

    train_raw = load_local_jsonl("data/raw/train.jsonl")
    val_raw = load_local_jsonl("data/raw/validation.jsonl")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    print("[INFO] Precomputing label frequencies (once)...")
    intent_counts, slot_counts = precompute_label_counts(
        train_dataset, len(intent_to_id), len(slot_to_id), pad_idx=-100
    )
    print("[INFO] Label frequencies ready.")

    print(f"[INFO] Starting Optuna search...\n")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="joint_nlu_slot_f1",
    )

    def objective(trial):
        return train_one_trial(
            cfg, trial, train_dataset, val_dataset,
            id_to_intent, id_to_slot,
            intent_counts, slot_counts, device,
        )

    # 10 trial yeterli (hizli bitirelim)
    study.optimize(objective, n_trials=10)

    print("\n" + "=" * 70)
    print("OPTUNA RESULTS")
    print("=" * 70)
    print(f"Best Slot F1: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_cfg = copy.deepcopy(cfg)
    best_cfg["model"]["dropout_rate"] = study.best_params["dropout"]
    best_cfg["training"]["learning_rate"] = study.best_params["lr"]
    best_cfg["training"]["weight_decay"] = study.best_params["weight_decay"]
    best_cfg["training"]["batch_size"] = 16
    best_cfg["loss"]["use_class_weights"] = study.best_params["use_class_weights"]
    best_cfg["loss"]["label_smoothing"] = study.best_params["label_smoothing"]
    if "weight_cap" in study.best_params:
        best_cfg["loss"]["weight_cap"] = study.best_params["weight_cap"]

    with open("configs/joint_bert_best.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\n[SAVED] Best config -> configs/joint_bert_best.yaml")


if __name__ == "__main__":
    main()