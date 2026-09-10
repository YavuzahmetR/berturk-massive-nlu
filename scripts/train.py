import os
import json
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from src.modeling.joint_nlu_model import JointTurkishNLUModel, MassiveAnnotationParser, TokenAlignmentEncoder
from src.data.dataset import MassiveJointNLUDataset, joint_nlu_collate_fn
from src.modeling.losses import JointNLULoss

def load_local_jsonl(file_path: str) -> list:
    """Reads processed JSON Lines from local cache securely."""
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def main():
    print("[INFO] Turkish Joint NLU Training Loop Initiated.\n")

    # 1. Load configuration contract from YAML
    config_path = os.path.join("configs", "joint_bert.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Set deterministic random seed anchor for reproducibility
    torch.manual_seed(cfg["training"]["seed"])

    # 2. Load locked canonical lookups artifacts
    with open(os.path.join("data", "processed", "intent_to_id.json"), "r", encoding="utf-8") as f:
        intent_to_id = json.load(f)
    with open(os.path.join("data", "processed", "slot_to_id.json"), "r", encoding="utf-8") as f:
        slot_to_id = json.load(f)

    # 3. Load raw independent dataset split lines cache
    train_raw = load_local_jsonl(os.path.join("data", "raw", "train.jsonl"))
    val_raw = load_local_jsonl(os.path.join("data", "raw", "validation.jsonl"))

    # 4. Instantiate pipeline core workers
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    parser = MassiveAnnotationParser()
    encoder = TokenAlignmentEncoder(slot_to_id=slot_to_id)

    # 5. Build strict PyTorch training and validation partitions
    train_dataset = MassiveJointNLUDataset(
        raw_examples=train_raw, parser=parser, encoder=encoder,
        tokenizer=tokenizer, intent_to_id=intent_to_id, max_length=cfg["model"]["max_length"]
    )
    val_dataset = MassiveJointNLUDataset(
        raw_examples=val_raw, parser=parser, encoder=encoder,
        tokenizer=tokenizer, intent_to_id=intent_to_id, max_length=cfg["model"]["max_length"]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], 
        shuffle=True, collate_fn=joint_nlu_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["training"]["batch_size"], 
        shuffle=False, collate_fn=joint_nlu_collate_fn
    )

    # 6. Initialize Joint Neural Model and Hardware allocation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Computing Hardware Engine Target: {device}")

    model = JointTurkishNLUModel(
        model_name=cfg["model"]["model_name"],
        num_intents=len(intent_to_id),
        num_slots=len(slot_to_id),
        dropout_rate=cfg["model"]["dropout_rate"]
    ).to(device)

    # 7. Optimizer, Loss Engine and Linear Warmup Scheduling setup
    loss_fn = JointNLULoss(slot_pad_idx=cfg["loss"]["slot_pad_idx"], lambda_slot=cfg["loss"]["lambda_slot"])
    optimizer = AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=cfg["training"]["weight_decay"])
    
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * cfg["training"]["warmup_ratio"]), 
        num_training_steps=total_steps
    )

    # Keep track of the best validation loss for checkpointing
    best_val_loss = float("inf")
    model_save_path = "checkpoints/best_joint_nlu_model.pt"

    # 8. Optimization Execution Loop (Epoch blocks)
    for epoch in range(cfg["training"]["epochs"]):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intent_label = batch["intent_label"].to(device)

            optimizer.zero_grad()
            intent_logits, slot_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss, intent_loss, slot_loss = loss_fn(
                intent_logits=intent_logits, intent_labels=intent_label,
                slot_logits=slot_logits, slot_labels=slot_labels
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["training"]["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"[TRAIN] Epoch [{epoch+1}/{cfg['training']['epochs']}] -> Average Multi-Task Train Loss: {avg_train_loss:.4f}")

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                slot_labels = batch["slot_labels"].to(device)
                intent_label = batch["intent_label"].to(device)

                intent_logits, slot_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                
                loss, _, _ = loss_fn(
                    intent_logits=intent_logits, intent_labels=intent_label,
                    slot_logits=slot_logits, slot_labels=slot_labels
                )
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"[EVAL] Epoch [{epoch+1}/{cfg['training']['epochs']}] -> Average Validation Loss: {avg_val_loss:.4f}")

        # --- CHECKPOINT LOGIC ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"[CHECKPOINT] Validation loss improved. Model saved to {model_save_path}")
        
        print("-" * 50)

    print("\n[SUCCESS] Training phase successfully executed. Optimization process completed.")

if __name__ == "__main__":
    main()
