import torch
from torch.utils.data import DataLoader
from src.data.dataset import joint_nlu_collate_fn

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True) -> DataLoader:
    """
    DataLoader'ı yüksek performanslı ve güvenli ayarlarla oluşturur.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=joint_nlu_collate_fn
    )

def train_one_epoch(
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: getattr, 
        loss_fn: torch.nn.Module,
        device: torch.device
) -> float:
    """
    Executes a single optimization epoch across the dataset pipeline.
    """
    model.train()
    total_epoch_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        slot_labels = batch["slot_labels"].to(device, non_blocking=True)
        intent_label = batch["intent_label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        intent_logits, slot_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        loss, _, _ = loss_fn(
            intent_logits=intent_logits,
            intent_labels=intent_label,
            slot_logits=slot_logits,
            slot_labels=slot_labels
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_epoch_loss += loss.item()

    return total_epoch_loss / len(dataloader)