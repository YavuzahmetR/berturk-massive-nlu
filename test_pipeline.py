from transformers import BertTokenizerFast

from src.modeling.joint_nlu_model import (
    JointTurkishNLUModel,
    MassiveAnnotationParser,
    TokenAlignmentEncoder,
)
from src.data.dataset import (
    MassiveJointNLUDataset,
    joint_nlu_collate_fn,
)
from src.modeling.losses import JointNLULoss


MODEL_NAME = "dbmdz/bert-base-turkish-cased"


def run_preflight_test():
    print("🚀 [PREFLIGHT] Robust Turkish Joint NLU test simulasyonu basliyor...\n")

    # ---------------------------------------------------------
    # 1. TEST LABEL HARITALARI
    # ---------------------------------------------------------
    intent_to_id = {
        "alarm_set": 0,
        "weather_query": 1,
    }

    slot_to_id = {
        "O": 0,
        "B-time": 1,
        "I-time": 2,
    }

    # ---------------------------------------------------------
    # 2. TOKENIZER
    # ---------------------------------------------------------
    print(f"⏳ BERTurk Tokenizer yukleniyor ({MODEL_NAME})...")

    tokenizer = BertTokenizerFast.from_pretrained(
        MODEL_NAME
    )

    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"is_fast: {tokenizer.is_fast}")

    # Offset mapping desteğini daha dataset'e geçmeden kontrol et.
    tokenizer_test = tokenizer(
        "yarın sabah yediye alarm kur",
        return_offsets_mapping=True,
    )

    if "offset_mapping" not in tokenizer_test:
        raise RuntimeError(
            "Tokenizer offset_mapping uretemiyor. "
            "Dataset pipeline'ina devam edilmeyecek."
        )

    print("✅ offset_mapping basariyla uretiliyor.")

    # ---------------------------------------------------------
    # 3. PARSER + TOKEN ALIGNMENT
    # ---------------------------------------------------------
    parser = MassiveAnnotationParser()

    encoder = TokenAlignmentEncoder(
        slot_to_id=slot_to_id
    )

    # ---------------------------------------------------------
    # 4. JOINT MODEL
    # ---------------------------------------------------------
    print("⏳ Çift Başlı Joint Model hazirlaniyor...")

    model = JointTurkishNLUModel(
        model_name=MODEL_NAME,
        num_intents=len(intent_to_id),
        num_slots=len(slot_to_id),
    )

    # ---------------------------------------------------------
    # 5. KÜÇÜK TEST VERİSİ
    # ---------------------------------------------------------
    raw_examples = [
        {
            "annot_utt": "yarın [time : sabah yediye] alarm kur",
            "intent": "alarm_set",
        }
    ]

    # ---------------------------------------------------------
    # 6. DATASET
    # ---------------------------------------------------------
    dataset = MassiveJointNLUDataset(
        raw_examples=raw_examples,
        parser=parser,
        encoder=encoder,
        tokenizer=tokenizer,
        intent_to_id=intent_to_id,
        max_length=16,
    )

    sample = dataset[0]

    print("\n🔍 --- DATASET INPUT CHECK ---")

    print(
        'Temizlenmis Girdi: "yarın sabah yediye alarm kur"'
    )

    print(
        f"Token ID'leri (input_ids): "
        f"{sample['input_ids']}"
    )

    print(
        f"Hizalanmis BIO Etiketleri (slot_labels): "
        f"{sample['slot_labels']}"
    )

    print(
        f"Hedef Niyet ID (intent_label): "
        f"{sample['intent_label']}"
    )

    # ---------------------------------------------------------
    # 7. COLLATE → BATCH OLUŞTUR
    # ---------------------------------------------------------
    batch = joint_nlu_collate_fn([sample])

    print("\n📦 --- BATCH SHAPE CHECK ---")

    print(
        f"input_ids shape: "
        f"{list(batch['input_ids'].shape)}"
    )

    print(
        f"attention_mask shape: "
        f"{list(batch['attention_mask'].shape)}"
    )

    print(
        f"slot_labels shape: "
        f"{list(batch['slot_labels'].shape)}"
    )

    print(
        f"intent_label shape: "
        f"{list(batch['intent_label'].shape)}"
    )

    # ---------------------------------------------------------
    # 8. MODEL FORWARD PASS
    # ---------------------------------------------------------
    print("\n🤖 --- MODEL FORWARD PASS CHECK ---")

    model.eval()

    with torch.no_grad():
        intent_logits, slot_logits = model(
            batch["input_ids"],
            batch["attention_mask"],
        )

    print(
        f"Intent Logits Shape "
        f"(Beklenen: [1, {len(intent_to_id)}]): "
        f"{list(intent_logits.shape)}"
    )

    print(
        f"Slot Logits Shape "
        f"(Beklenen: [1, 16, {len(slot_to_id)}]): "
        f"{list(slot_logits.shape)}"
    )

    # ---------------------------------------------------------
    # 9. LOSS ENGINE
    # ---------------------------------------------------------
    loss_fn = JointNLULoss(
        slot_pad_idx=-100,
        lambda_slot=1.0,
    )

    total_loss, intent_loss, slot_loss = loss_fn(
        intent_logits=intent_logits,
        intent_labels=batch["intent_label"],
        slot_logits=slot_logits,
        slot_labels=batch["slot_labels"],
    )

    print("\n📉 --- LOSS ENGINE CHECK ---")

    print(
        f"Hesaplanan Toplam Hata (Total Loss): "
        f"{total_loss.item():.4f}"
    )

    print(
        f"-> Niyet Kaybi (Intent Loss): "
        f"{intent_loss.item():.4f}"
    )

    print(
        f"-> Slot Kaybi (Slot Loss): "
        f"{slot_loss.item():.4f}"
    )

    print(
        "\n✅ [SUCCESS] "
        "Preflight pipeline basariyla tamamlandi!"
    )


if __name__ == "__main__":
    import torch

    run_preflight_test()