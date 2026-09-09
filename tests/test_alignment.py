import pytest
from transformers import BertTokenizerFast

# Import our custom engineering modules safely
from src.modeling.joint_nlu_model import MassiveAnnotationParser, TokenAlignmentEncoder

@pytest.fixture
def shared_setup():
    """
    Provides a fast cased tokenizer and fixed label map for reproducibility.
    """
    tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-cased")
    slot_to_id = {"O": 0, "B-time": 1, "I-time": 2, "B-date": 3, "I-date": 4}
    
    parser = MassiveAnnotationParser()
    encoder = TokenAlignmentEncoder(slot_to_id=slot_to_id)
    
    return tokenizer, parser, encoder

def test_standard_alignment(shared_setup):
    """
    Test standard slot tokenization and proper BIO tag assignment.
    """
    tokenizer, parser, encoder = shared_setup
    annotated_text = "yarın [time : sabah yediye] alarm kur"
    
    # 1. Parse text and fetch exact character positions
    cleaned_text, spans = parser.parse(annotated_text)
    
    # 2. Tokenize and extract offsets mapping matrix
    encoding = tokenizer(cleaned_text, return_offsets_mapping=True)
    offset_mapping = encoding["offset_mapping"]
    
    # 3. Generate aligned label ids
    labels = encoder.align_slots_to_tokens(spans, offset_mapping)
    
    # Validation Invariants
    assert cleaned_text == "yarın sabah yediye alarm kur"
    assert len(spans) == 1
    assert spans[0]["type"] == "time"
    # [CLS] and [SEP] tokens must be completely masked out
    assert labels[0] == -100
    assert labels[-1] == -100

def test_multiple_consecutive_slots(shared_setup):
    """
    Test edge case where multiple slots are structured right next to each other.
    """
    tokenizer, parser, encoder = shared_setup
    annotated_text = "[date : yarın] [time : sabah yedi]"
    
    cleaned_text, spans = parser.parse(annotated_text)
    encoding = tokenizer(cleaned_text, return_offsets_mapping=True)
    offset_mapping = encoding["offset_mapping"]
    labels = encoder.align_slots_to_tokens(spans, offset_mapping)
    
    assert cleaned_text == "yarın sabah yedi"
    assert len(spans) == 2
    assert spans[0]["type"] == "date"
    assert spans[1]["type"] == "time"
    # Ensure no label bleeding or collision happened between consecutive entities
    assert shared_setup[2].slot_to_id["B-date"] in labels
    assert shared_setup[2].slot_to_id["B-time"] in labels

def test_turkish_character_casing_robustness(shared_setup):
    """
    Test strict validation with complex Turkish characters (İ, ı, ş, ğ, ç) and capitalization.
    """
    tokenizer, parser, encoder = shared_setup
    annotated_text = "İSTANBUL'DA [time : İKİNDİ VAKTİ] ŞEVKET'İ ARARIM"
    
    cleaned_text, spans = parser.parse(annotated_text)
    encoding = tokenizer(cleaned_text, return_offsets_mapping=True)
    offset_mapping = encoding["offset_mapping"]
    labels = encoder.align_slots_to_tokens(spans, offset_mapping)
    
    assert "İKİNDİ VAKTİ" in cleaned_text
    assert len(spans) == 1
    # Check that subword overlapping parser functions properly under strict Turkish casing
    assert shared_setup[2].slot_to_id["B-time"] in labels
