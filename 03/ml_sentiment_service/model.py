from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file as safe_load_file

_tokenizer = None
_model = None
_id2label: Dict[int, str] = {}

MODEL_DIR = os.getenv("MODEL_DIR", "./model")
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", os.path.join(MODEL_DIR, "model.safetensors"))

def _load_model() -> Tuple[Any, Any, Dict[int, str]]:
    global _tokenizer, _model, _id2label
    if _tokenizer and _model:
        return _tokenizer, _model, _id2label

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(config)

    if os.path.isfile(MODEL_WEIGHTS):
        if MODEL_WEIGHTS.endswith(".safetensors"):
            state_dict = safe_load_file(MODEL_WEIGHTS)
        else:
            state_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

    model.eval()
    model.to("cpu")
    id2label = {int(k): v for k, v in getattr(config, "id2label", {}).items()} or {
        i: f"label_{i}" for i in range(model.config.num_labels)
    }
    _tokenizer, _model, _id2label = tokenizer, model, id2label
    return _tokenizer, _model, _id2label

def predict(texts: List[str]) -> List[Dict[str, Any]]:
    tokenizer, model, id2label = _load_model()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**enc)
        probs = F.softmax(outputs.logits, dim=-1)

    # Русский перевод меток
    label_map_ru = {
        "neutral": "Нейтрально",
        "positive": "Положительно",
        "negative": "Отрицательно",
        "0": "Нейтрально",
        "1": "Положительно",
        "2": "Отрицательно",
    }

    results = []
    for text, row in zip(texts, probs.tolist()):
        idx = int(max(range(len(row)), key=lambda i: row[i]))
        label = id2label.get(idx, str(idx)).lower()
        prob = float(row[idx])
        label_ru = label_map_ru.get(label, label)
        results.append({
            "text": text,
            "label": label_ru,
        })
    return results