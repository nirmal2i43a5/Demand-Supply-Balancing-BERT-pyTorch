# app/ner_pipeline.py
import os
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

from utils.chunked_ner_utils import load_pipeline ,chunked_ner

MODEL_PATH = os.getenv("MODEL_PATH", "outputs/models/biobert_ner_baseline_v1")

# def load_pipeline(
#     model_path: str | None = None,
#     aggregation_strategy: str = "simple",
#     device: int | None = None,
# ):
    
#     """Short-text pipeline (no windowing)."""
    
#     model_path = model_path or MODEL_PATH
#     tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#     mdl = AutoModelForTokenClassification.from_pretrained(model_path)
#     if device is None:
#         device = 0 if torch.cuda.is_available() else -1
#     return pipeline(
#         task="token-classification",
#         model=mdl,
#         tokenizer=tok,
#         device=device,
#         aggregation_strategy=aggregation_strategy,
#     )

def predict_ner(pipeline_obj, text: str) -> List[Dict[str, Any]]:
    raw = pipeline_obj(text)
    # Already aggregated; return as-is for short texts
    return [
        {
            "entity": r.get("entity_group") or r.get("entity"),
            "score": float(r.get("score", 0.0)),
            "word": r.get("word"),
            "start": int(r["start"]),
            "end": int(r["end"]),
        }
        for r in raw
        if "start" in r and "end" in r
    ]

def load_chunked_pipeline(
    model_path: str | None = None,
    aggregation_strategy: str = "simple",
    device: int | None = None,
):
    """Long-text capable pipeline (same model; helper wrapper)."""
    return load_pipeline(
        model_name_or_path=MODEL_PATH,
        aggregation_strategy=aggregation_strategy,
        device=device,
    )

def predict_ner(
    pipeline_obj,
    text: str,
    max_tokens: int = 512,
    stride_tokens: int = 128,
):
    return chunked_ner(
        ner_pipe=pipeline_obj,
        text=text,
        max_tokens=max_tokens,
        stride_tokens=stride_tokens,
    )
