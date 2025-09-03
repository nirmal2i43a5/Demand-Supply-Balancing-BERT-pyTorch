import os
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "bert_ner_baseline_v1"))


def fix_subword_tokens(text, ner_results):
    
    aggregated = []
    current_entity =  None
    
    def label_of(r): 
        return r.get("entity_group") or r.get("entity")
    
    for tok in ner_results:
        if "start" not in tok or "end" not in tok: 
            continue
        
        lbl = label_of(tok)
        
        if current_entity and lbl == current_entity["entity"] and tok["start"] == current_entity["end"]:
            current_entity["end"] = tok["end"]
            current_entity["word"] = text[current_entity["start"]:current_entity["end"]]
            current_entity["score"] = max(current_entity["score"], float(tok.get("score", 0.0)))
            
        else:
            if current_entity: 
                aggregated.append(current_entity)
                
            current_entity = {
                "entity": lbl,
                "start": tok["start"],
                "end": tok["end"],
                "score": float(tok.get("score", 0.0)),
                "word": text[tok["start"]:tok["end"]],
            }
    if current_entity: 
        aggregated.append(current_entity)
    
    return aggregated


def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    device = 0 if torch.cuda.is_available() else -1
    ner_pipeline = pipeline("token-classification", 
                            model=model,
                            tokenizer=tokenizer, 
                            device=device, 
                            aggregation_strategy="simple")
    return ner_pipeline



def predict_ner(pipeline_obj, text: str):
    raw = pipeline_obj(text)  # token-level
    cleaned_ner = fix_subword_tokens(text, raw)
    return cleaned_ner 
