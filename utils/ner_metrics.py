
import os
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

from evaluate import load as load_metric

label_list = ['O', 'B-CHEMICAL', 'I-CHEMICAL', 'B-DISEASE', 'I-DISEASE']

id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in enumerate(label_list)}


seqeval = load_metric("seqeval")

def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Remove ignored indices (-100) & convert to label strings
    true_labels, true_preds = [], []
    
    for pred, lab in zip(preds, labels):
        cur_true_labels, cur_true_preds = [], []
        
        for p, l in zip(pred, lab):
            if l == -100:
                continue
            
            cur_true_labels.append(id2label[l])
            cur_true_preds.append(id2label[p])
        true_labels.append(cur_true_labels)
        true_preds.append(cur_true_preds)

    results = seqeval.compute(predictions=true_preds, references=true_labels)
    
     # Aggregate main metrics
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }