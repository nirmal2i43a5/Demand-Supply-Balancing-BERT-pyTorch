from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch


def merge_subwords(ents, text):
    
#     Merge subword pieces when they clearly belong together(like same label, touching spans)
#     and either a WordPiece continuation (##...) or only space/hyphen in between. For instance, "anti" + "##biotic" -> "antibiotic"
    
    if not ents:
        return []

    ents = sorted(ents, key=lambda d: (d["start"], d["end"]))
    out = []

    def _trivial_gap(a_end, b_start):
        return text[a_end:b_start].strip(" \t\r\n-–—") == ""

    for e in ents:
        item = dict(e)  # avoid mutating caller data
        raw = item.get("word") or ""
        is_wp_cont = raw.startswith("##")

        if out:
            last = out[-1]
            same_label = item["entity"] == last["entity"]
            touching = item["start"] <= last["end"] + 1
            continuation = is_wp_cont or _trivial_gap(last["end"], item["start"])

            if same_label and touching and continuation:
                last["end"] = max(last["end"], item["end"])
                last["score"] = max(last.get("score", 0.0), item.get("score", 0.0))
                last["word"] = text[last["start"]: last["end"]]
                continue

        item["word"] = text[item["start"]: item["end"]]
        out.append(item)

    return out



def merge_touching(ents):
    
    """
    1. Merge adjacent spans with the same label.
    For instance, "New York" as could show sometimes as two tokens like (New and Work) separately
    both labeled as LOC(for location). 
    """
    
    if not ents:
        return []

    ents = sorted(ents, key=lambda d: (d["start"], d["end"]))
    merged = [dict(ents[0])]

    for e in ents[1:]:
        cur = dict(e)
        last = merged[-1]

        if cur["entity"] == last["entity"] and cur["start"] <= last["end"] + 1:
            
            last["end"] = max(last["end"], cur["end"])
            last["score"] = max(float(last.get("score", 0.0)), float(cur.get("score", 0.0)))
            # print("Merging", last, "WITH", cur)
            if len(cur.get("word") or "") > len(last.get("word") or ""):
                last["word"] = cur.get("word")
        else:
            merged.append(cur)

    return merged


def _windows(total, size, stride):
    
    
    # Token-index windows for sliding inference: [i, j) with overlap of `stride`.
    
    if size <= 0:
        raise ValueError("size must be > 0")
    
    if stride >= size:
        raise ValueError("stride must be < size")

    spans = []
    i = 0
    while i < total:
        j = min(i + size, total)
        
        spans.append((i, j))
        if j == total:
            break
        i = j - stride
        
    return spans



def load_pipeline(model_name_or_path, aggregation_strategy="simple", device=None):
  
    
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    mdl = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
    
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
        
    return pipeline(
        task="ner",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy=aggregation_strategy,
        device=device,
    )


def chunked_ner(ner_pipe, text, max_tokens=512, stride_tokens=128):
    
    # Run NER over  long text by sliding a token window and Returns entity spans with offsets into the ORIGINAL text.

    if not text:
        return []

    tok = ner_pipe.tokenizer

    # effective per-window token budget after specials (usually [CLS] + [SEP])
    specials = tok.num_special_tokens_to_add(pair=False)  # typically 2
    eff_window = max_tokens - specials
    if eff_window <= 0:
        raise ValueError(
            f"max_tokens={max_tokens} too small for specials={specials}. Increase max_tokens."
        )
        
    if stride_tokens >= eff_window:
        raise ValueError("stride_tokens must be < (max_tokens - specials)")

    # map every token to its (start, end) char offsets in the ORIGINAL text
    enc = tok(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=False,
    )
    offsets = enc["offset_mapping"]
    

    spans = _windows(len(offsets), size=eff_window, stride=stride_tokens)

    all_ents = []
    for t0, t1 in spans:
        # convert token window to character slice
        win_start = offsets[t0][0]
        win_end = offsets[t1 - 1][1]
        chunk = text[win_start:win_end]


        # pipeline already returns aggregated entities (by strategy)
        for r in ner_pipe(chunk):
            start = r.get("start")
            end = r.get("end")
            
            if start is None or end is None:
                continue
            
            label = r.get("entity_group") or r.get("entity")
            all_ents.append(
                {
                    "entity": label,
                    "score": float(r.get("score", 0.0)),
                    "word": r.get("word"),
                    "start": int(start) + win_start,
                    "end": int(end) + win_start,
                }
            )

    # keep best duplicate span by score, then merge neighbors, then clean subwords
    best = {}
    for e in all_ents:
        key = (e["start"], e["end"], e["entity"])
        if key not in best or e["score"] > best[key]["score"]:
            best[key] = e

    merged = merge_touching(list(best.values()))
    
    merged.sort(key=lambda d: d["start"])
    
    return merge_subwords(merged, text) # return after merging subwords for subtokens 
