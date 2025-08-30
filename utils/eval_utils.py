import polars as pl
import torch
import contextlib

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import yaml
import numpy as np
import torch.nn.functional as F
from contextlib import nullcontext
from tqdm.auto import tqdm

def get_labels(df):
    all_concepts_eval = df['id'].unique().to_list()
    id_2_idx = dict(zip(all_concepts_eval, np.arange(len(all_concepts_eval))))
    df_id_map = pl.DataFrame({
        "id": list(id_2_idx.keys()),
        "label": list(id_2_idx.values())
    })
    return df.join(df_id_map, on="id")

def get_labels_from_list(list_nodes, mode = "dict"):
    id_2_idx = dict(zip(list_nodes, np.arange(len(list_nodes))))
    if mode == "dict":
        return id_2_idx
    else: 
        df_id_map = pl.DataFrame({
            "id": list(id_2_idx.keys()),
            "label": list(id_2_idx.values())
        })
        return df_id_map
    
@torch.no_grad()
def evaluate_batched_tqdm(
    model,
    anchors,
    positives,
    labels,
    device,
    batch_size=2048,
    max_k=10,
    head="q",                 # "q" for quantized, "emb" for raw encoder
    dtype="fp16",             # or "fp32"
):
    assert len(anchors) == len(positives), "anchors and positives must be same length"
    N = len(anchors)

    model.eval().to(device)

    def encode(texts):
        out = model(texts)                    # dict with "emb"/"q"
        vec = out[head] if isinstance(out, dict) else out
        return F.normalize(vec, p=2, dim=-1)  # ensure unit-norm for cosine via dot

    use_amp = (device == "cuda" and dtype == "fp16")
    amp_ctx = torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()

    # 1) encode all positives once
    pos_emb_chunks = []
    with amp_ctx:
        for i in tqdm(range(0, N, batch_size), desc="Encoding positives", leave=False):
            vec = encode(positives[i:i+batch_size])
            pos_emb_chunks.append(vec)
    positives_emb = torch.cat(pos_emb_chunks, dim=0).to(device, non_blocking=True).contiguous()  # [N, D]

    
    ranks_all = []
    mrr_sum = 0.0

    pbar = tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc="Evaluating")
    for start in pbar:
        end = min(start + batch_size, N)
        batch_anchors = anchors[start:end]
        batch_size_actual = end - start
        
        with amp_ctx:
            A = encode(batch_anchors)  # [B, D]
            B = positives_emb          # [N, D]

        # Compute similarity scores and sort in descending order
        scores = A @ B.T  # [B, N]
        sorted_indices = torch.argsort(scores, descending=True)  # [B, N]
        
        # Much faster vectorized approach
        batch_labels = torch.tensor(labels[start:end], device=device)  # [B]
        all_labels = torch.tensor(labels, device=device)  # [N]
        
        # Create mask for matching labels and apply to sorted indices
        label_matches = all_labels.unsqueeze(0) == batch_labels.unsqueeze(1)  # [B, N]
        label_matches_sorted = label_matches.gather(1, sorted_indices)  # [B, N]
        
        # Find first True in each row (first occurrence of target label)
        batch_ranks = torch.argmax(label_matches_sorted.int(), dim=1)  # [B]

        ranks_all.append(batch_ranks)
        done = end
        mrr_sum += (1.0 / (batch_ranks.float() + 1.0)).sum().item()
        pbar.set_postfix({"MRR": f"{mrr_sum / done:.3f}", "done": done})

    ranks = torch.cat(ranks_all, dim=0)
    topk = {f"top@{k}": (ranks < k).float().mean().item() for k in range(1, max_k+1)}
    mrr = (1.0 / (ranks.float() + 1.0)).mean().item()
    return ranks, topk, mrr


@torch.no_grad()
def evaluate_batched_tqdm_low_memory(
    model,
    anchors,
    positives,
    labels,
    device,
    batch_size=2048,
    max_k=10,
    head="q",                  # "q" for quantized, "emb" for raw encoder
    dtype="fp16",              # or "fp32"
    pos_chunk_size=50000       # tune 10k–100k based on VRAM
):
    assert len(anchors) == len(positives) == len(labels), "anchors/positives/labels must align"
    N = len(anchors)

    model.eval().to(device)

    def encode(texts):
        out = model(texts)                           # dict with "emb"/"q"
        vec = out[head] if isinstance(out, dict) else out
        return F.normalize(vec, p=2, dim=-1)         # L2 normalize for cosine

    use_amp = (device == "cuda" and dtype == "fp16")
    amp_ctx = torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()

    # 1) Encode all positives once -> CPU (fp16 to save RAM)
    pos_emb_cpu = []
    with amp_ctx:
        for i in tqdm(range(0, N, batch_size), desc="Encoding positives", leave=False):
            vec = encode(positives[i:i+batch_size]).to(device, non_blocking=True)
            pos_emb_cpu.append(vec.to(torch.float16 if use_amp else vec.dtype).cpu())
            del vec
            if device == "cuda":
                torch.cuda.empty_cache()
    pos_emb_cpu = torch.cat(pos_emb_cpu, dim=0)            # [N, D] on CPU
    labels_cpu = torch.tensor(labels, dtype=torch.long).cpu()

    ranks_all = []
    mrr_sum = 0.0

    pbar = tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc="Evaluating")
    for start in pbar:
        end = min(start + batch_size, N)
        a_texts = anchors[start:end]
        batch_labels = torch.tensor(labels[start:end], dtype=torch.long, device=device)  # [B]

        # Encode anchor batch
        with amp_ctx:
            A = encode(a_texts).to(device, non_blocking=True)                           # [B, D]

        B = A.size(0)

        # ---- Pass 1: get best score among positives with the same label (no sort) ----
        best_same = torch.full((B,), float("-inf"), device=device)

        for c0 in range(0, N, pos_chunk_size):
            c1 = min(c0 + pos_chunk_size, N)
            P_chunk = pos_emb_cpu[c0:c1].to(device, non_blocking=True)                  # [C, D]
            L_chunk = labels_cpu[c0:c1].to(device, non_blocking=True)                   # [C]

            with amp_ctx:
                S = A @ P_chunk.T                                                       # [B, C]

            mask_same = (L_chunk.unsqueeze(0) == batch_labels.unsqueeze(1))             # [B, C]
            S_masked = S.masked_fill(~mask_same, float("-inf"))
            best_same = torch.maximum(best_same, S_masked.max(dim=1).values)

            del P_chunk, L_chunk, S, S_masked, mask_same
            if device == "cuda":
                torch.cuda.empty_cache()

        # Safety: if some anchor had no matching label in positives (shouldn't happen)
        missing = torch.isinf(best_same)
        if missing.any():
            # set to very low so rank counts all; you may also choose to skip these
            best_same[missing] = -1e9

        # ---- Pass 2: count how many candidates beat the best same-label score ----
        count_greater = torch.zeros(B, dtype=torch.int64, device=device)

        for c0 in range(0, N, pos_chunk_size):
            c1 = min(c0 + pos_chunk_size, N)
            P_chunk = pos_emb_cpu[c0:c1].to(device, non_blocking=True)                  # [C, D]

            with amp_ctx:
                S = A @ P_chunk.T                                                       # [B, C]

            count_greater += (S > best_same.unsqueeze(1)).sum(dim=1)

            del P_chunk, S
            if device == "cuda":
                torch.cuda.empty_cache()

        batch_ranks = count_greater                                                      # [B], 0 = top-1
        ranks_all.append(batch_ranks.cpu())

        done = end
        mrr_sum += (1.0 / (batch_ranks.to(torch.float32) + 1.0)).sum().item()
        pbar.set_postfix({"MRR": f"{mrr_sum / done:.3f}", "done": done})

        del A, best_same, count_greater, batch_ranks
        if device == "cuda":
            torch.cuda.empty_cache()

    ranks = torch.cat(ranks_all, dim=0)                                                  # [N]
    topk = {f"top@{k}": (ranks < k).float().mean().item() for k in range(1, max_k+1)}
    mrr = (1.0 / (ranks.to(torch.float32) + 1.0)).mean().item()
    return ranks, topk, mrr