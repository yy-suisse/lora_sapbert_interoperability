import torch
import torch.nn.functional as F
import contextlib
from contextlib import nullcontext
from tqdm.auto import tqdm
import polars as pl


def load_checkpoint(model,
                    path,
                    optimizer=None,
                    scaler=None,
                    device="cuda",
                    strict=True):
    """
    Loads a checkpoint saved by:
      torch.save({"model": state_dict, "optimizer": ..., "scaler": ..., "step": step}, path)

    Returns:
      start_step (int)
    """
    ckpt = torch.load(path, map_location=device)

    # --- fix possible DataParallel prefixes mismatches ---
    sd = ckpt["model"]
    def has_module_prefix(state_dict): 
        k = next(iter(state_dict))
        return k.startswith("module.")
    model_sd = model.state_dict()
    if has_module_prefix(sd) and not has_module_prefix(model_sd):
        # remove 'module.' from checkpoint keys
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    elif not has_module_prefix(sd) and has_module_prefix(model_sd):
        # add 'module.' to checkpoint keys
        sd = {f"module.{k}": v for k, v in sd.items()}

    # --- load model ---
    # If your architecture changed a bit, set strict=False to ignore missing/unexpected keys.
    info = model.load_state_dict(sd, strict=strict)
    if hasattr(info, "missing_keys") and (info.missing_keys or info.unexpected_keys):
        print("[load_state_dict] missing:", info.missing_keys)
        print("[load_state_dict] unexpected:", info.unexpected_keys)

    model.to(device)

    # --- load optimizer (optional) ---
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    # --- load AMP scaler (optional) ---
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    # --- training step (default 0 if absent) ---
    start_step = int(ckpt.get("step", 0))
    return start_step


@torch.inference_mode()  # same as no_grad + a few extra runtime wins
def encode(model, texts, batch_size=2048, device="cuda", head="q", return_dtype=torch.float16):
    """
    Encode a list of texts in small CUDA-friendly batches.
    - Works if model(texts) returns a tensor or a dict (use `head` for the dict key).
    - Normalizes embeddings to unit length.
    - Stores results on CPU (fp16 by default) to keep VRAM low.
    """
    # Move model once
    was_training = model.training
    model.eval().to(device)

    N = len(texts)
    chunks = []

    use_amp = (device.startswith("cuda"))
    amp_ctx = torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()

    for start in tqdm(range(0, N, batch_size), desc="Encoding", leave=False):
        batch = texts[start:start+batch_size]  # leave as Python list/strings

        with amp_ctx:
            out = model(batch)                 # tensor or dict
            vec = out[head] if isinstance(out, dict) else out
            vec = F.normalize(vec, p=2, dim=-1)

        # Move to CPU immediately to free VRAM; keep fp16 by default
        chunks.append(vec.detach().to(dtype=return_dtype, device="cpu"))

        # Hard free GPU buffers for long runs
        del vec, out
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    embs = torch.cat(chunks, dim=0)

    # Restore training mode if needed
    if was_training:
        model.train()

    return embs

def get_embeddings(emb_model, df_concept, col_name):
    concept_labels = df_concept[col_name].to_list()
    emb_concepts = encode(emb_model, concept_labels, batch_size=2048, device="cuda", head="q", return_dtype=torch.float16)
    return emb_concepts