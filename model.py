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

device = "cuda" if torch.cuda.is_available() else "cpu"

class Sapbert(nn.Module):
    def __init__(
        self,
        max_length: int = 96,
        sapbert_path: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(sapbert_path, use_fast=True)
        base_model = AutoModel.from_pretrained(sapbert_path)
        self.model = base_model


    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, texts):
        toks = self.tokenize(texts)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.model(**toks, return_dict=True)
        cls_rep = out.last_hidden_state[:, 0, :]  # [CLS]
        return cls_rep  

class LoraSapbert(nn.Module):
    def __init__(
        self,
        max_length: int = 96,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        sapbert_path: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: str = "cuda",
        lora_dropout: float = 0.1,
        target_modules = ("key", "query", "value"),
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(sapbert_path, use_fast=True)
        base_model = AutoModel.from_pretrained(sapbert_path)

        self.lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
        )
        self.model = get_peft_model(base_model, self.lora_config).to(device)
        # Optional sanity check:
        self.model.print_trainable_parameters()

    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, texts):
        toks = self.tokenize(texts)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.model(**toks, return_dict=True)
        cls_rep = out.last_hidden_state[:, 0, :]  # [CLS]
        return cls_rep  