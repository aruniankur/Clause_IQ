# train_qwen3_lora
import os, random, math, json, re
from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer
)
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from packaging.version import parse as V
import transformers

# -----------------------
# Config
# -----------------------
SEED = 42
CSV_PATH = "training_dataset/training_dataset.csv"  

MODEL_NAME = "Qwen/Qwen3-1.7B"
OUT_DIR = "aiquest_custom_qwen_1.7B"

# LoRA (no QLoRA)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj","k_proj","v_proj","o_proj",          # attention projections
    "gate_proj","up_proj","down_proj"             # MLP (FFN) projections
]

# Loss mixing
TRIPLET_MARGIN = 0.20            
LAMBDA_TRIPLET = 0.20           
LAMBDA_INFO_NCE = 0.25           
INFO_NCE_TAU = 0.07
INFO_NCE_QUEUE = 4096            # size of memory bank

# Training args
EPOCHS = 25
BATCH_SIZE = 1
GRAD_ACCUM = 32
LR = 2e-4
EVAL_STEPS = 200
SAVE_STEPS = 200
MAX_LEN = 1024
TRIPLET_BSZ = 1  
# -----------------------
# Repro
# -----------------------
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------
# Load & normalize data
# -----------------------
df = pd.read_csv(CSV_PATH)

# normalize columns
col_map = {
    "Attribute": "attribute",
    "Anchor Element": "anchor",
    "GeneratedClause": "clause",
    "Candidate Clause": "clause",
    "Label": "label",
}
for k,v in col_map.items():
    if k in df.columns:
        df.rename(columns={k:v}, inplace=True)

if not {"attribute","anchor","clause","label"}.issubset(df.columns):
    raise ValueError(f"CSV must have columns: {list(col_map.keys())}")

# normalize labels
df["label"] = df["label"].astype(str).str.strip().str.lower().map({
    "standard":"Standard",
    "non-standard":"Non-Standard",
    "nonstandard":"Non-Standard",
    "non standard":"Non-Standard",
})
df = df.dropna(subset=["attribute","anchor","clause","label"]).reset_index(drop=True)

# -----------------------
# Build triplets per anchor
# -----------------------
triplets = []
for anchor, g in df.groupby("anchor"):
    pos = g[g["label"]=="Standard"]["clause"].tolist()
    neg = g[g["label"]=="Non-Standard"]["clause"].tolist()
    if not pos or not neg:
        continue
    m = min(len(pos), len(neg))
    for i in range(m):
        triplets.append({
            "anchor": anchor,
            "pos": pos[i % len(pos)],
            "neg": neg[i % len(neg)],
            "attribute": g["attribute"].iloc[0]
        })
trip_df = pd.DataFrame(triplets)

# Split by anchor to avoid leakage
anchors_all = df["anchor"].drop_duplicates().tolist()
train_a, eval_a = train_test_split(anchors_all, test_size=0.2, random_state=SEED)

sft_train = df[df["anchor"].isin(train_a)].reset_index(drop=True)
sft_eval  = df[df["anchor"].isin(eval_a)].reset_index(drop=True)
trip_train = trip_df[trip_df["anchor"].isin(train_a)].reset_index(drop=True)
trip_eval  = trip_df[trip_df["anchor"].isin(eval_a)].reset_index(drop=True)

print(f"SFT train/eval: {len(sft_train)}/{len(sft_eval)}")
print(f"Triplet train/eval: {len(trip_train)}/{len(trip_eval)}")

# -----------------------
# Tokenizer & model (LoRA only; no QLoRA)
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Keep the label tokens consistent (leading space matters for BPE)
LABEL_TEXT = {"Standard": " Standard", "Non-Standard": " Non-Standard"}
tokenizer.truncation_side = "left"   # keep label at the end if truncation happens
tokenizer.padding_side = "right"

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    trust_remote_code=True
)
model.config.use_cache = False
model.config.output_hidden_states = True

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=LORA_TARGETS,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# If you keep gradient_checkpointing=True in TrainingArguments, you must do both:
model.gradient_checkpointing_enable()
model.enable_input_require_grads()   # <<< CRITICAL for checkpointing to work

model.train()
print(model.print_trainable_parameters())

# -----------------------
# Prompt formatting
# -----------------------
SYSTEM_RULES = (
"Decide if the candidate clause matches the standard template.\n"
"Standard: exact structural match, value substitution with same intent, or minor wording changes.\n"
"Non-Standard: carve-outs or exceptions, added conditions/timing, or different reimbursement methodologies."
)

def make_chat(attribute, anchor, clause):
    user = (
        f"Attribute: {attribute}\n"
        f"Anchor (Standard): {anchor}\n"
        f"Candidate: {clause}\n"
        f"Answer with exactly one word: Standard or Non-Standard.\n"
        f"Label:"
    )
    msgs = [
        {"role":"system","content":SYSTEM_RULES},
        {"role":"user","content":user}
    ]
    return tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )

def build_sft_dataset(frame: pd.DataFrame) -> Dataset:
    prompts, labels = [], []
    for r in frame.itertuples(index=False):
        prompts.append(make_chat(r.attribute, r.anchor, r.clause))
        labels.append(r.label)
    return Dataset.from_dict({"prompt":prompts, "label":labels})

sft_train_ds = build_sft_dataset(sft_train)
sft_eval_ds  = build_sft_dataset(sft_eval)

# After: sft_train_ds = build_sft_dataset(sft_train)
#        sft_eval_ds  = build_sft_dataset(sft_eval)

MAX_LEN = 512  # keep short for memory

LABEL_TEXT = {"Standard": " Standard", "Non-Standard": " Non-Standard"}

def tokenize_sft(batch):
    prompts = batch["prompt"]
    labels  = batch["label"]
    # full text includes the gold label suffix we want to supervise
    full_texts = [p + LABEL_TEXT[l] for p, l in zip(prompts, labels)]
    # tokenize both prompt-only and full text; truncation is important
    full_enc   = tokenizer(full_texts, padding=False, truncation=True, max_length=MAX_LEN)
    prompt_enc = tokenizer(prompts,    padding=False, truncation=True, max_length=MAX_LEN)

    new_labels = []
    for ids, p_ids in zip(full_enc["input_ids"], prompt_enc["input_ids"]):
        lab = [-100] * len(ids)
        cut = len(p_ids)
        if cut < len(ids):
            lab[cut:] = ids[cut:]   # supervise ONLY the appended label tokens
        new_labels.append(lab)

    full_enc["labels"] = new_labels
    return full_enc

tokenized_train = sft_train_ds.map(tokenize_sft, batched=True, remove_columns=["prompt","label"])
tokenized_eval  = sft_eval_ds.map(tokenize_sft,  batched=True, remove_columns=["prompt","label"])


trip_train_ds = Dataset.from_pandas(trip_train)
trip_eval_ds  = Dataset.from_pandas(trip_eval)

# -----------------------
# Collators
# -----------------------
@dataclass
class SFTCollator:
    max_length: int = MAX_LEN
    def __call__(self, batch):
        prompts = [b["prompt"] for b in batch]
        labels  = [b["label"]  for b in batch]
        # Build "prompt + gold label" and then mask prompt tokens
        suffixed = [p + LABEL_TEXT[l] for p,l in zip(prompts, labels)]
        enc = tokenizer(suffixed, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        labs = enc["input_ids"].clone()
        for i, (p, l) in enumerate(zip(prompts, labels)):
            plen = len(tokenizer(p, add_special_tokens=False)["input_ids"])
            # mask everything before the appended label
            labs[i, :plen] = -100
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labs}

sft_collator = SFTCollator()

@dataclass
class TripletCollator:
    max_length: int

    def __call__(self, batch):
        # batch may be dict-of-lists (HF slicing) or list-of-dicts (DataLoader)
        if isinstance(batch, dict):
            anchors = batch["anchor"]
            pos     = batch["pos"]
            neg     = batch["neg"]
        else:
            anchors = [b["anchor"] for b in batch]
            pos     = [b["pos"]    for b in batch]
            neg     = [b["neg"]    for b in batch]

        a = tokenizer(anchors, padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
        p = tokenizer(pos,     padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
        n = tokenizer(neg,     padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
        return {"anc": a, "pos": p, "neg": n}

# instantiate it (tokenizer and MAX_LEN must already be defined)
trip_collator = TripletCollator(max_length=MAX_LEN)


# -----------------------
# InfoNCE feature queue
# -----------------------
class FeatureQueue:
    def __init__(self, dim, max_size=INFO_NCE_QUEUE, device=None):
        self.max_size = max_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bank = torch.empty(0, dim, device=self.device)
    @torch.no_grad()
    def add(self, x):  # x: [B, D]
        x = x.detach()
        if self.bank.numel() == 0:
            self.bank = x[-self.max_size:]
        else:
            self.bank = torch.cat([self.bank, x], dim=0)[-self.max_size:]
    def get(self):
        return self.bank if self.bank.numel() else None

def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1)  # [B,T,1]
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1)
    return summed / denom

def info_nce_from_pairs(emb_q, emb_pos, bank, tau=INFO_NCE_TAU):
    q = F.normalize(emb_q, dim=-1)
    p = F.normalize(emb_pos, dim=-1)
    pos_logit = (q * p).sum(dim=-1, keepdim=True)     # [B,1]
    if bank is not None:
        n = F.normalize(bank, dim=-1)                 # [M,D]
        neg_logits = q @ n.T                          # [B,M]
        logits = torch.cat([pos_logit, neg_logits], dim=1)
    else:
        logits = pos_logit
    logits = logits / tau
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)  # positive at index 0
    return F.cross_entropy(logits, labels)

# -----------------------
# Custom Trainer (SFT + Triplet + InfoNCE)
# -----------------------
class MultiLossTrainer(Trainer):
    def __init__(self, *args,
                 triplet_dataset: Dataset = None,
                 triplet_collator: TripletCollator = None,
                 lambda_triplet=LAMBDA_TRIPLET,
                 triplet_margin=TRIPLET_MARGIN,
                 lambda_infonce=LAMBDA_INFO_NCE,
                 tau=INFO_NCE_TAU,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.triplet_dataset = triplet_dataset
        self.triplet_collator = triplet_collator
        self.lambda_triplet = lambda_triplet
        self.triplet_margin = triplet_margin
        self.lambda_infonce = lambda_infonce
        self.tau = tau
        self._trip_idx = 0
        self._feature_queue = None  # init lazily on first pass

    def get_triplet_batch(self, bsz):
        # simple cycling batcher
        N = len(self.triplet_dataset)
        if N == 0:
            return None
        if self._trip_idx + bsz > N:
            self._trip_idx = 0
        sl = self.triplet_dataset[self._trip_idx:self._trip_idx + bsz]
        self._trip_idx += bsz
        return self.triplet_collator(sl)

    def forward_hidden(self, model, batch):
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True
        )
        hid = out.hidden_states[-1]
        emb = mean_pool(hid, batch["attention_mask"])
        return emb

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1) SFT CE loss (keep graph; no hidden states needed)
        outputs = model(**inputs, output_hidden_states=False, use_cache=False)
        loss_sft = outputs.loss  # requires grad

        # 2) Triplet + 3) InfoNCE
        trip_bsz = TRIPLET_BSZ  # tiny to save memory
        trip = self.get_triplet_batch(trip_bsz)
        if trip is None:
            loss = loss_sft
            return (loss, outputs) if return_outputs else loss

        # move to device (no no_grad here)
        device = model.device
        for k in ["anc","pos","neg"]:
            for kk in trip[k]:
                trip[k][kk] = trip[k][kk].to(device)

        out_anc = model(input_ids=trip["anc"]["input_ids"],
                        attention_mask=trip["anc"]["attention_mask"],
                        output_hidden_states=True, use_cache=False)
        out_pos = model(input_ids=trip["pos"]["input_ids"],
                        attention_mask=trip["pos"]["attention_mask"],
                        output_hidden_states=True, use_cache=False)
        out_neg = model(input_ids=trip["neg"]["input_ids"],
                        attention_mask=trip["neg"]["attention_mask"],
                        output_hidden_states=True, use_cache=False)

        emb_a = mean_pool(out_anc.hidden_states[-1], trip["anc"]["attention_mask"])
        emb_p = mean_pool(out_pos.hidden_states[-1], trip["pos"]["attention_mask"])
        emb_n = mean_pool(out_neg.hidden_states[-1], trip["neg"]["attention_mask"])

        # Triplet margin loss (keeps grad)
        sim_ap = F.cosine_similarity(emb_a, emb_p)
        sim_an = F.cosine_similarity(emb_a, emb_n)
        loss_triplet = torch.clamp(self.triplet_margin - sim_ap + sim_an, min=0).mean()

        # InfoNCE with queue (keeps grad w.r.t emb_a/emb_p)
        if self._feature_queue is None:
            self._feature_queue = FeatureQueue(dim=emb_a.size(-1), max_size=INFO_NCE_QUEUE, device=device)
        bank = self._feature_queue.get()
        loss_nce = info_nce_from_pairs(emb_a, emb_p, bank, tau=self.tau)

        # Update memory bank (detached!)
        with torch.no_grad():
            self._feature_queue.add(torch.cat([emb_p, emb_n], dim=0))

        # Total loss
        loss = loss_sft + self.lambda_triplet * loss_triplet + self.lambda_infonce * loss_nce

        # Log detached copies (no graph break)
        self.log({
            "loss_sft":     loss_sft.detach().float(),
            "loss_triplet": loss_triplet.detach().float(),
            "loss_infonce": loss_nce.detach().float()
        })
        return (loss, outputs) if return_outputs else loss


# -----------------------
# HF datasets
# -----------------------
train_ds = tokenized_train
eval_ds  = tokenized_eval

# -----------------------
# Train
# -----------------------


ver = V(transformers.__version__)

# common kwargs that are valid across versions
base_args = dict(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.05,           # scheduler/warmup set below for v5
    logging_steps=25,
    save_steps=SAVE_STEPS,       # overridden by set_save for v5
    bf16=(dtype==torch.bfloat16),
    fp16=(dtype==torch.float16),
    gradient_checkpointing=True,
    report_to="none",
)

args = transformers.TrainingArguments(**base_args)

if ver.major >= 5:
    # New v5 style: use setters
    args = args.set_evaluate(strategy="steps", steps=EVAL_STEPS)
    args = args.set_save(strategy="steps", steps=SAVE_STEPS)
    args = args.set_lr_scheduler(name="cosine", warmup_ratio=0.05)
    # (optional) logging strategy also has a setter:
    # args = args.set_logging(strategy="steps", steps=25, report_to="none")
else:
    # Legacy v4 style fields still exist
    args.evaluation_strategy = "steps"
    args.eval_steps = EVAL_STEPS
    args.lr_scheduler_type = "cosine"


trainer = MultiLossTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=None,             
    processing_class=tokenizer,     
    triplet_dataset=trip_train_ds,
    triplet_collator=trip_collator,
    lambda_triplet=LAMBDA_TRIPLET,
    triplet_margin=TRIPLET_MARGIN,
    lambda_infonce=LAMBDA_INFO_NCE,
    tau=INFO_NCE_TAU
)

trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("Done. Model saved to:", OUT_DIR)
