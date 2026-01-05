#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-task SFT training:
L_total = w_p * L_planner + w_u * L_utterance + w_a * L_action
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import bitsandbytes as bnb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, set_seed
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score


# =========================
# Labels
# =========================
ACTION_LABELS = {
    "movement": [
        "standing up", "stepping back", "freezing mid-step", "rubbing fingers",
        "fiddling with clothing", "touching forehead", "clenching fist", "slapping table",
        "shaking hands", "nodding", "shaking head", "lowering head", "looking around",
        "throwing objects", "pacing back and forth", "fidgeting", "gripping armrest tightly",
        "covering ears", "holding caregiver's hand", "pushing caregiver away",
    ],
    "facial_expression": [
        "avoiding eye contact", "staring blankly", "frowning", "smiling",
        "laughing", "vacant expression", "very surprised (wow)",
    ],
    "voice": [
        "sighing", "verbal hesitation (um / uh)", "murmuring / self-talk",
        "silence for several seconds", "crying", "repetitive words", "groaning in pain",
    ],
}

PLAN_START = "[PLAN]"
SPEAK_START = "[SPEAK]"
ACT_START = "[ACT]"


def build_action_vocab() -> Tuple[Dict[str, int], int]:
    vocab: Dict[str, int] = {}
    idx = 0
    for category, labels in ACTION_LABELS.items():
        for label in labels:
            vocab[f"{category}:{label}"] = idx
            idx += 1
    return vocab, idx


ACTION_VOCAB, NUM_ACTION_LABELS = build_action_vocab()


# =========================
# Config
# =========================
@dataclass
class Config:
    train_data_path: str = "DemMa_planner_labeled_dialogue_corpus.jsonl"
    val_split: float = 0.15
    base_model: str = "Qwen/Qwen3-8B"
    output_dir: str = "checkpoints/planner_true_multitask"

    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-6
    num_epochs: int = 5
    max_length: int = 1536

    warmup_ratio: float = 0.15
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    weight_planner: float = 0.35
    weight_utterance: float = 0.45
    weight_action: float = 0.20

    dropout: float = 0.1
    label_smoothing: float = 0.1

    seed: int = 42
    logging_steps: int = 5
    early_stopping_patience: int = 3

    fp16: bool = True
    gradient_checkpointing: bool = True

    num_workers: int = 0
    save_every_epoch: bool = False
    quiet_logs: bool = False


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train a multi-task planner/speech/action model.")
    p.add_argument("--train_data_path", type=str, default=Config.train_data_path)
    p.add_argument("--val_split", type=float, default=Config.val_split)
    p.add_argument("--base_model", type=str, default=Config.base_model)
    p.add_argument("--output_dir", type=str, default=Config.output_dir)

    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--gradient_accumulation_steps", type=int, default=Config.gradient_accumulation_steps)
    p.add_argument("--learning_rate", type=float, default=Config.learning_rate)
    p.add_argument("--num_epochs", type=int, default=Config.num_epochs)
    p.add_argument("--max_length", type=int, default=Config.max_length)

    p.add_argument("--warmup_ratio", type=float, default=Config.warmup_ratio)
    p.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    p.add_argument("--max_grad_norm", type=float, default=Config.max_grad_norm)

    p.add_argument("--weight_planner", type=float, default=Config.weight_planner)
    p.add_argument("--weight_utterance", type=float, default=Config.weight_utterance)
    p.add_argument("--weight_action", type=float, default=Config.weight_action)

    p.add_argument("--dropout", type=float, default=Config.dropout)
    p.add_argument("--label_smoothing", type=float, default=Config.label_smoothing)

    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--logging_steps", type=int, default=Config.logging_steps)
    p.add_argument("--early_stopping_patience", type=int, default=Config.early_stopping_patience)

    p.add_argument("--no_fp16", action="store_true", help="Disable fp16.")
    p.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing.")
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--quiet_logs", action="store_true")

    a = p.parse_args()
    cfg = Config(**vars(a))
    cfg.fp16 = not a.no_fp16
    cfg.gradient_checkpointing = not a.no_gradient_checkpointing
    return cfg


# =========================
# Dataset
# =========================
class TrueMultiTaskDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 1536, quiet: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[Dict[str, Any]] = []
        self.quiet = quiet

        if not self.quiet:
            print(f"Loading data: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        if not self.quiet:
            print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        messages = [
            {"role": "system", "content": "You are a patient dialogue planner."},
            {"role": "user", "content": self._build_input(sample)},
            {"role": "assistant", "content": self._build_output(sample)},
        ]

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0).tolist()

        planner_mask, utterance_mask, full_mask = self._create_segment_masks(
            full_text=full_text,
            offsets=offsets,
            seq_len=input_ids.size(0),
        )

        action_labels = self._extract_action_labels(sample)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "planner_mask": planner_mask,
            "utterance_mask": utterance_mask,
            "full_mask": full_mask,
            "action_labels": torch.tensor(action_labels, dtype=torch.float),
            "persona": sample.get("persona", "unknown"),
        }

    def _build_input(self, sample: Dict) -> str:
        icf = sample["icf_b126_profile"]
        mem = sample["memory_profile"]

        prompt = (
            f"Persona: {sample.get('persona','unknown')}\n"
            f"ICF: E={icf.get('extraversion')}, A={icf.get('agreeableness')}, "
            f"C={icf.get('conscientiousness')}, ES={icf.get('emotional_stability')}\n"
            f"Memory: {mem.get('deficit_type')}, "
            f"Recent={'yes' if mem.get('has_recent_episodic') else 'no'}, "
            f"Cues={'yes' if mem.get('benefits_from_cues') else 'no'}\n"
            f"History:\n"
        )

        history = sample.get("dialogue_history", [])[-5:]
        if history:
            for t in history:
                utterance = t.get("utterance", "")
                if t.get("speaker") == "Patient":
                    utterance = re.sub(r"\s*\[Movement:.*?\]\s*$", "", utterance).strip()
                prompt += f"{t.get('speaker','Unknown')}: {utterance}\n"
        else:
            prompt += "(first turn)\n"

        prompt += f"\nCaregiver: {sample.get('caregiver_utterance','')}"
        return prompt

    def _build_output(self, sample: Dict) -> str:
        planner = sample["planner_rationale"]
        utterance = sample["patient_utterance"]
        actions = sample["actions"]

        emo = planner["step4_emotion_prediction"]["emotion"]
        inten = planner["step4_emotion_prediction"]["intensity"]
        mem_acc = planner["step3_memory_query"]["accessibility"]
        mem_def = planner["step3_memory_query"]["deficit_type"]
        resp_intent = planner["step5_action_plan"]["response_intent"]
        icf_rationale = (planner["step4_emotion_prediction"].get("icf_b126_rationale", "") or "")[:120]

        out = (
            f"{PLAN_START}\n"
            f"Emotion: {emo} (I={inten})\n"
            f"Memory: {mem_acc} | {mem_def}\n"
            f"Intent: {resp_intent}\n"
            f"Reason: {icf_rationale}\n"
            f"{SPEAK_START}\n"
            f"{utterance}\n"
            f"{ACT_START}\n"
        )
        for cat in ["movement", "facial_expression", "voice"]:
            lbls = actions.get(cat, [])
            if lbls:
                out += f"{cat}: {', '.join(lbls)}\n"
        return out.strip()

    def _create_segment_masks(self, full_text: str, offsets: List[List[int]], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        planner_mask = torch.zeros(seq_len, dtype=torch.long)
        utterance_mask = torch.zeros(seq_len, dtype=torch.long)
        full_mask = torch.zeros(seq_len, dtype=torch.long)

        # This pattern is model/template dependent; keep it conservative.
        m = re.search(r"<\|im_start\|>assistant\n", full_text)
        if not m:
            return planner_mask, utterance_mask, full_mask

        assistant_start_char = m.end()
        plan_char = full_text.find(PLAN_START, assistant_start_char)
        speak_char = full_text.find(SPEAK_START, assistant_start_char)
        act_char = full_text.find(ACT_START, assistant_start_char)

        if plan_char == -1 or speak_char == -1:
            return planner_mask, utterance_mask, full_mask
        if act_char == -1:
            act_char = 10**9

        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0 and i > 0:
                continue
            if s >= assistant_start_char:
                full_mask[i] = 1
            if plan_char <= s < speak_char:
                planner_mask[i] = 1
            if speak_char <= s < act_char:
                utterance_mask[i] = 1

        return planner_mask, utterance_mask, full_mask

    def _extract_action_labels(self, sample: Dict) -> List[float]:
        vec = [0.0] * NUM_ACTION_LABELS
        actions = sample["actions"]
        for category, labels in actions.items():
            if not isinstance(labels, list):
                continue
            for label in labels:
                key = f"{category}:{label}"
                if key in ACTION_VOCAB:
                    vec[ACTION_VOCAB[key]] = 1.0
        return vec


# =========================
# Model
# =========================
class TrueMultiTaskModel(nn.Module):
    def __init__(self, base_model_name: str, num_action_labels: int, fp16: bool, gradient_checkpointing: bool, dropout: float, quiet: bool):
        super().__init__()
        attn_implementation = "eager"
        if fp16:
            try:
                import flash_attn  # noqa: F401
                attn_implementation = "flash_attention_2"
                if not quiet:
                    print("Flash Attention 2 detected")
            except ImportError:
                if not quiet:
                    print("Flash Attention 2 not available; using eager attention")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

        hidden_size = self.base_model.config.hidden_size
        self.action_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_action_labels),
        )

        if fp16:
            self.action_classifier = self.action_classifier.half()

        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        batch_size = input_ids.size(0)
        seq_lengths = attention_mask.sum(dim=1) - 1
        action_features = hidden_states[torch.arange(batch_size), seq_lengths]
        action_logits = self.action_classifier(action_features)
        return logits, action_logits


# =========================
# Loss
# =========================
class TrueMultiTaskLoss(nn.Module):
    def __init__(self, w_planner: float, w_utterance: float, w_action: float, label_smoothing: float = 0.0, quiet: bool = False):
        super().__init__()
        self.w_p = w_planner
        self.w_u = w_utterance
        self.w_a = w_action

        self.planner_criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing, reduction="mean")
        self.utterance_criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing, reduction="mean")
        self.action_criterion = nn.BCEWithLogitsLoss(reduction="none")

        if not quiet:
            print(f"Loss weights: w_p={w_planner}, w_u={w_utterance}, w_a={w_action}, label_smoothing={label_smoothing}")

    def forward(self, logits, input_ids, planner_mask, utterance_mask, action_logits, action_labels):
        batch_size, seq_len, vocab_size = logits.shape

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        shift_planner_mask = planner_mask[:, 1:].contiguous()
        shift_utterance_mask = utterance_mask[:, 1:].contiguous()

        planner_labels = shift_labels.clone()
        planner_labels[shift_planner_mask == 0] = -100

        utterance_labels = shift_labels.clone()
        utterance_labels[shift_utterance_mask == 0] = -100

        if (planner_labels != -100).any():
            loss_planner = self.planner_criterion(shift_logits.view(-1, vocab_size).float(), planner_labels.view(-1))
        else:
            loss_planner = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        if (utterance_labels != -100).any():
            loss_utterance = self.utterance_criterion(shift_logits.view(-1, vocab_size).float(), utterance_labels.view(-1))
        else:
            loss_utterance = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        bce = self.action_criterion(action_logits.float(), action_labels)
        probs = torch.sigmoid(action_logits.float())
        pt = action_labels * probs + (1 - action_labels) * (1 - probs)
        focal_weight = (1 - pt) ** 2
        loss_action = (bce * focal_weight).mean()

        total = self.w_p * loss_planner + self.w_u * loss_utterance + self.w_a * loss_action

        return total, {
            "loss_planner": float(loss_planner.detach().cpu()),
            "loss_utterance": float(loss_utterance.detach().cpu()),
            "loss_action": float(loss_action.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }


# =========================
# Metrics
# =========================
class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.planner = 0.0
        self.utterance = 0.0
        self.action = 0.0
        self.count = 0
        self.action_preds = []
        self.action_targets = []

    def update(self, losses: Dict[str, float], action_logits: torch.Tensor, action_targets: torch.Tensor):
        self.total += losses["total_loss"]
        self.planner += losses["loss_planner"]
        self.utterance += losses["loss_utterance"]
        self.action += losses["loss_action"]
        self.count += 1
        self.action_preds.append(torch.sigmoid(action_logits).detach().cpu())
        self.action_targets.append(action_targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        n = max(self.count, 1)
        out: Dict[str, float] = {
            "loss": self.total / n,
            "loss_planner": self.planner / n,
            "loss_utterance": self.utterance / n,
            "loss_action": self.action / n,
        }
        if self.action_preds:
            preds = torch.cat(self.action_preds, dim=0).numpy()
            targets = torch.cat(self.action_targets, dim=0).numpy()
            preds_bin = (preds > 0.5).astype(int)
            out["action_f1"] = f1_score(targets, preds_bin, average="micro", zero_division=0)
            out["action_precision"] = precision_score(targets, preds_bin, average="micro", zero_division=0)
            out["action_recall"] = recall_score(targets, preds_bin, average="micro", zero_division=0)
        return out


# =========================
# Train / Eval
# =========================
def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, cfg: Config, epoch: int):
    model.train()
    metrics = Metrics()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=cfg.quiet_logs)

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device).unsqueeze(0) if batch["input_ids"].dim() == 1 else batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device).unsqueeze(0) if batch["attention_mask"].dim() == 1 else batch["attention_mask"].to(device)
        planner_mask = batch["planner_mask"].to(device).unsqueeze(0) if batch["planner_mask"].dim() == 1 else batch["planner_mask"].to(device)
        utterance_mask = batch["utterance_mask"].to(device).unsqueeze(0) if batch["utterance_mask"].dim() == 1 else batch["utterance_mask"].to(device)
        action_labels = batch["action_labels"].to(device).unsqueeze(0) if batch["action_labels"].dim() == 1 else batch["action_labels"].to(device)

        logits, action_logits = model(input_ids, attention_mask)

        loss, loss_dict = loss_fn(logits, input_ids, planner_mask, utterance_mask, action_logits, action_labels)
        (loss / cfg.gradient_accumulation_steps).backward()

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        metrics.update(loss_dict, action_logits, action_labels)

        if not cfg.quiet_logs and (step + 1) % cfg.logging_steps == 0:
            m = metrics.compute()
            pbar.set_postfix({
                "L": f"{m['loss']:.3f}",
                "Lp": f"{m['loss_planner']:.3f}",
                "Lu": f"{m['loss_utterance']:.3f}",
                "La": f"{m['loss_action']:.3f}",
                "F1": f"{m.get('action_f1', 0.0):.3f}",
            })

    return metrics.compute()


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, cfg: Config):
    model.eval()
    metrics = Metrics()
    pbar = tqdm(loader, desc="Eval", disable=cfg.quiet_logs)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device).unsqueeze(0) if batch["input_ids"].dim() == 1 else batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device).unsqueeze(0) if batch["attention_mask"].dim() == 1 else batch["attention_mask"].to(device)
        planner_mask = batch["planner_mask"].to(device).unsqueeze(0) if batch["planner_mask"].dim() == 1 else batch["planner_mask"].to(device)
        utterance_mask = batch["utterance_mask"].to(device).unsqueeze(0) if batch["utterance_mask"].dim() == 1 else batch["utterance_mask"].to(device)
        action_labels = batch["action_labels"].to(device).unsqueeze(0) if batch["action_labels"].dim() == 1 else batch["action_labels"].to(device)

        logits, action_logits = model(input_ids, attention_mask)
        _, loss_dict = loss_fn(logits, input_ids, planner_mask, utterance_mask, action_logits, action_labels)
        metrics.update(loss_dict, action_logits, action_labels)

    return metrics.compute()


# =========================
# Main
# =========================
def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    print("=" * 80)
    print("Multi-task SFT training")
    print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"))
    print(f"L_total = {cfg.weight_planner}*L_planner + {cfg.weight_utterance}*L_utterance + {cfg.weight_action}*L_action")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA memory (device 0): {mem_gb:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TrueMultiTaskDataset(cfg.train_data_path, tokenizer, cfg.max_length, quiet=cfg.quiet_logs)

    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    print(f"Train size: {train_size}, Val size: {val_size}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = TrueMultiTaskModel(
        base_model_name=cfg.base_model,
        num_action_labels=NUM_ACTION_LABELS,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dropout=cfg.dropout,
        quiet=cfg.quiet_logs,
    ).to(device)

    optimizer = bnb.optim.AdamW8bit(
        [
            {"params": model.base_model.parameters(), "lr": cfg.learning_rate},
            {"params": model.action_classifier.parameters(), "lr": cfg.learning_rate * 5},
        ],
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.num_epochs // max(1, cfg.gradient_accumulation_steps)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_fn = TrueMultiTaskLoss(cfg.weight_planner, cfg.weight_utterance, cfg.weight_action, cfg.label_smoothing, quiet=cfg.quiet_logs)

    best_val = float("inf")
    patience = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")
        train_m = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, cfg, epoch + 1)
        val_m = evaluate(model, val_loader, loss_fn, device, cfg)

        history.append({"epoch": epoch + 1, "train": train_m, "val": val_m})

        print(f"Train: loss={train_m['loss']:.4f}, f1={train_m.get('action_f1', 0.0):.4f}")
        print(f"Val:   loss={val_m['loss']:.4f}, f1={val_m.get('action_f1', 0.0):.4f}")

        improved = val_m["loss"] < best_val
        if improved:
            best_val = val_m["loss"]
            patience = 0
            best_dir = output_dir / "best"
            best_dir.mkdir(exist_ok=True)

            model.base_model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            torch.save(model.action_classifier.state_dict(), best_dir / "action_classifier.pt")
            (best_dir / "metrics.json").write_text(json.dumps({"epoch": epoch + 1, "train": train_m, "val": val_m}, indent=2), encoding="utf-8")
            print(f"Saved best checkpoint to: {best_dir}")
        else:
            patience += 1
            print(f"No improvement. Patience {patience}/{cfg.early_stopping_patience}")
            if patience >= cfg.early_stopping_patience:
                print("Early stopping triggered.")
                break

        if cfg.save_every_epoch:
            ep_dir = output_dir / f"epoch_{epoch + 1}"
            ep_dir.mkdir(exist_ok=True)
            model.base_model.save_pretrained(ep_dir)
            tokenizer.save_pretrained(ep_dir)
            torch.save(model.action_classifier.state_dict(), ep_dir / "action_classifier.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
