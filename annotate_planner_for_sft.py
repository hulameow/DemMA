#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate_planner_for_sft.py

Generate planner-style annotations for SFT training.
Output fields:
- planner_rationale
- patient_utterance
- actions (movement / facial_expression / voice)

This script is anonymization-safe and English-only.
"""

import argparse
import json
import os
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from openai import OpenAI


# =========================
# Action labels (STRICT)
# =========================
ACTION_LABELS = {
    "movement": [
        "standing up", "stepping back", "freezing mid-step", "rubbing fingers",
        "fiddling with clothing", "touching forehead", "clenching fist",
        "slapping table", "shaking hands", "nodding", "shaking head",
        "lowering head", "looking around", "throwing objects",
        "pacing back and forth", "fidgeting", "gripping armrest tightly",
        "covering ears", "holding caregiver's hand", "pushing caregiver away",
    ],
    "facial_expression": [
        "avoiding eye contact", "staring blankly", "frowning",
        "smiling", "laughing", "vacant expression",
        "very surprised (wow)",
    ],
    "voice": [
        "sighing", "verbal hesitation (um / uh)",
        "murmuring / self-talk", "silence for several seconds",
        "crying", "repetitive words", "groaning in pain",
    ],
}


# =========================
# OpenAI client helper
# =========================
def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Please export OPENAI_API_KEY before running."
        )
    return OpenAI()


def safe_json_loads(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# =========================
# Action label validation
# =========================
def validate_action_labels(actions: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
    errors = []
    for cat in ["movement", "facial_expression", "voice"]:
        for lbl in actions.get(cat, []):
            if lbl not in ACTION_LABELS[cat]:
                errors.append(f"Invalid {cat} label: {lbl}")
    return len(errors) == 0, errors


# =========================
# Teacher call
# =========================
def call_teacher(
        client: OpenAI,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_retries: int,
        delay: float,
) -> Optional[Dict[str, Any]]:
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2000,
            )
            parsed = safe_json_loads(resp.choices[0].message.content)
            if parsed:
                return parsed
        except Exception:
            pass

        if attempt < max_retries:
            time.sleep(delay)
    return None


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate planner-style SFT annotations (English-only, anonymized)"
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--teacher_model", type=str, default=os.getenv("TEACHER_MODEL", "gpt-4o"))
    p.add_argument("--api_delay", type=float, default=0.5)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--drop_names", action="store_true")
    return p.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()
    client = get_client()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    dialogues = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            dialogues.append(json.loads(line))

    print(f"Loaded {len(dialogues)} dialogues")
    print(f"Teacher model: {args.teacher_model}")

    success, failed = 0, 0

    for d_idx, dialogue in enumerate(dialogues):
        persona = dialogue["persona"]
        turns = dialogue["turns"]

        for t_idx, turn in enumerate(turns):
            if turn["speaker"] != "Patient":
                continue

            user_prompt = build_teacher_prompt_sft(dialogue, t_idx)
            result = call_teacher(
                client,
                args.teacher_model,
                TEACHER_SYSTEM_PROMPT,
                user_prompt,
                args.max_retries,
                args.api_delay,
            )

            if result is None:
                failed += 1
                continue

            is_valid, _ = validate_action_labels(result.get("actions", {}))

            record = {
                "dialogue_id": dialogue.get("dialogue_id"),
                "patient_id": dialogue.get("patient_id"),
                "patient_name": None if args.drop_names else dialogue.get("patient_name"),
                "persona": persona,
                "round": turn["round"],
                "turn_index": t_idx,
                "icf_b126_profile": dialogue["icf_b126_profile"],
                "memory_profile": dialogue["memory_profile"],
                "dialogue_history": dialogue["turns"][:t_idx],
                "caregiver_utterance": (
                    turns[t_idx - 1]["utterance"]
                    if t_idx > 0 and turns[t_idx - 1]["speaker"] == "Caregiver"
                    else ""
                ),
                "planner_rationale": result["planner_rationale"],
                "patient_utterance": result["patient_utterance"],
                "actions": result["actions"],
                "teacher_model": args.teacher_model,
                "label_validation_passed": is_valid,
                "timestamp": time.time(),
