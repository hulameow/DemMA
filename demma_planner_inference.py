#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_planner_chatbot.py
----------------------------------
推理脚本（训练/推理一致 + Chatbot模式）
"""
import json
import re
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "checkpoints/planner_true_multitask/best"
ACTION_CLASSIFIER_PATH = "checkpoints/planner_true_multitask/best/action_classifier.pt"

ACTION_LABELS = {
    "movement": [
        "standing up", "stepping back", "freezing mid-step", "rubbing fingers",
        "fiddling with clothing", "touching forehead", "clenching fist", "slapping table",
        "shaking hands", "nodding", "shaking head", "lowering head", "looking around",
        "throwing objects", "pacing back and forth", "fidgeting", "gripping armrest tightly",
        "covering ears", "holding caregiver's hand", "pushing caregiver away"
    ],
    "facial_expression": [
        "avoiding eye contact", "staring blankly", "frowning", "smiling",
        "laughing", "vacant expression", "very surprised (wow)"
    ],
    "voice": [
        "sighing", "verbal hesitation (um / uh)", "murmuring / self-talk",
        "silence for several seconds", "crying", "repetitive words", "groaning in pain"
    ]
}

def build_action_vocab():
    vocab = {}
    idx_to_label = {}
    idx = 0
    for category, labels in ACTION_LABELS.items():
        for label in labels:
            key = f"{category}:{label}"
            vocab[key] = idx
            idx_to_label[idx] = (category, label)
            idx += 1
    return vocab, idx_to_label, idx

ACTION_VOCAB, IDX_TO_LABEL, NUM_ACTION_LABELS = build_action_vocab()

def load_model_with_classifier(model_path: str, classifier_path: str, device: str = "cuda"):
    print(f"📖 Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        trust_remote_code=True,
        device_map=device
    )

    print(f"📖 Loading action classifier...")
    hidden_size = base_model.config.hidden_size

    action_classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.LayerNorm(hidden_size // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, NUM_ACTION_LABELS)
    )

    action_classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    action_classifier = action_classifier.half().to(device)
    action_classifier.eval()
    base_model.eval()

    print(f"✅ Loaded on {device}")
    return tokenizer, base_model, action_classifier

def build_inference_prompt(
        persona: str,
        icf_b126_profile: Dict[str, int],
        memory_profile: Dict[str, Any],
        dialogue_history: List[Dict[str, str]],
        caregiver_utterance: str
) -> str:
    icf = icf_b126_profile
    mem = memory_profile

    prompt = f"""Patient: {persona}
ICF: E={icf['extraversion']}, A={icf['agreeableness']}, C={icf['conscientiousness']}, ES={icf['emotional_stability']}
Memory: {mem['deficit_type']}, Recent={'✓' if mem.get('has_recent_episodic') else '✗'}, Cues={'yes' if mem.get('benefits_from_cues') else 'no'}

History:
"""

    history = dialogue_history[-5:] if len(dialogue_history) > 5 else dialogue_history

    if history:
        for turn in history:
            utterance = turn['utterance']
            if turn['speaker'] == 'Patient':
                utterance = re.sub(r'\s*\[Movement:.*?\]\s*$', '', utterance)
                utterance = utterance.strip()
            prompt += f"{turn['speaker']}: {utterance}\n"
    else:
        prompt += "(First turn)\n"

    prompt += f"\nCaregiver: {caregiver_utterance}"

    prompt += """

Generate in this structure:
[PLAN]
Emotion: <emotion> (I=<intensity>)
Memory: <accessibility> | <deficit_type>
Intent: <response_intent>
Reason: <rationale>

[SPEAK]
<patient utterance>

[ACT]
(predicted by classifier)"""

    return prompt

@torch.no_grad()
def generate_response(
        tokenizer,
        base_model,
        action_classifier,
        persona: str,
        icf_b126_profile: Dict,
        memory_profile: Dict,
        dialogue_history: List[Dict],
        caregiver_utterance: str,
        temperature: float = 0.5,
        max_new_tokens: int = 3200
) -> Dict[str, Any]:

    user_content = build_inference_prompt(
        persona, icf_b126_profile, memory_profile,
        dialogue_history, caregiver_utterance
    )

    messages = [
        {"role": "system", "content": "You are a dementia patient planner."},
        {"role": "user", "content": user_content}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(base_model.device)

    outputs = base_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_ids = outputs[0]

    with torch.no_grad():
        attention_mask = (generated_ids != tokenizer.pad_token_id).long()

        full_outputs = base_model(
            input_ids=generated_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = full_outputs.hidden_states[-1]
        seq_length = attention_mask.sum().item() - 1

        special_tokens = {tokenizer.eos_token_id, tokenizer.pad_token_id}
        if hasattr(tokenizer, 'im_end_id'):
            special_tokens.add(tokenizer.im_end_id)

        actual_last_idx = seq_length
        for i in range(seq_length, max(0, seq_length - 10), -1):
            if generated_ids[i].item() not in special_tokens:
                actual_last_idx = i
                break

        last_valid_hidden = hidden_states[0, actual_last_idx, :]

        action_logits = action_classifier(last_valid_hidden.unsqueeze(0))
        action_probs = torch.sigmoid(action_logits).squeeze(0).cpu().numpy()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    m = re.search(r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)', generated_text, re.DOTALL)
    if m:
        response_text = m.group(1).strip()
    else:
        response_text = generated_text.split("assistant")[-1].strip()

    parsed = parse_generated_text(response_text)
    parsed["actions"] = decode_action_predictions(action_probs, threshold=0.5)
    parsed["action_probs"] = action_probs.tolist()

    return parsed

def parse_generated_text(text: str) -> Dict[str, Any]:
    result = {"planner": {}, "utterance": "", "raw_text": text}

    plan_match = re.search(r'\[PLAN\](.*?)(?:\[SPEAK\]|$)', text, re.DOTALL)
    if plan_match:
        result["planner"] = parse_planner_text(plan_match.group(1).strip())

    speak_match = re.search(r'\[SPEAK\](.*?)(?:\[ACT\]|$)', text, re.DOTALL)
    if speak_match:
        result["utterance"] = speak_match.group(1).strip()

    return result

def parse_planner_text(text: str) -> Dict:
    planner = {}

    if m := re.search(r'Emotion:\s*(\w+)\s*\(I=(\d+)\)', text):
        planner["emotion"] = m.group(1)
        planner["intensity"] = int(m.group(2))

    if m := re.search(r'Memory:\s*(\w+)\s*\|\s*([\w_]+)', text):
        planner["memory_accessibility"] = m.group(1)
        planner["deficit_type"] = m.group(2)

    if m := re.search(r'Intent:\s*(.+?)(?:\n|Reason:)', text):
        planner["intent"] = m.group(1).strip()

    if m := re.search(r'Reason:\s*(.+?)(?:\n\[|$)', text, re.DOTALL):
        planner["reason"] = m.group(1).strip()

    return planner

def decode_action_predictions(probs: np.ndarray, threshold: float = 0.5) -> Dict[str, List[str]]:
    """Per-category top-1"""
    res = {"movement": [], "facial_expression": [], "voice": []}

    for cat in ACTION_LABELS:
        indices = [i for i, (c, _) in IDX_TO_LABEL.items() if c == cat]
        best_idx = max(indices, key=lambda i: probs[i])
        best_prob = probs[best_idx]

        if best_prob >= threshold:
            _, label = IDX_TO_LABEL[best_idx]
            res[cat].append(label)

    return res

class DementiaPatientChatbot:
    """交互式Chatbot[1]"""

    def __init__(self, model_path, classifier_path, patient_profile, device="cuda"):
        self.tokenizer, self.base_model, self.action_classifier = load_model_with_classifier(
            model_path, classifier_path, device
        )
        self.patient_profile = patient_profile
        self.dialogue_history = []

        print("\n" + "=" * 80)
        print(f"🧠 Dementia Patient Chatbot Initialized")
        print(f"   Persona: {patient_profile['persona']}")
        print(f"   Memory: {patient_profile['memory_profile']['deficit_type']}")
        print("=" * 80 + "\n")

    def chat(self, caregiver_utterance: str, temperature: float = 0.5) -> Dict[str, Any]:
        result = generate_response(
            self.tokenizer, self.base_model, self.action_classifier,
            self.patient_profile["persona"],
            self.patient_profile["icf_b126_profile"],
            self.patient_profile["memory_profile"],
            self.dialogue_history,
            caregiver_utterance,
            temperature=temperature
        )

        self.dialogue_history.append({"speaker": "Caregiver", "utterance": caregiver_utterance})
        self.dialogue_history.append({"speaker": "Patient", "utterance": result["utterance"]})

        return result

    def reset(self):
        self.dialogue_history = []
        print("🔄 Dialogue history reset")

    def get_history(self):
        return self.dialogue_history

def main_interactive():
    patient_profile = {
        "persona": "AD-early",
        "icf_b126_profile": {
            "extraversion": -1,
            "agreeableness": -1,
            "conscientiousness": -2,
            "emotional_stability": -2
        },
        "memory_profile": {
            "deficit_type": "encoding_deficit",
            "has_recent_episodic": False,
            "has_remote_episodic": True,
            "benefits_from_cues": False
        }
    }

    chatbot = DementiaPatientChatbot(MODEL_PATH, ACTION_CLASSIFIER_PATH, patient_profile)

    print("💬 Interactive Chatbot Mode")
    print("   Commands: /quit, /reset, /history")
    print("=" * 80 + "\n")

    while True:
        caregiver_input = input("\n🩺 Caregiver: ").strip()

        if caregiver_input == "/quit":
            print("👋 Goodbye!")
            break
        elif caregiver_input == "/reset":
            chatbot.reset()
            continue
        elif caregiver_input == "/history":
            print(json.dumps(chatbot.get_history(), indent=2, ensure_ascii=False))
            continue
        elif not caregiver_input:
            continue

        try:
            result = chatbot.chat(caregiver_input)

            print(f"\n🔍 [DEBUG] Raw Text:")
            print(result.get("raw_text", "N/A"))
            print("...\n")

            print(f"🧠 [Planner]:")
            print(f"   Emotion: {result['planner'].get('emotion', 'N/A')} (I={result['planner'].get('intensity', 'N/A')})")
            print(f"   Memory: {result['planner'].get('memory_accessibility', 'N/A')}")
            print(f"   Intent: {result['planner'].get('intent', 'N/A')}")

            if result['utterance']:
                print(f"\n👤 Patient: {result['utterance']}")
            else:
                print(f"\n⚠️  Patient: (EMPTY - 生成被截断)")

            if any(result["actions"].values()):
                print(f"\n🎭 [Actions]:")
                for cat, labels in result["actions"].items():
                    if labels:
                        print(f"   {cat}: {', '.join(labels)}")

            print(f"\n📊 [Action Probs - Top 5]:")
            probs = result["action_probs"]
            top_indices = np.argsort(probs)[::-1][:5]
            for idx in top_indices:
                cat, label = IDX_TO_LABEL[idx]
                print(f"   {probs[idx]:.3f} - {cat}: {label}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    tokenizer, base_model, action_classifier = load_model_with_classifier(
        MODEL_PATH, ACTION_CLASSIFIER_PATH
    )

    patient_profile = {
        "persona": "AD-early",
        "icf_b126_profile": {
            "extraversion": -1,
            "agreeableness": -1,
            "conscientiousness": -2,
            "emotional_stability": -2
        },
        "memory_profile": {
            "deficit_type": "encoding_deficit",
            "has_recent_episodic": False,
            "has_remote_episodic": True,
            "benefits_from_cues": False
        }
    }

    dialogue_history = [
        {"speaker": "Caregiver", "utterance": "Good morning. Did you have breakfast?"},
        {"speaker": "Patient", "utterance": "Um... I don't remember. Did I?"}
    ]

    caregiver_utterance = "Yes, you had toast an hour ago. Would you like tea?"

    print("\n" + "=" * 80)
    print("🧠 Planner Inference：")
    print("=" * 80)

    result = generate_response(
        tokenizer, base_model, action_classifier,
        patient_profile["persona"],
        patient_profile["icf_b126_profile"],
        patient_profile["memory_profile"],
        dialogue_history,
        caregiver_utterance
    )

    print("\n📊 Result:")
    print("=" * 80)
    print("\n🧠 Planner:")
    print(json.dumps(result["planner"], indent=2, ensure_ascii=False))
    print(f"\n💬 Utterance:\n   \"{result['utterance']}\"")
    print("\n🎭 Actions:")
    for cat, labels in result["actions"].items():
        if labels:
            print(f"   {cat}: {', '.join(labels)}")
        else:
            print(f"   {cat}: (none)")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        main_interactive()
    else:
        main()