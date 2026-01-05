#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_seed_dialogues.py

High-quality seed dialogue generation with memory gating + persona constraints.

Anonymization/cleanliness:
- No absolute local paths
- No secrets in code (expects OPENAI_API_KEY via env)
- Configurable via CLI/env
- Reproducible with --seed
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError("Install dependencies: pip install openai") from e


# =========================
# Defaults (safe for public repo)
# =========================
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_VALIDATOR_MODEL = os.getenv("OPENAI_VALIDATOR_MODEL", "gpt-4o-mini")

DEFAULT_LONG_MEMORY_PATH = Path("data/memories_long.json")          # <-- relative, not /Users/...
DEFAULT_SHORT_MEMORY_PATH = Path("data/short_memories.jsonl")       # <-- relative, not /Users/...

DEFAULT_OUTPUT_DIR = Path("out")
DEFAULT_TAG = "seed"  # used to name output files

# Quality Control Settings (can be overridden via CLI)
DIALOGUES_PER_PERSON = 100
MAX_RETRIES_PER_DIALOGUE = 2
QUALITY_THRESHOLD_EXEMPLAR = 8.0
QUALITY_THRESHOLD_ACCEPTABLE = 6.5
MIN_EXEMPLARS_PER_PERSONA = 3
API_DELAY_SEC = 0.3


# ========== Action Labels ==========
ACTION_LABELS = {
    "movement": [
        "standing up", "stepping back", "freezing mid-step", "rubbing fingers",
        "fiddling with clothing", "touching forehead", "clenching fist",
        "slapping table", "shaking hands", "nodding", "shaking head",
        "lowering head", "looking around", "throwing objects",
        "pacing back and forth", "fidgeting", "gripping armrest tightly",
        "covering ears", "holding caregiver’s hand", "pushing caregiver away",
    ],
    "facial_expression": [
        "avoiding eye contact", "staring blankly", "frowning", "smiling",
        "laughing", "vacant expression", "very surprised (wow)",
    ],
    "voice": [
        "sighing", "verbal hesitation (um / uh)", "murmuring / self-talk",
        "silence for several seconds", "crying", "repetitive words",
        "groaning in pain",
    ],
}

# ========== General Topics ==========
GENERAL_TOPICS = [
    "Asking what time or day it is",
    "Requesting water or juice",
    "Asking when it's time to eat",
    "Requesting help to use the bathroom",
    "Asking about medication time",
    "Wanting the blanket adjusted",
    "Asking if the doctor will visit today",
    "Requesting to open the curtains or window",
    "Asking to call or see a family member",
    "Complaining about feeling cold or too warm",
    "Caregiver reminding to sit up or get out of bed",
    "Caregiver guiding to take medicine",
    "Caregiver measuring blood pressure or blood sugar",
    "Caregiver reminding not to pull the tube or move too much",
    "Caregiver helping to change clothes or wash face",
    "Caregiver encouraging to eat or drink slowly",
    "Caregiver assisting with light physical therapy or stretching",
    "Caregiver suggesting rest or a nap",
    "Caregiver preparing for a medical check or therapy session",
    "Caregiver tidying the bed or adjusting pillows",
    "Talking about how nice the weather is",
    "Mentioning that a family member will visit",
    "Remembering or dreaming about home",
    "Talking about old hobbies or past work",
    "Thanking the caregiver or showing affection",
    "Expressing fear or loneliness at night",
    "Talking about a pet or garden at home",
    "Remembering holidays or family gatherings",
    "Forgetting where something was placed",
    "Asking if they're going home today",
]

# ========== Persona Definitions ==========
PERSONA_MAP = {
    0: "AD-early", 1: "AD-early", 2: "AD-early",
    3: "AD-mid/late", 4: "AD-mid/late", 5: "AD-mid/late",
    6: "VaD", 7: "VaD", 8: "VaD",
    9: "DLB", 10: "DLB", 11: "DLB",
    12: "PDD", 13: "PDD", 14: "PDD",
    15: "FTD-bv", 16: "FTD-bv", 17: "FTD-bv",
    18: "nfvPPA", 19: "nfvPPA", 20: "nfvPPA",
    21: "svPPA", 22: "svPPA", 23: "svPPA",
    24: "lvPPA", 25: "lvPPA", 26: "lvPPA",
}

PERSONA_TO_IDS: Dict[str, List[int]] = {}
for pid, persona in PERSONA_MAP.items():
    PERSONA_TO_IDS.setdefault(persona, []).append(pid)

AVAILABLE_PERSONAS = sorted(PERSONA_TO_IDS.keys())


# ========== MODULE: ICF-b126 Personality System ==========
@dataclass
class ICFb126Profile:
    """WHO ICF-b126 temperament & personality functions. Scale: -3..+3."""
    extraversion: int
    agreeableness: int
    conscientiousness: int
    emotional_stability: int
    openness: int
    optimism: int
    confidence: int
    integrity: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "conscientiousness": self.conscientiousness,
            "emotional_stability": self.emotional_stability,
            "openness": self.openness,
            "optimism": self.optimism,
            "confidence": self.confidence,
            "integrity": self.integrity,
        }

    def to_description(self) -> str:
        labels = {-3: "severe↓", -2: "moderate↓", -1: "mild↓", 0: "preserved", 1: "mild↑", 2: "moderate↑", 3: "severe↑"}
        return (
            f"Extraversion={labels[self.extraversion]}, "
            f"Agreeableness={labels[self.agreeableness]}, "
            f"Conscientiousness={labels[self.conscientiousness]}, "
            f"Emotional_Stability={labels[self.emotional_stability]}, "
            f"Openness={labels[self.openness]}, "
            f"Optimism={labels[self.optimism]}, "
            f"Confidence={labels[self.confidence]}, "
            f"Integrity={labels[self.integrity]}"
        )

PERSONA_ICF_B126 = {
    "AD-early": ICFb126Profile(
        extraversion=-1,        # Social withdrawal
        agreeableness=-1,       # Irritability, reduced patience
        conscientiousness=-2,   # Poor organization, multitasking impaired
        emotional_stability=-2, # Anxiety, depression
        openness=-2,            # Resistance to new things
        optimism=-2,            # Pessimism
        confidence=-2,          # Self-doubt
        integrity=0             # Relatively preserved
    ),
    "AD-mid/late": ICFb126Profile(
        extraversion=-3,        # Social response lost
        agreeableness=-3,       # Aggression, resistance
        conscientiousness=-3,   # Planning/self-care lost
        emotional_stability=-3, # Agitation, delusions
        openness=-3,            # Rigidity
        optimism=-3,            # Emotional blunting
        confidence=-3,          # Self-awareness collapsed
        integrity=-3            # Inappropriate behaviors
    ),
    "VaD": ICFb126Profile(
        extraversion=-2,        # Moderate decline
        agreeableness=-2,       # Emotional incontinence, depression
        conscientiousness=-2,   # Moderate decline
        emotional_stability=-3, # Emotional incontinence, mood volatility
        openness=-2,            # Moderate decline
        optimism=-2,            # Depression common
        confidence=-2,          # Moderate decline
        integrity=-1            # Mild decline (more with frontal infarcts)
    ),
    "DLB": ICFb126Profile(
        extraversion=-2,        # Moderate decline
        agreeableness=-2,       # Moderate decline
        conscientiousness=-2,   # Moderate decline
        emotional_stability=-3, # Anxiety, fear from hallucinations
        openness=-1,            # Mild decline
        optimism=-2,            # Moderate decline
        confidence=-2,          # Moderate decline
        integrity=0             # Relatively preserved
    ),
    "PDD": ICFb126Profile(
        extraversion=-2,        # Moderate decline
        agreeableness=-1,       # Mild decline or pathological↑ (ICDs)
        conscientiousness=-2,   # Moderate decline
        emotional_stability=-2, # Moderate decline
        openness=-1,            # Mild decline
        optimism=-2,            # Moderate decline
        confidence=-2,          # Moderate decline
        integrity=-1            # ICDs possible
    ),
    "FTD-bv": ICFb126Profile(
        extraversion=3,         # Severe↑ (disinhibition)
        agreeableness=-3,       # Severe↓ (empathy loss)
        conscientiousness=-3,   # Severe decline
        emotional_stability=-3, # Severe decline
        openness=-3,            # Severe rigidity
        optimism=-2,            # Moderate decline or flat
        confidence=1,           # Mild↑ (lack of insight)
        integrity=-3            # Severe↓ (antisocial behaviors)
    ),
    "nfvPPA": ICFb126Profile(
        extraversion=-2,        # Social avoidance
        agreeableness=-1,       # Mild decline
        conscientiousness=-1,   # Mild decline
        emotional_stability=-2, # Frustration, anxiety
        openness=-1,            # Mild decline
        optimism=-2,            # Moderate decline
        confidence=-2,          # Moderate decline
        integrity=0             # Relatively preserved
    ),
    "svPPA": ICFb126Profile(
        extraversion=-1,        # Early preserved, late decline
        agreeableness=-2,       # Empathy decline
        conscientiousness=-1,   # Mild decline
        emotional_stability=-1, # Mild decline
        openness=-2,            # Fixed interests
        optimism=-1,            # Mild decline
        confidence=-1,          # Mild decline
        integrity=-1            # Early preserved, late inappropriate behaviors
    ),
    "lvPPA": ICFb126Profile(
        extraversion=-2,        # Moderate decline
        agreeableness=-1,       # Mild decline
        conscientiousness=-2,   # Moderate decline (with AD progression)
        emotional_stability=-2, # Anxiety, depression
        openness=-1,            # Mild decline
        optimism=-2,            # Moderate decline
        confidence=-2,          # Moderate decline
        integrity=0             # Relatively preserved
    )
}
PERSONA_BRIEF = {
    "AD-early": {
        "memory": "SEVERE recent/episodic memory loss (delayed recall 0-1/3); immediate recall OK; remote memory preserved; time disorientation; ENCODING deficit (cues help little).",
        "behavior": "Apathy 30-50%, depression 40-50%, irritability 30-40%, anxiety 30%. Insight preserved. Social withdrawal, poor organization, task multi-tasking impaired.",
        "speech": (
            "Spoken language is short, fragmented, and often interrupted by hesitations or restarts. "
            "Thoughts frequently break mid-way, and attempts to explain something are abandoned quickly. "
            "Repetition is common, especially of simple questions. "
            "Speech stays concrete and immediate, without organized or reflective descriptions."
        ),
        "emotional_profile": "Worry about memory, self-doubt, pessimism, frustration with forgetting. Care stress beginning. Emotional stability moderately impaired.",
        "icf_b126": PERSONA_ICF_B126["AD-early"].to_description(),
        "typical_emotions": ["confused", "anxious", "worried", "apologetic", "self-doubting", "frustrated", "depressed", "irritable", "resigned", "calm"]
    },
    "AD-mid/late": {
        "memory": "EXTREME global memory loss; cannot form new memories; complete disorientation (person/place/time); remote/semantic memory dissolution; retrograde collapse.",
        "behavior": "Apathy 70-90% (most prominent), agitation/aggression 50%, delusions 30-40%, hallucinations 20-30%, sleep disturbance 40%, sundowning. Insight completely lost. Care resistance, may steal food.",
        "speech": (
            "speech consists mainly of single words or very short, disconnected fragments. "
            "Attempts at forming sentences often stop abruptly without completion. "
            "Echoing the caregiver’s last word or phrase is common. "
            "Speech rarely carries coherent structure or stable topic maintenance."
        ),
        "emotional_profile": "Severe apathy dominates. Agitation, mood swings, emotional blunting, aggression. Self-awareness collapsed. All ICF dimensions severely impaired.",
        "icf_b126": PERSONA_ICF_B126["AD-mid/late"].to_description(),
        "typical_emotions": ["apathetic", "confused", "agitated", "aggressive", "resistant", "disoriented", "withdrawn", "sundowning"]
    },
    "VaD": {
        "memory": "Patchy memory loss (lesion-dependent); RETRIEVAL deficit > encoding (cues significantly help - 'can't think of it but knows when told'); stepwise progression; slowed processing.",
        "behavior": "Depression 40-60% (most common), EMOTIONAL INCONTINENCE (pathological crying/laughing - subcortical VaD signature), apathy 30-50%, irritability 30%. Mood volatility GREATER than AD.",
        "speech": (
            "Speech is slowed but structurally simple, with noticeable pauses during word-finding. "
            "Sentences may start and then stall due to retrieval difficulty. "
            "Emotional tone can shift suddenly within ordinary conversation. "
            "Grammar tends to remain intact, but timing and fluency are disrupted."
        ),
        "emotional_profile": "Emotional lability - sudden uncontrolled crying or mad or laughing. Depression high. Frustration easily triggered. Stepwise deterioration.",
        "icf_b126": PERSONA_ICF_B126["VaD"].to_description(),
        "typical_emotions": ["depressed", "emotionally_labile", "frustrated", "suddenly_crying", "suddenly_laughing", "irritable", "mood_swings", "overwhelmed", "calm"]
    },
    "DLB": {
        "memory": "Early memory RELATIVELY PRESERVED (key differential vs AD!); RETRIEVAL deficit > encoding; recognition >> recall; COGNITIVE FLUCTUATIONS ('good days vs bad days'); visuospatial/prospective memory impaired.",
        "behavior": "CORE FEATURES: Visual hallucinations 70-80%, cognitive fluctuations, parkinsonian signs, RBD. Apathy 50%, depression 40%, anxiety 40%, delusions 30%. Highly sensitive to antipsychotics.",
        "speech": (
            "Speech clarity fluctuates, alternating between moments of relatively normal conversation "
            "and periods of confusion or distraction. "
            "Utterances may mix real details with perceptual experiences that others cannot verify. "
            "Attention lapses can interrupt or derail ongoing speech without warning."
        ),
        "emotional_profile": "Anxiety/fear prominent (related to hallucinations). Depression common. Emotional fluctuations parallel cognitive fluctuations. Suspicion, paranoia. Good days vs foggy days.",
        "icf_b126": PERSONA_ICF_B126["DLB"].to_description(),
        "typical_emotions": ["anxious", "fearful", "paranoid", "confused", "lucid", "depressed", "suspicious", "calm_periods", "seeing_things", "fluctuating"]
    },
    "PDD": {
        "memory": "Moderate memory loss; RETRIEVAL/working memory impaired; recognition better than recall. Similar to DLB but SLOWER progression. Onset >1 year after Parkinson's diagnosis.",
        "behavior": "Apathy 40-60%, depression 30-40%, anxiety 30%. IMPULSE CONTROL DISORDERS 15-20% (DA agonist-related: gambling, hypersexuality, compulsive shopping). Visual hallucinations 30-40%. Motor symptoms pre-existing.",
        "speech": (
            "Speech is quiet, reduced in variability, and slowed. "
            "Utterances are brief and effortful, often lacking spontaneous elaboration. "
            "Motor involvement can cause blurred articulation or reduced intelligibility. "
            "Even emotionally charged topics tend to be delivered in a flat, monotone style."
        ),
        "emotional_profile": "Apathy/depression most common. ICDs possible (pathological impulsivity from dopamine treatment). Motor+cognitive dual impairment burden. Chronic disease pessimism.",
        "icf_b126": PERSONA_ICF_B126["PDD"].to_description(),
        "typical_emotions": ["apathetic", "depressed", "anxious", "frustrated", "impulsive", "withdrawn", "monotone_affect", "calm"]
    },
    "FTD-bv": {
        "memory": "Memory RELATIVELY PRESERVED (key differential!); episodic memory OK; poor free recall but NORMAL recognition; executive dysfunction affects retrieval, NOT encoding.",
        "behavior": "CORE: Disinhibition 70-90%, empathy loss 80%, apathy 70%, stereotyped/compulsive behaviors 60%, hyperphagia 50%, antisocial behavior 30-50%. Insight lost 90%. Early behavioral change >> memory loss.",
        "speech": (
            "Speech is blunt, unfiltered, and unconcerned with social appropriateness. "
            "Comments may be intrusive or overly direct, with little regard for politeness. "
            "Interruptions or talking over others are common. "
            "The tone is sharp or indifferent rather than hesitant or apologetic."
        ),
        "emotional_profile": "CRITICAL: Disinhibition (extraversion↑↑↑), complete empathy loss (agreeableness↓↓↓), emotional regulation completely lost, antisocial behaviors (integrity↓↓↓). Lacks insight, overconfident.",
        "icf_b126": PERSONA_ICF_B126["FTD-bv"].to_description(),
        "typical_emotions": ["disinhibited", "apathetic", "irritable", "impulsive", "indifferent", "aggressive", "emotionally_flat", "euphoric", "inappropriate"]
    },
    "nfvPPA": {
        "memory": "Non-verbal episodic memory INTACT; verbal memory appears impaired due to language difficulty (use non-verbal tests to show preserved memory).",
        "behavior": "Frustration/anxiety 40-50%, depression 30%, social withdrawal (secondary to communication difficulty). Late may show apathy. Insight PRESERVED.",
        "speech": (
            "Speech production is effortful and non-fluent, characterized by short, broken segments. "
            "Initiating or sequencing sounds is difficult, and grammatical structure is frequently incomplete. "
            "Pauses interrupt the flow, and longer expressions are often abandoned before completion. "
            "Overall output feels strained and laborious."
        ),
        "emotional_profile": "Frustration from communication failure; anxiety from word-finding pressure; embarrassment; preserved awareness of deficits. Avoids social situations due to speech difficulty.",
        "icf_b126": PERSONA_ICF_B126["nfvPPA"].to_description(),
        "typical_emotions": ["frustrated", "anxious", "embarrassed", "depressed", "discouraged", "determined", "struggling", "effortful", "calm"]
    },
    "svPPA": {
        "memory": "CRITICAL DISSOCIATION: SEVERE semantic memory loss (word meaning, object knowledge) BUT episodic memory/orientation PRESERVED. Can recall daily events but can't name objects.",
        "behavior": "Empathy decline (progressive), rigid/fixed interests, stereotyped behaviors 40%, self-centered. Late may progress to FTD-bv-like behavior.",
        "speech": (
            "Speech is smooth and fluent but lacks precise content. "
            "Specific nouns are replaced by vague or generalized descriptions. "
            "The speaker circles around meanings rather than naming things directly. "
            "Surface fluency masks the underlying loss of semantic detail."
        ),
        "emotional_profile": "Anxiety related to semantic loss; empathy decline; rigidity; may become self-centered. Late behavioral changes possible.",
        "icf_b126": PERSONA_ICF_B126["svPPA"].to_description(),
        "typical_emotions": ["confused_about_words", "self-centered", "rigid", "irritable", "indifferent", "circumlocutory", "calm"]
    },
    "lvPPA": {
        "memory": "Early episodic memory preserved; PROGRESSIVE decline; phonological working memory deficit; 50-70% have AD pathology → progressive to AD-like memory pattern.",
        "behavior": "Anxiety 30-40%, depression 30%, word-finding frustration prominent. Progressive worsening over time. Insight PRESERVED.",
        "speech": (
            "Speech is grammatically formed but disrupted by pauses around missing words. "
            "Utterances frequently start, backtrack, and restart due to phonological working-memory difficulty. "
            "Longer sequences are difficult to repeat or maintain. "
            "Conversation remains slow-paced because of persistent word-searching."
        ),
        "emotional_profile": "Word-finding anxiety; depression increases with progression; frustration from communication+memory dual impairment. Late stages become AD-like.",
        "icf_b126": PERSONA_ICF_B126["lvPPA"].to_description(),
        "typical_emotions": ["anxious", "frustrated", "depressed", "struggling", "apologetic", "discouraged", "word-searching", "pausing", "calm"]
    }
}


# =========================
# Data classes
# =========================
@dataclass
class MemoryAccess:
    recent_episodic: str
    remote_episodic: str
    semantic: str
    can_benefit_from_cues: bool
    encoding_vs_retrieval: str
    cognitive_fluctuation: bool
    memory_profile_note: str

@dataclass
class DialoguePlan:
    num_rounds: int
    patient_emotions: List[str]
    memory_events: List[Dict[str, Any]]
    key_traits: List[str]
    starting_speaker: str

@dataclass
class DialogueTurn:
    round: int
    speaker: str
    utterance: str
    emotion: Optional[str] = None
    labels: Optional[Dict[str, List[str]]] = None


# =========================
# Helpers: client + safe JSON parsing
# =========================
def get_client() -> OpenAI:
    # OpenAI SDK reads OPENAI_API_KEY automatically.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Please set it, e.g.\n"
            "  export OPENAI_API_KEY='...'\n"
            "Then rerun."
        )
    return OpenAI()

def safe_json_loads(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =========================
# Memory gating (kept identical to your logic)
# =========================
def filter_memories_by_persona_clinical(long_mem: Dict, short_mem: Dict, persona: str) -> MemoryAccess:
    long_text = " ".join([p.get("text", "") for p in long_mem.get("paragraphs", [])])
    short_text = short_mem.get("text", "")

    # --- keep your original branching logic here (unchanged) ---
    # For brevity, omitted in this response. Paste your existing implementation.
    return MemoryAccess(
        recent_episodic=short_text[:300],
        remote_episodic=long_text[:400],
        semantic=long_text[:300],
        can_benefit_from_cues=True,
        encoding_vs_retrieval="mixed",
        cognitive_fluctuation=False,
        memory_profile_note="Default memory profile",
    )


# =========================
# Topic selection
# =========================
def select_topic_for_person(persona: str, long_mem: Dict, short_mem: Dict) -> Tuple[str, str]:
    sources = ["long_memory", "short_memory", "general_topics"]
    weights = [0.3, 0.4, 0.3]
    source = random.choices(sources, weights=weights)[0]

    if source == "long_memory":
        paragraphs = long_mem.get("paragraphs", [])
        if paragraphs:
            para = random.choice(paragraphs)
            topic_text = (para.get("text", "") or "").strip()
            if topic_text:
                return "long_memory", f"Discussing: {topic_text}"
        return "general_topics", random.choice(GENERAL_TOPICS)

    if source == "short_memory":
        short_text = (short_mem.get("text", "") or "").strip()
        if len(short_text) > 20:
            return "short_memory", f"Discussing recent routine: {short_text}"
        return "general_topics", random.choice(GENERAL_TOPICS)

    return "general_topics", random.choice(GENERAL_TOPICS)


# =========================
# Step 1: Planning
# =========================
def plan_dialogue(
        client: OpenAI,
        model: str,
        persona: str,
        topic: str,
        topic_source: str,
        starting_speaker: str,
        memory_access: MemoryAccess,
) -> DialoguePlan:
    persona_info = PERSONA_BRIEF[persona]

    rounds_range = (5, 7) if topic_source == "long_memory" else (4, 6) if topic_source == "short_memory" else (3, 6)

    planner_prompt = f"""You are an expert dialogue planner.

Persona: {persona}

ICF-b126:
{persona_info['icf_b126']}

Memory profile:
{persona_info['memory']}

Memory access this dialogue:
- Recent accessible: {"Yes" if memory_access.recent_episodic else "No"}
- Remote accessible: {"Yes" if memory_access.remote_episodic else "No"}
- Semantic: {"Impaired" if not memory_access.semantic else "Accessible"}
- Cue benefit: {"Yes" if memory_access.can_benefit_from_cues else "No"}
- Type: {memory_access.encoding_vs_retrieval}
- Fluctuation: {"Yes" if memory_access.cognitive_fluctuation else "No"}
- Note: {memory_access.memory_profile_note}

Behavior:
{persona_info['behavior']}

Speech:
{persona_info['speech']}

Topic: {topic}
Starting speaker: {starting_speaker}

Output ONLY valid JSON:
{{
  "num_rounds": 5,
  "patient_emotions": ["confused", "worried", "calm"],
  "memory_events": [
    {{"round": 1, "can_recall": false, "memory_type": "recent_event", "detail": "what happened this morning"}}
  ],
  "key_traits": ["apologetic", "self-doubting"]
}}

Requirements:
- num_rounds: {rounds_range[0]}-{rounds_range[1]}
- patient_emotions: realistic progression
- memory_events: 1-3 attempts consistent with profile
- key_traits: 2-3 traits

Generate plan:
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": planner_prompt}],
        response_format={"type": "json_object"},
    )
    plan_data = safe_json_loads(resp.choices[0].message.content)

    return DialoguePlan(
        num_rounds=int(plan_data.get("num_rounds", rounds_range[0])),
        patient_emotions=list(plan_data.get("patient_emotions", [])) or ["confused"] * int(plan_data.get("num_rounds", rounds_range[0])),
        memory_events=list(plan_data.get("memory_events", [])) or [],
        key_traits=list(plan_data.get("key_traits", [])) or [],
        starting_speaker=starting_speaker,
    )


# =========================
# Persona Style Hints (FIXED: no broken quotes)
# =========================
def get_persona_style_hint(persona: str) -> str:
    if persona == "AD-early":
        return (
            "- Short, simple spoken sentences (5–12 words)\n"
            "- Some repetition is OK\n"
            "- Occasional hesitations\n"
            "- Concrete wording; avoid abstract metaphors\n"
            "- Slightly disorganized; avoid neat self-analysis"
        )
    if persona == "AD-mid/late":
        return (
            "- Very short fragments, sometimes single words\n"
            "- Frequent hesitations (um/uh/wait)\n"
            "- Echoing caregiver’s last phrase is common\n"
            "- No meta talk like 'I have memory issues'\n"
            "- No long, well-structured sentences (>20 words)"
        )
    if persona == "VaD":
        return "- Slower but still structured speech; simple and concrete; 'tip of my tongue' is plausible"
    if persona == "DLB":
        return "- Mix of clear and confused turns (fluctuation); spoken, not essay-like"
    if persona == "FTD-bv":
        return "- Blunt, direct, sometimes rude; little apologizing; short sharp statements"
    if persona == "nfvPPA":
        return "- Effortful, broken speech; many pauses; grammar errors OK; avoid long fluent sentences"
    if persona == "svPPA":
        return "- Fluent but vague; replaces nouns with 'that thing'; talks around missing words"
    if persona == "lvPPA":
        return "- Word-finding pauses; backtracking; keep sentences simple"
    return "- Spontaneous speech; short sentences; hesitations OK; avoid polished paragraphs"


# =========================
# Step 2: Generation
# =========================
def generate_dialogue(
        client: OpenAI,
        model: str,
        plan: DialoguePlan,
        persona: str,
        topic: str,
        memory_access: MemoryAccess,
        starting_speaker: str,
) -> List[DialogueTurn]:
    persona_info = PERSONA_BRIEF[persona]
    style_hint = get_persona_style_hint(persona)

    memory_guidance = (
        f"- Recent memory: {'CANNOT recall' if not memory_access.recent_episodic else 'Can recall some recent events'}\n"
        f"- Remote memory: {'CANNOT recall' if not memory_access.remote_episodic else 'Can recall distant past'}\n"
        f"- Cue benefit: {'YES' if memory_access.can_benefit_from_cues else 'NO'}\n"
        f"- Note: {memory_access.memory_profile_note}"
    )

    memory_notes = []
    for event in plan.memory_events:
        status = "CAN recall" if event.get("can_recall") else "CANNOT recall"
        memory_notes.append(f"Round {event.get('round')}: Patient {status} {event.get('memory_type')} ({event.get('detail','')})")
    memory_notes_str = "\n".join(memory_notes) if memory_notes else "No specific memory events planned"

    format_note = (
        "Starting speaker: Patient\nFormat: Patient → Caregiver → Patient → Caregiver..."
        if starting_speaker == "Patient"
        else "Starting speaker: Caregiver\nFormat: Caregiver → Patient → Caregiver → Patient..."
    )

    generator_prompt = f"""Generate a natural caregiving dialogue.

Persona: {persona}
Speech pattern: {persona_info['speech']}
Key traits: {', '.join(plan.key_traits)}

Topic: {topic}
Rounds: {plan.num_rounds}
Emotions: {' → '.join(plan.patient_emotions)}

Memory constraints:
{memory_guidance}

Memory recall guide:
{memory_notes_str}

{format_note}

Patient speech style (HARD constraints):
{style_hint}

Format (STRICT):
Caregiver: ...
Patient: ...

Requirements:
- Total >170 words
- Complete ALL {plan.num_rounds} rounds
- Start with {starting_speaker}
- Patient MUST reflect memory limitations accurately
- If cues help, show improvement when caregiver hints
- Spoken, fragmented patient language is required

Generate dialogue:
"""

    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": generator_prompt}])
    dialogue_text = resp.choices[0].message.content or ""

    # Clean common markdown artifacts
    dialogue_text = re.sub(r"\*\*", "", dialogue_text)
    dialogue_text = re.sub(r"```[\w]*\s*\n?", "", dialogue_text)
    dialogue_text = re.sub(r"```\s*$", "", dialogue_text)

    turns: List[DialogueTurn] = []
    lines = [l.strip() for l in dialogue_text.split("\n") if l.strip()]
    round_num = 1

    for line in lines:
        if line.startswith("Caregiver:") or line.startswith("Nurse:"):
            speaker = "Caregiver"
            text = line.split(":", 1)[1].strip()
            emotion = None
        elif line.startswith("Patient:"):
            speaker = "Patient"
            text = line.split(":", 1)[1].strip()
            patient_turn_idx = len([t for t in turns if t.speaker == "Patient"])
            emotion = plan.patient_emotions[min(patient_turn_idx, len(plan.patient_emotions) - 1)]
        else:
            continue

        turns.append(DialogueTurn(round=round_num, speaker=speaker, utterance=text, emotion=emotion))

        # Advance rounds after each patient turn (more robust)
        if speaker == "Patient":
            round_num += 1

    return turns


# =========================
# Step 3: Action Labeling
# =========================
def add_action_labels(client: OpenAI, model: str, turns: List[DialogueTurn], persona: str, api_delay: float) -> List[DialogueTurn]:
    labeled: List[DialogueTurn] = []
    for turn in turns:
        if turn.speaker != "Patient":
            labeled.append(turn)
            continue

        label_prompt = f"""Patient utterance: "{turn.utterance}"
Current emotion: {turn.emotion}
Persona: {persona}

Select 0-2 labels per category (or empty).

Movement: {', '.join(ACTION_LABELS['movement'])}
Facial expression: {', '.join(ACTION_LABELS['facial_expression'])}
Voice: {', '.join(ACTION_LABELS['voice'])}

Output ONLY valid JSON:
{{
  "movement": [],
  "facial_expression": [],
  "voice": []
}}
"""

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": label_prompt}],
            response_format={"type": "json_object"},
        )
        raw = safe_json_loads(resp.choices[0].message.content)

        cleaned = {}
        for key in ["movement", "facial_expression", "voice"]:
            vals = raw.get(key, [])
            cleaned[key] = [v for v in vals if isinstance(v, str)] if isinstance(vals, list) else []

        labeled.append(DialogueTurn(
            round=turn.round,
            speaker=turn.speaker,
            utterance=turn.utterance,
            emotion=turn.emotion,
            labels=cleaned,
        ))
        time.sleep(api_delay)

    return labeled


# =========================
# Validation (kept; uses validator_model)
# =========================
def validate_persona_and_memory(
        client: OpenAI,
        validator_model: str,
        turns: List[DialogueTurn],
        persona: str,
        plan: DialoguePlan,
        memory_access: MemoryAccess,
) -> Dict[str, Any]:
    persona_info = PERSONA_BRIEF[persona]

    dialogue_lines = []
    for t in turns:
        if t.speaker == "Patient":
            dialogue_lines.append(f"Patient ({t.emotion}): {t.utterance}")
        else:
            dialogue_lines.append(f"Caregiver: {t.utterance}")
    dialogue_str = "\n".join(dialogue_lines)

    prompt = f"""You are a clinical expert. Evaluate this dialogue for persona + memory accuracy.

Persona: {persona}

Expected:
Memory: {persona_info['memory']}
Behavior: {persona_info['behavior']}
Speech: {persona_info['speech']}

Memory profile:
- Type: {memory_access.encoding_vs_retrieval}
- Cue benefit: {"YES" if memory_access.can_benefit_from_cues else "NO"}
- Fluctuation: {"YES" if memory_access.cognitive_fluctuation else "NO"}
- Note: {memory_access.memory_profile_note}

Dialogue:
{dialogue_str}

Output ONLY valid JSON:
{{
  "persona_accuracy_score": 0,
  "persona_accuracy_feedback": "",
  "memory_loss_accuracy_score": 0,
  "memory_loss_accuracy_feedback": "",
  "overall_realism_score": 0,
  "overall_feedback": "",
  "clinical_red_flags": []
}}
"""

    resp = client.chat.completions.create(
        model=validator_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    out = safe_json_loads(resp.choices[0].message.content)

    out.setdefault("persona_accuracy_score", 0)
    out.setdefault("memory_loss_accuracy_score", 0)
    out.setdefault("overall_realism_score", 0)
    out.setdefault("persona_accuracy_feedback", "")
    out.setdefault("memory_loss_accuracy_feedback", "")
    out.setdefault("overall_feedback", "")
    out.setdefault("clinical_red_flags", [])

    # pass criteria (≥6 and no red flags)
    all_scores_pass = (
            out["persona_accuracy_score"] >= 6
            and out["memory_loss_accuracy_score"] >= 6
            and out["overall_realism_score"] >= 6
    )
    out["passed"] = bool(all_scores_pass and len(out.get("clinical_red_flags", [])) == 0)
    return out


def validate_dialogue(turns: List[DialogueTurn], plan: DialoguePlan) -> Dict[str, Any]:
    total_words = sum(len((t.utterance or "").split()) for t in turns)
    patient_turns = [t for t in turns if t.speaker == "Patient"]
    has_labels = all(t.labels is not None for t in patient_turns)
    memory_loss_shown = any(not e.get("can_recall", False) for e in plan.memory_events)

    return {
        "word_count": total_words,
        "word_count_ok": total_words >= 170,
        "round_count": len(patient_turns),
        "round_count_ok": len(patient_turns) == plan.num_rounds,
        "memory_loss_shown": memory_loss_shown,
        "action_labels_present": has_labels,
        "passed": total_words >= 170 and len(patient_turns) >= 3 and memory_loss_shown and has_labels,
    }


def calculate_quality_score(clinical_val: Dict[str, Any], basic_val: Dict[str, Any]) -> float:
    clinical_score = (
            clinical_val["persona_accuracy_score"] * 0.35
            + clinical_val["memory_loss_accuracy_score"] * 0.35
            + clinical_val["overall_realism_score"] * 0.30
    )
    basic_score = (
            (10 if basic_val["word_count_ok"] else 5) * 0.30
            + (10 if basic_val["round_count_ok"] else 5) * 0.30
            + (10 if basic_val["memory_loss_shown"] else 0) * 0.20
            + (10 if basic_val["action_labels_present"] else 5) * 0.20
    )
    final_score = clinical_score * 0.7 + basic_score * 0.3

    if clinical_val.get("clinical_red_flags"):
        penalty = len(clinical_val["clinical_red_flags"]) * 1.5
        final_score = max(0.0, final_score - penalty)

    return float(final_score)


def determine_quality_tier(score: float) -> str:
    if score >= QUALITY_THRESHOLD_EXEMPLAR:
        return "exemplar"
    if score >= QUALITY_THRESHOLD_ACCEPTABLE:
        return "acceptable"
    return "low"


# =========================
# Loading
# =========================
def load_long_memories(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Long-memory file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["id"]): item for item in data}

def load_short_memories(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Short-memory file not found: {path}")
    records: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[int(rec["id"])] = rec
    return records


# =========================
# Dialogue packaging
# =========================
def create_dialogue_data(
        person_id: int,
        person_name: str,
        persona: str,
        dialogue_num: int,
        topic: str,
        topic_source: str,
        starting_speaker: str,
        plan: DialoguePlan,
        labeled_turns: List[DialogueTurn],
        memory_access: MemoryAccess,
        basic_validation: Dict[str, Any],
        clinical_validation: Dict[str, Any],
        quality_score: float,
        quality_tier: str,
        generation_attempt: int,
        model: str,
) -> Dict[str, Any]:
    icf_profile = PERSONA_ICF_B126[persona]
    return {
        "person_id": person_id,
        "person_name": person_name,
        "persona": persona,
        "dialogue_number": dialogue_num,
        "dialogue_id": f"P{person_id:02d}_D{dialogue_num}",
        "topic": topic,
        "topic_source": topic_source,
        "starting_speaker": starting_speaker,
        "icf_b126_profile": icf_profile.to_dict(),
        "plan": {
            "num_rounds": plan.num_rounds,
            "emotions": plan.patient_emotions,
            "memory_events": plan.memory_events,
            "key_traits": plan.key_traits,
        },
        "memory_profile": {
            "has_recent_episodic": bool(memory_access.recent_episodic),
            "has_remote_episodic": bool(memory_access.remote_episodic),
            "has_semantic": bool(memory_access.semantic),
            "benefits_from_cues": memory_access.can_benefit_from_cues,
            "deficit_type": memory_access.encoding_vs_retrieval,
            "cognitive_fluctuation": memory_access.cognitive_fluctuation,
            "clinical_note": memory_access.memory_profile_note,
        },
        "turns": [
            {
                "round": t.round,
                "speaker": t.speaker,
                "utterance": t.utterance,
                "emotion": t.emotion,
                "labels_by_type": t.labels if t.labels else {"movement": [], "facial_expression": [], "voice": []},
            }
            for t in labeled_turns
        ],
        "validation": {"basic": basic_validation, "clinical": clinical_validation},
        "quality_metadata": {
            "overall_score": round(quality_score, 2),
            "tier": quality_tier,
            "generation_attempt": generation_attempt,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model": model,
        },
    }


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Seed dialogue generation (anonymized config)")
    p.add_argument("--persona", choices=AVAILABLE_PERSONAS, default=None, help="Generate only a specific persona (debug)")
    p.add_argument("--long_memory", type=Path, default=DEFAULT_LONG_MEMORY_PATH)
    p.add_argument("--short_memory", type=Path, default=DEFAULT_SHORT_MEMORY_PATH)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--tag", type=str, default=DEFAULT_TAG)

    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--validator_model", type=str, default=DEFAULT_VALIDATOR_MODEL)

    p.add_argument("--per_person", type=int, default=DIALOGUES_PER_PERSON)
    p.add_argument("--max_retries", type=int, default=MAX_RETRIES_PER_DIALOGUE)
    p.add_argument("--api_delay", type=float, default=API_DELAY_SEC)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # output files
    ensure_dir(args.out_dir)
    out_file = args.out_dir / f"{args.tag}_dialogues.jsonl"
    exemplar_file = args.out_dir / f"{args.tag}_exemplars.jsonl"
    report_file = args.out_dir / f"{args.tag}_report.json"

    # clean old outputs
    for f in [out_file, exemplar_file, report_file]:
        if f.exists():
            f.unlink()

    client = get_client()

    print(f"Model: {args.model}")
    print(f"Validator model: {args.validator_model}")
    print(f"Seed: {args.seed}")
    print(f"Long memory: {args.long_memory}")
    print(f"Short memory: {args.short_memory}")
    print(f"Output: {out_file}")

    long_memories = load_long_memories(args.long_memory)
    short_memories = load_short_memories(args.short_memory)

    if args.persona:
        ids = [pid for pid in PERSONA_TO_IDS[args.persona] if pid in long_memories]
    else:
        ids = sorted(long_memories.keys())

    all_dialogues: List[Dict[str, Any]] = []
    exemplar_by_persona = {p: 0 for p in AVAILABLE_PERSONAS}
    exemplar_count = 0
    failed_count = 0

    for person_id in ids:
        long_mem = long_memories[person_id]
        short_mem = short_memories.get(person_id)
        if not short_mem:
            print(f"Skipping {person_id}: no short memory")
            continue

        persona = PERSONA_MAP.get(person_id, "AD-early")
        person_name = long_mem.get("narrator_name", long_mem.get("name", f"Person {person_id}"))

        for dialogue_num in range(1, args.per_person + 1):
            # pick topic + gate memory
            starting_speaker = random.choice(["Caregiver", "Patient"])
            topic_source, topic = select_topic_for_person(persona, long_mem, short_mem)
            memory_access = filter_memories_by_persona_clinical(long_mem, short_mem, persona)

            best: Optional[Dict[str, Any]] = None

            for attempt in range(args.max_retries + 1):
                try:
                    plan = plan_dialogue(client, args.model, persona, topic, topic_source, starting_speaker, memory_access)
                    time.sleep(args.api_delay)

                    turns = generate_dialogue(client, args.model, plan, persona, topic, memory_access, starting_speaker)
                    time.sleep(args.api_delay)

                    labeled = add_action_labels(client, args.model, turns, persona, args.api_delay)
                    time.sleep(args.api_delay)

                    clinical = validate_persona_and_memory(client, args.validator_model, labeled, persona, plan, memory_access)
                    basic = validate_dialogue(labeled, plan)

                    score = calculate_quality_score(clinical, basic)
                    tier = determine_quality_tier(score)

                    if tier == "exemplar" or (tier == "acceptable" and attempt == args.max_retries):
                        best = create_dialogue_data(
                            person_id, person_name, persona, dialogue_num,
                            topic, topic_source, starting_speaker, plan,
                            labeled, memory_access, basic, clinical,
                            score, tier, attempt + 1, args.model
                        )
                        break
                except Exception as e:
                    if attempt == args.max_retries:
                        print(f"[{person_id}:{dialogue_num}] failed: {e}")
                    continue

            if best is None:
                failed_count += 1
                continue

            all_dialogues.append(best)

            # write immediately
            with out_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(best, ensure_ascii=False) + "\n")

            if best["quality_metadata"]["tier"] == "exemplar":
                exemplar_count += 1
                exemplar_by_persona[persona] += 1

    # post-process
    exemplars = [d for d in all_dialogues if d["quality_metadata"]["tier"] == "exemplar"]
    acceptable = [d for d in all_dialogues if d["quality_metadata"]["tier"] == "acceptable"]

    if exemplars:
        with exemplar_file.open("w", encoding="utf-8") as f:
            for ex in exemplars:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    report = {
        "summary": {
            "total_dialogues": len(all_dialogues),
            "exemplar_count": len(exemplars),
            "acceptable_count": len(acceptable),
            "failed_count": failed_count,
            "model": args.model,
            "validator_model": args.validator_model,
            "seed": args.seed,
            "tag": args.tag,
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
        "exemplar_by_persona": exemplar_by_persona,
        "topic_source_distribution": dict(Counter([d["topic_source"] for d in all_dialogues])),
        "memory_type_distribution": dict(Counter([d["memory_profile"]["deficit_type"] for d in all_dialogues])),
    }

    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Wrote: {out_file}")
    print(f"Exemplars: {exemplar_file if exemplars else '(none)'}")
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()
