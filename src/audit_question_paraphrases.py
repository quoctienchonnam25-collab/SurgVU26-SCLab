"""Routing audit over organizer-style question paraphrases.

Motivation: on 2026-08-31 an audit of the 11 official public samples found 3 of
them mishandled - a 27% defect rate - because every detector in this project was
written against generate_qa.py's own synthetic templates and never against the
organizers' real wording. Fixing that was worth a confirmed +0.0446 on the final
leaderboard (0.5526 -> 0.5972, same checkpoint). But 11 questions is a very small
audit set, so this script extends the same check to a much larger set of
paraphrases written in the organizers' observed register.

The reference style, taken from the 11 real samples:
  "Are there forceps being used here?"          "What type of forceps is mentioned?"
  "Is a large needle driver among the listed tools?"  "What organ is being manipulated?"
  "Is a needle driver involved in the procedure?"     "Is tissue being cut during this clip?"
  "Was a large needle driver used during the surgery?"
i.e. short, natural, and using verbs ("mentioned", "involved", "among the listed
tools") that this project's own templates never use.

Each case below declares the routing it MUST get. A mismatch is a defect of the
same class as the "Which forceps type...?" -> "Yes" bug: the container answers in
the wrong FORM, which BERTScore punishes far more than a wrong-but-well-formed
answer. Run this before packaging any container.
"""
import re
import sys

from answer_postprocess import (
    extract_tool_purpose_answer,
    is_forceps_type_question,
    is_organ_question,
    is_procedure_type_question,
    is_tool_purpose_question,
)
from generate_qa import ALL_PRESENCE_TARGETS

TOOL_PRESENCE_WORDS = (
    "list", "install", "installation record", "tool record",
    "use", "present", "involve",
)
TOOL_PRESENCE_NAMES = tuple(
    sorted((name.lower() for name in ALL_PRESENCE_TARGETS), key=len, reverse=True)
)
OPEN_ENDED_PREFIXES = (
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
    "name", "list", "describe", "identify",
)


def _norm(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _is_binary(question):
    return not _norm(question).startswith(OPEN_ENDED_PREFIXES)


def route(question):
    """Mirrors predict_dual_encoder_vqa.answer_question()'s dispatch order."""
    normalized = _norm(question)
    if (
        _is_binary(question)
        and not is_forceps_type_question(question)
        and any(name in normalized for name in TOOL_PRESENCE_NAMES)
        and any(word in normalized for word in TOOL_PRESENCE_WORDS)
    ):
        return "tool_presence"
    if _is_binary(question) and (
        re.search(r"\bcut(ting)?\b", normalized)
        or ("tissue" in normalized and "dissect" in normalized)
    ):
        return "tissue_cutting"
    if _is_binary(question) and "sutur" in normalized and "required" in normalized:
        return "suture_required"
    if is_procedure_type_question(question):
        return "procedure_type"
    if is_tool_purpose_question(question):
        return "tool_purpose" if extract_tool_purpose_answer(question) else "tool_purpose_NO_LOOKUP"
    if is_forceps_type_question(question):
        return "forceps_type"
    if is_organ_question(question):
        return "organ"
    return "free_generation"


# (question, required routing). "free_generation" is correct for description and
# task_id: neither has a defensible deterministic prior, so both are deliberately
# left to the model - see answer_postprocess's module docstring.
CASES = [
    # ── tool_presence: Yes/No, must reach the calibrated binary path ──────────
    ("Are there forceps being used here?", "tool_presence"),
    ("Is a large needle driver among the listed tools?", "tool_presence"),
    ("Was a large needle driver used in this clip?", "tool_presence"),
    ("Is a needle driver involved in the procedure?", "tool_presence"),
    ("Was a large needle driver used during the surgery?", "tool_presence"),
    ("Is a stapler present in this clip?", "tool_presence"),
    ("Was a vessel sealer used here?", "tool_presence"),
    ("Are monopolar curved scissors listed for this clip?", "tool_presence"),
    ("Is a clip applier among the tools used?", "tool_presence"),
    ("Does this clip involve a grasping retractor?", "tool_presence"),
    ("Was force bipolar used at any point?", "tool_presence"),
    ("Is a tip-up fenestrated grasper present?", "tool_presence"),
    ("Are prograsp forceps involved in this step?", "tool_presence"),
    ("Was a permanent cautery hook/spatula used?", "tool_presence"),
    ("Is bipolar forceps listed as installed?", "tool_presence"),

    # ── forceps_type: open-ended, must NOT be answered Yes/No ─────────────────
    ("What type of forceps is mentioned?", "forceps_type"),
    ("Which forceps type is listed as installed for this clip?", "forceps_type"),
    ("Which forceps types are listed as installed for this clip?", "forceps_type"),
    ("Are any forceps types listed as installed for this clip?", "forceps_type"),
    ("What forceps type is used in this clip?", "forceps_type"),
    ("Which type of forceps is present here?", "forceps_type"),
    ("What type of forceps is listed?", "forceps_type"),
    ("Which forceps type is recorded for this clip?", "forceps_type"),

    # ── tissue_cutting ────────────────────────────────────────────────────────
    ("Is tissue being cut during this clip?", "tissue_cutting"),
    ("Is tissue dissection occurring in this clip?", "tissue_cutting"),
    ("Is any cutting happening in this clip?", "tissue_cutting"),
    ("Is tissue cut at any point here?", "tissue_cutting"),
    ("Was tissue being cut during the surgery?", "tissue_cutting"),
    ("Is the surgeon cutting tissue in this clip?", "tissue_cutting"),

    # ── suture_required ───────────────────────────────────────────────────────
    ("Is a suture required in this surgical step?", "suture_required"),
    ("Is suturing required here?", "suture_required"),
    ("Was a suture required during this step?", "suture_required"),

    # ── procedure_type ────────────────────────────────────────────────────────
    ("What procedure is this summary describing?", "procedure_type"),
    ("What type of surgery is being performed?", "procedure_type"),
    ("What kind of surgical procedure is this?", "procedure_type"),

    # ── tool_purpose: must resolve to a purpose string, never NO_LOOKUP ───────
    ("What is the purpose of using forceps in this procedure?", "tool_purpose"),
    ("What is the purpose of using a needle driver in this procedure?", "tool_purpose"),
    ("What is the general purpose of a stapler?", "tool_purpose"),
    ("What role is bipolar forceps intended to serve?", "tool_purpose"),
    ("What is the purpose of the vessel sealer here?", "tool_purpose"),
    ("What is the purpose of using monopolar curved scissors?", "tool_purpose"),
    ("What is the purpose of the clip applier in this procedure?", "tool_purpose"),

    # ── organ ─────────────────────────────────────────────────────────────────
    ("What organ is being manipulated?", "organ"),
    ("What anatomical structure is being operated on?", "organ"),
    ("What organ is involved in this step?", "organ"),
    ("What organ or structure does this exercise represent?", "organ"),
    ("Which anatomical structure is simulated by this training task?", "organ"),
    ("What anatomical structure is this training exercise intended to represent?", "organ"),

    # ── deliberately unrouted ─────────────────────────────────────────────────
    ("What task is the surgeon performing?", "free_generation"),
    ("What surgical step is being performed?", "free_generation"),
    ("What is the current surgical task?", "free_generation"),
    ("Describe what is happening in this surgical clip.", "free_generation"),
    ("What is the surgeon doing in this step?", "free_generation"),
    ("What activity is the surgeon performing?", "free_generation"),
]


def main():
    failures = []
    for question, expected in CASES:
        actual = route(question)
        if actual != expected:
            failures.append((question, expected, actual))

    print(f"checked {len(CASES)} organizer-style paraphrases")
    if not failures:
        print("PASS - every question routes as required")
        return 0

    print(f"FAIL - {len(failures)} misrouted:\n")
    for question, expected, actual in failures:
        print(f"  {question}")
        print(f"      expected {expected!r}, got {actual!r}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
