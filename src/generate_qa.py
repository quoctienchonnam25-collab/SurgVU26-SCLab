"""
SurgVU 2026 Category 2 - Expanded QA Generator v2
Generates 9+ question types matching the public sample's test cases (case122-132).
Includes rebalancing for Yes/No and task distribution.
"""
import json
import os
import re
import random
from collections import defaultdict, Counter
from tqdm import tqdm

# ─── Tool metadata ────────────────────────────────────────────────────────────

TOOL_SINGULAR_PLURAL = {
    "needle driver":                ("needle driver", "needle drivers"),
    "monopolar curved scissors":    ("monopolar curved scissors", "scissors"),
    "force bipolar":                ("force bipolar", "force bipolars"),
    "clip applier":                 ("clip applier", "clip appliers"),
    "cadiere forceps":              ("cadiere forceps", "forceps"),
    "bipolar forceps":              ("bipolar forceps", "forceps"),
    "vessel sealer":                ("vessel sealer", "vessel sealers"),
    "permanent cautery hook/spatula": ("permanent cautery hook/spatula", "cautery hooks"),
    "prograsp forceps":             ("prograsp forceps", "forceps"),
    "stapler":                      ("stapler", "staplers"),
    "grasping retractor":           ("grasping retractor", "retractors"),
    "tip-up fenestrated grasper":   ("tip-up fenestrated grasper", "graspers"),
}

# Aliases mapping common public-sample variants → canonical name
TOOL_ALIASES = {
    "large needle driver": "needle driver",
    "needle driver":       "needle driver",
}

# groundtruth_toolname (used above) collapses several distinct commercial variants
# tracked separately in tools.csv's commercial_toolname column - e.g. "needle driver"
# covers "Large Needle Driver", "Mega Needle Driver", "Large SutureCut Needle Driver"
# and "Mega SutureCut Needle Driver", which are NOT interchangeable. The organizers
# confirmed the public sample's test questions probe this level of detail (e.g. "was a *large* needle
# driver used"). preprocess.py now also records these as each segment's
# 'commercial_tools' set; COMMERCIAL_VARIANTS lists the known variants per generic
# tool so a question can be asked - and answered - at that specific granularity
# instead of silently checking the generic category (the root cause of the
# case126/132 failures: the question named a specific variant, but the answer was
# keyed off "is any needle driver present" regardless of which one).
COMMERCIAL_VARIANTS = {
    "needle driver": [
        "large suturecut needle driver",
        "large needle driver",
        "mega needle driver",
        "mega suturecut needle driver",
    ],
    "clip applier": ["large clip applier", "small clip applier"],
    "bipolar forceps": ["maryland bipolar forceps", "fenestrated bipolar forceps"],
    "stapler": [
        "sureform stapler 60", "stapler 45", "sureform stapler 45",
        "stapler 30 curved-tip", "stapler 45 curved-tip",
    ],
}

# Tool → purpose (for "What is the purpose of using X?" questions)
TOOL_PURPOSE = {
    "needle driver":                "to hold and drive needles during suturing",
    "monopolar curved scissors":    "to cut and dissect tissue using electrical energy",
    "force bipolar":                "to grasp tissue and provide hemostasis using bipolar energy",
    "clip applier":                 "to apply clips for ligation of vessels or ducts",
    "cadiere forceps":              "to grasp and hold tissues or objects during the surgery",
    "bipolar forceps":              "to grasp tissue and coagulate using bipolar energy",
    "vessel sealer":                "to seal and divide blood vessels",
    "permanent cautery hook/spatula": "to dissect and cauterize tissue",
    "prograsp forceps":             "to grasp and retract tissues or objects",
    "stapler":                      "to staple and divide tissue",
    "grasping retractor":           "to retract and expose the surgical field",
    "tip-up fenestrated grasper":   "to grasp tissue with a fenestrated jaw for better grip",
}

# Task → organ mapping
TASK_ORGAN = {
    "Uterine horn":                  "uterine horn",
    "Rectal artery/vein":            "rectal artery and vein",
    "Suspensory ligaments":          "suspensory ligaments",
    "Skills application":            "gallbladder",
    "Suturing":                      None,  # generic, no specific organ
    "Range of motion":               None,
    "Retraction and collision avoidance": None,
    "Other":                         None,
}

FORCEPS_TOOLS = ["cadiere forceps", "bipolar forceps", "prograsp forceps"]

ALL_TARGET_TOOLS = list(TOOL_SINGULAR_PLURAL.keys())

# ─── Question generators ──────────────────────────────────────────────────────

def _sp(tool):
    return TOOL_SINGULAR_PLURAL.get(tool, (tool, tool + "s"))


def gen_tool_presence(active_tools, target_tool=None, active_commercial_tools=None,
                       force_variant=None, force_generic=False):
    """Public sample examples: case122, 123, 126, 128, 132.

    About half the time, if target_tool has known commercial variants (see
    COMMERCIAL_VARIANTS), asks about one specific variant and checks presence
    against active_commercial_tools - NOT the generic active_tools - so "was a
    large needle driver used" is answered correctly even when a *different*
    needle driver variant (e.g. Mega SutureCut) is the one actually present.

    The returned balance_key is the generic tool name for generic-mode questions,
    but the SPECIFIC variant string (e.g. "large clip applier") for variant-mode
    questions - self-distillation error analysis on v5 found the model
    false-positives specifically on variant questions (e.g. "was a maryland
    bipolar forceps used?" -> answered Yes because *some* bipolar forceps is
    almost always present), a bias invisible to balancing that only tracked the
    generic key, since it averaged away against the differently-skewed
    generic-mode questions for the same tool. force_variant/force_generic let
    balance_tool_presence request a specific mode instead of the random 50/50
    split, so it can top up exactly the (tool, mode) pair that's actually skewed.
    """
    active_commercial_tools = active_commercial_tools or set()

    if force_variant is not None:
        display_name = force_variant
        p = display_name + "s"
        is_present = display_name in active_commercial_tools
        balance_key = display_name
    else:
        if target_tool is None:
            target_tool = random.choice(ALL_TARGET_TOOLS)
        _resolved_tool = target_tool

        variants = COMMERCIAL_VARIANTS.get(target_tool)
        use_variant = (not force_generic) and variants and random.random() < 0.5
        if use_variant:
            display_name = random.choice(variants)
            is_present = display_name in active_commercial_tools
            p = display_name + "s"
            balance_key = display_name
        else:
            display_name, p = _sp(target_tool)
            is_present = target_tool in active_tools
            balance_key = target_tool

    q_templates = [
        f"Is a {display_name} among the listed tools?",
        f"Is a {display_name} being used here?",
        f"Are there {p} being used here?",
        f"Is there a {display_name} present?",
        f"Was a {display_name} used during the surgery?",
        f"Was a {display_name} used in this clip?",
        f"Is a {display_name} involved in the procedure?",
    ]
    q = random.choice(q_templates)

    if is_present:
        answers = [
            "Yes",
            f"Yes, a {display_name} is listed.",
            f"Yes, a {display_name} is being used.",
            f"Yes, a {display_name} was utilized.",
            f"Yes, the procedure involved a {display_name}.",
        ]
    else:
        answers = [
            "No",
            f"No, a {display_name} was not used.",
            f"No, a {display_name} is not listed.",
            f"No, there's no {display_name}.",
            f"There is no indication a {display_name} was used.",
        ]
    return q, answers, balance_key


def gen_forceps_type(active_tools):
    """Public sample example: case124"""
    active_forceps = [t for t in active_tools if t in FORCEPS_TOOLS]
    q = "What type of forceps is mentioned?"

    if not active_forceps:
        answers = [
            "No forceps are mentioned.",
            "No forceps are listed.",
            "There are no forceps referenced.",
            "No forceps are being used.",
            "No forceps.",
        ]
    else:
        tool_name = random.choice(active_forceps)
        cap = " ".join(w.capitalize() for w in tool_name.split())
        answers = [
            cap,
            f"The type of forceps mentioned is {cap}.",
            f"{cap} are the type mentioned.",
            f"The forceps type is {cap}.",
            f"{cap} is the specific type referenced.",
        ]
    return q, answers


def gen_suture_required(task_name):
    """Public sample example: case125"""
    q = "Is a suture required in this surgical step?"
    is_suturing = "suturing" in task_name.lower()

    if is_suturing:
        answers = [
            "Yes",
            "Yes, sutures are required.",
            "Yes, a suture is necessary.",
            "Yes, the procedure involves sutures.",
            "Yes, suturing is part of the procedure.",
        ]
    else:
        answers = [
            "No",
            "No, a suture is not required.",
            "No sutures are necessary.",
            "No, this step does not involve suturing.",
            "No, suturing is not part of this step.",
        ]
    return q, answers


def gen_task_identification(task_name):
    """Ask what task the surgeon is performing."""
    q_templates = [
        "What task is the surgeon performing?",
        "What surgical step is being performed?",
        "What is the current surgical task?",
    ]
    q = random.choice(q_templates)
    cap = task_name.capitalize() if task_name else "Other"
    answers = [
        cap,
        f"The task is {cap}.",
        f"{cap} is being performed.",
        f"The surgeon is performing {cap}.",
        f"It is {cap}.",
    ]
    return q, answers


def gen_organ_question(task_name, description):
    """Public sample example: case127 — "What organ is being manipulated?" → "Uterine horn" """
    q_templates = [
        "What organ is being manipulated?",
        "What anatomical structure is being operated on?",
        "What organ is involved in this step?",
    ]
    q = random.choice(q_templates)

    organ = TASK_ORGAN.get(task_name)

    # Try to extract organ from description if not mapped
    if organ is None and description:
        desc_lower = str(description).lower()
        organ_keywords = {
            "gallbladder": "gallbladder",
            "uterine horn": "uterine horn",
            "uterus": "uterus",
            "rectal": "rectum",
            "artery": "artery",
            "vein": "vein",
            "ligament": "ligaments",
            "peritoneum": "peritoneum",
            "colon": "colon",
            "bladder": "bladder",
            "liver": "liver",
            "kidney": "kidney",
        }
        for kw, organ_name in organ_keywords.items():
            if kw in desc_lower:
                organ = organ_name
                break

    if organ is None:
        organ = "the tissue in the surgical field"

    cap_organ = organ.capitalize() if organ[0].islower() else organ
    answers = [
        cap_organ,
        f"The organ being manipulated is the {organ}.",
        f"{cap_organ} is being manipulated.",
        f"The manipulation involves the {organ}.",
        f"The organ in focus is the {organ}.",
    ]
    return q, answers


def gen_procedure_type():
    """Public sample example: case129 — "What procedure is this summary describing?"
    All videos are from da Vinci robot → endoscopic/laparoscopic surgery."""
    q_templates = [
        "What procedure is this summary describing?",
        "What type of surgery is being performed?",
        "What kind of surgical procedure is this?",
    ]
    q = random.choice(q_templates)
    answers = [
        "Endoscopic surgery or a laparoscopic surgery",
        "The summary is describing endoscopic or laparoscopic surgery.",
        "This appears to be endoscopic or laparoscopic surgery.",
        "The procedure is likely endoscopic or laparoscopic surgery.",
        "It's endoscopic surgery or laparoscopic surgery based on the summary.",
    ]
    return q, answers


def gen_tool_purpose(active_tools):
    """Public sample example: case130 — "What is the purpose of using forceps in this procedure?"
    → "To grasp and hold tissues or objects during the surgery." """
    forceps_active = [t for t in active_tools if t in FORCEPS_TOOLS]
    other_active = [t for t in active_tools if t not in FORCEPS_TOOLS]

    # Prefer forceps if present (matches the public sample), otherwise pick any tool
    if forceps_active:
        tool = random.choice(forceps_active)
        tool_display = "forceps"
    elif other_active:
        tool = random.choice(other_active)
        tool_display = _sp(tool)[0]
    else:
        # No tools → pick a random one and describe its general purpose
        tool = random.choice(ALL_TARGET_TOOLS)
        tool_display = _sp(tool)[0]

    purpose = TOOL_PURPOSE.get(tool, "to assist during the surgical procedure")

    q_templates = [
        f"What is the purpose of using {tool_display} in this procedure?",
        f"Why is a {tool_display} being used?",
        f"What role does the {tool_display} play in this step?",
    ]
    q = random.choice(q_templates)

    cap_purpose = purpose[0].upper() + purpose[1:]
    answers = [
        f"{cap_purpose}.",
        f"The {tool_display} is used {purpose}.",
        f"{tool_display.capitalize()} is utilized {purpose}.",
        f"The purpose of {tool_display} is {purpose}.",
        f"{tool_display.capitalize()} is used {purpose}.",
    ]
    return q, answers


def gen_tissue_cutting(active_tools, task_name):
    """Public sample example: case131 — "Is tissue being cut during this clip?" → "Yes" """
    cutting_tools = ["monopolar curved scissors", "permanent cautery hook/spatula", "vessel sealer"]
    cutting_tasks = ["suspensory ligaments", "uterine horn", "rectal artery/vein", "skills application"]

    has_cutting_tool = any(t in active_tools for t in cutting_tools)
    has_cutting_task = task_name.lower() in [t.lower() for t in cutting_tasks]
    is_cutting = has_cutting_tool or has_cutting_task

    q_templates = [
        "Is tissue being cut during this clip?",
        "Is tissue dissection occurring in this clip?",
        "Is any cutting happening in this clip?",
    ]
    q = random.choice(q_templates)

    if is_cutting:
        answers = [
            "Yes",
            "Yes, tissue is being cut.",
            "Yes, the clip shows tissue being cut.",
            "Yes, cutting of tissue is occurring in this clip.",
            "Yes, tissue cutting is taking place.",
        ]
    else:
        answers = [
            "No",
            "No, tissue is not being cut.",
            "No cutting is visible in this clip.",
            "No, there is no tissue cutting in this clip.",
            "No, tissue cutting is not occurring.",
        ]
    return q, answers


def gen_description_question(description):
    """Generate a question from matched_description if available."""
    if not description or str(description) == 'nan' or len(str(description).strip()) < 10:
        return None, None

    desc = str(description).strip()
    q_templates = [
        "What is the surgeon doing in this step?",
        "Describe what is happening in this surgical clip.",
        "What activity is the surgeon performing?",
    ]
    q = random.choice(q_templates)

    # Create paraphrased answers from the description
    # Truncate very long descriptions
    if len(desc) > 200:
        desc = desc[:200].rsplit(' ', 1)[0] + "."

    answers = [
        desc,
        f"The surgeon is {desc[0].lower()}{desc[1:]}" if desc[0].isupper() else desc,
        f"In this step, {desc[0].lower()}{desc[1:]}" if desc[0].isupper() else f"In this step, {desc}",
        f"The activity involves: {desc}",
        f"The clip shows: {desc}",
    ]
    return q, answers


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main():
    segments_json = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\src\surgvu_segments.json"
    train_output  = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\src\train_vqa.json"
    val_output    = r"d:\SCLab\Surgical-Video-Understanding\SurgVU-challenge\src\val_vqa.json"

    with open(segments_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    case_groups = defaultdict(list)
    for seg in segments:
        case_groups[seg['case']].append(seg)

    random.seed(42)

    all_cases   = sorted(case_groups.keys())
    train_cases = all_cases[:140]
    val_cases   = all_cases[140:]
    print(f"Train cases: {len(train_cases)}, Val cases: {len(val_cases)}")

    # ── Question type generators with weights ──
    # Higher weight = more samples of that type
    QA_GENERATORS = [
        ("tool_presence",    3),  # most common in the public sample
        ("forceps_type",     1),
        ("suture_required",  1),
        ("task_id",          1),
        ("organ",            1),
        ("procedure_type",   1),
        ("tool_purpose",     1),
        ("tissue_cutting",   1),
        ("description",      1),
    ]

    def generate_for_segment(seg, qa_types_to_gen):
        active_tools = seg['tools']
        active_commercial_tools = set(seg.get('commercial_tools', []))
        task_name    = seg['task']
        description  = seg.get('description', '')
        results = []

        for qt in qa_types_to_gen:
            q, answers, tool_used = None, None, None
            if qt == "tool_presence":
                q, answers, tool_used = gen_tool_presence(active_tools, active_commercial_tools=active_commercial_tools)
            elif qt == "forceps_type":
                q, answers = gen_forceps_type(active_tools)
            elif qt == "suture_required":
                q, answers = gen_suture_required(task_name)
            elif qt == "task_id":
                q, answers = gen_task_identification(task_name)
            elif qt == "organ":
                q, answers = gen_organ_question(task_name, description)
            elif qt == "procedure_type":
                q, answers = gen_procedure_type()
            elif qt == "tool_purpose":
                q, answers = gen_tool_purpose(active_tools)
            elif qt == "tissue_cutting":
                q, answers = gen_tissue_cutting(active_tools, task_name)
            elif qt == "description":
                q, answers = gen_description_question(description)

            if q is not None and answers is not None:
                entry = {
                    "id": f"{seg['case']}_p{seg['part']}_t{int(seg['t_start'])}_{qt}_{random.randint(100,999)}",
                    "video": seg['video_path'],
                    "t_start": seg['t_start'],
                    "t_end": seg['t_end'],
                    "question": q,
                    "answers": answers,
                }
                if qt == "tool_presence":
                    entry["tool"] = tool_used
                results.append(entry)
        return results

    def sample_and_generate(cases_list, max_segs_per_case=60):
        vqa = []
        for case in tqdm(cases_list, desc="Generating VQA"):
            segs = case_groups[case]

            # ── Rebalancing: prioritise segments with tools or non-Other tasks ──
            interesting = [s for s in segs if s['tools'] or s['task'] != "Other"]
            boring      = [s for s in segs if not s['tools'] and s['task'] == "Other"]

            # Take up to 80% from interesting, 20% from boring
            n_interesting = min(len(interesting), int(max_segs_per_case * 0.8))
            n_boring      = min(len(boring), max_segs_per_case - n_interesting)

            sampled = []
            if interesting:
                sampled += random.sample(interesting, min(n_interesting, len(interesting)))
            if boring and n_boring > 0:
                sampled += random.sample(boring, min(n_boring, len(boring)))

            for seg in sampled:
                # Build weighted list of question types to generate for this segment
                qa_types = []
                for qt, weight in QA_GENERATORS:
                    qa_types.extend([qt] * weight)

                # Sample 3 question types per segment (diverse)
                chosen = random.sample(qa_types, min(3, len(qa_types)))

                vqa.extend(generate_for_segment(seg, chosen))

        return vqa

    def balance_tool_presence(vqa, cases_list, target_yes_ratio=0.5, tolerance=0.1, max_extra_per_tool=600):
        """Per-(tool, mode) Yes/No balancing.

        A few tools (cadiere forceps, needle driver) are physically present in most
        segments, so uniformly-random target_tool sampling makes tool_presence QA for
        those tools skew heavily Yes (~80%+) - the model then learns "this tool ->
        probably Yes" instead of actually checking the clip, which is exactly the
        false-positive failure seen on the real public sample (case122, case132). Fix by
        searching the full segment pool (not just what got sampled above) for
        counter-examples per tool and topping up whichever class is underrepresented.

        Balances generic tool keys (e.g. "clip applier") AND every individual
        commercial variant (e.g. "large clip applier") separately - v5's
        self-distillation error analysis found variant-mode questions ("was a
        maryland bipolar forceps used?") false-positive on Yes even when the
        generic tool key's aggregate Yes/No looked balanced, because generic-mode
        and variant-mode questions for the same tool can have very different
        underlying Yes-rates that average out when tracked under one key.
        """
        all_segs = [s for case in cases_list for s in case_groups[case]]
        all_variants = [v for variants in COMMERCIAL_VARIANTS.values() for v in variants]

        counts = defaultdict(lambda: [0, 0])  # balance_key -> [yes, no]
        for item in vqa:
            if "tool" not in item:
                continue
            counts[item["tool"]][0 if item["answers"][0] == "Yes" else 1] += 1

        extra = []
        for key in ALL_TARGET_TOOLS + all_variants:
            is_variant = key not in ALL_TARGET_TOOLS
            yes, no = counts[key]
            total = yes + no
            if total == 0:
                continue
            ratio = yes / total

            if ratio > target_yes_ratio + tolerance:
                # too many Yes -> top up with segments where this key is ABSENT
                need = min(int(yes / target_yes_ratio) - total, max_extra_per_tool)
                if is_variant:
                    pool = [s for s in all_segs if key not in s.get('commercial_tools', [])]
                else:
                    pool = [s for s in all_segs if key not in s['tools']]
            elif ratio < target_yes_ratio - tolerance:
                # too many No -> top up with segments where this key IS present
                need = min(int(no / (1 - target_yes_ratio)) - total, max_extra_per_tool)
                if is_variant:
                    pool = [s for s in all_segs if key in s.get('commercial_tools', [])]
                else:
                    pool = [s for s in all_segs if key in s['tools']]
            else:
                continue

            if need <= 0 or not pool:
                continue
            random.shuffle(pool)
            for seg in pool[:need]:
                commercial = set(seg.get('commercial_tools', []))
                if is_variant:
                    q, answers, _ = gen_tool_presence(
                        seg['tools'], active_commercial_tools=commercial, force_variant=key
                    )
                else:
                    q, answers, _ = gen_tool_presence(
                        seg['tools'], target_tool=key, active_commercial_tools=commercial,
                        force_generic=True
                    )
                extra.append({
                    "id": f"{seg['case']}_p{seg['part']}_t{int(seg['t_start'])}_tp_bal_{random.randint(100,999)}",
                    "video": seg['video_path'],
                    "t_start": seg['t_start'],
                    "t_end": seg['t_end'],
                    "question": q,
                    "answers": answers,
                    "tool": key,
                })

        return vqa + extra

    _TYPE_PATTERN = re.compile(r"_t\d+_(.+)_\d+$")

    def downsample_dominant_answers(vqa, target_types=("task_id", "description"), max_share=0.5):
        """v5's self-distillation error analysis found the model mode-collapses on
        task_id/description questions: it learned to answer the single most common
        label ("Other" task / "Activity outside of the structured training."
        description) regardless of what's actually in the clip, because that alone
        was ~74% correct in training data - a pure frequency-prior shortcut costing
        real accuracy. Cap the dominant answer's share per type instead of
        oversampling counter-examples (there's nothing to oversample here, the
        model just needs less incentive to always guess the majority class)."""
        by_type = defaultdict(list)
        for item in vqa:
            m = _TYPE_PATTERN.search(item['id'])
            by_type[m.group(1) if m else None].append(item)

        drop_ids = set()
        for t in target_types:
            items = by_type.get(t, [])
            if not items:
                continue
            counts = Counter(it['answers'][0] for it in items)
            dominant, dom_count = counts.most_common(1)[0]
            total = len(items)
            share = dom_count / total
            if share <= max_share:
                continue
            other_count = total - dom_count
            keep_dom = int((max_share * other_count) / (1 - max_share))
            keep_dom = max(0, min(keep_dom, dom_count))
            dominant_items = [it for it in items if it['answers'][0] == dominant]
            random.shuffle(dominant_items)
            drop = dominant_items[keep_dom:]
            drop_ids.update(id(it) for it in drop)
            print(f"  downsample[{t}]: dominant={dominant!r} {dom_count}/{total} ({share:.1%})"
                  f" -> keeping {keep_dom}, dropping {len(drop)}")

        return [it for it in vqa if id(it) not in drop_ids]

    train_vqa = sample_and_generate(train_cases, max_segs_per_case=60)
    val_vqa   = sample_and_generate(val_cases,   max_segs_per_case=60)

    train_vqa = balance_tool_presence(train_vqa, train_cases)
    val_vqa   = balance_tool_presence(val_vqa, val_cases)

    print("Downsampling dominant task_id/description answers (train):")
    train_vqa = downsample_dominant_answers(train_vqa)
    print("Downsampling dominant task_id/description answers (val):")
    val_vqa   = downsample_dominant_answers(val_vqa)
    # Not shuffled here: the Trainer already shuffles each epoch, and keeping QA
    # entries grouped by video/case helps extract_frames.py's decord cache locality.

    # ── Print stats ──
    print(f"\nGenerated {len(train_vqa)} train QA, {len(val_vqa)} val QA.")

    def print_stats(data, label):
        tp_items = [d for d in data if "tool" in d]
        yes_tp = sum(1 for d in tp_items if d['answers'][0] == 'Yes')
        no_tp  = len(tp_items) - yes_tp
        print(f"[{label}] Tool presence Yes/No: {yes_tp}/{no_tp}")

        per_tool = defaultdict(lambda: [0, 0])
        for d in tp_items:
            per_tool[d["tool"]][0 if d['answers'][0] == 'Yes' else 1] += 1
        for tool, (y, n) in sorted(per_tool.items(), key=lambda x: -sum(x[1])):
            tot = y + n
            print(f"[{label}]   {tool:30s} Yes={y:4d} No={n:4d}  Yes%={y/tot*100:.1f}")

        qt_counter = Counter()
        for d in data:
            qid = d['id'].rsplit('_', 2)
            if len(qid) >= 2:
                qt_counter[qid[-2]] += 1
        print(f"[{label}] Question types: {dict(qt_counter.most_common())}")

    print_stats(train_vqa, "train")
    print_stats(val_vqa,   "val")

    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_vqa, f, indent=2, ensure_ascii=False)
    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(val_vqa, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {train_output} and {val_output}")


if __name__ == "__main__":
    main()
