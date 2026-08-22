import random
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from curate_surgmotion_vqa import case_number, semantic_organ_answers
from generate_qa import (
    gen_description_question,
    gen_forceps_type,
    gen_tool_presence,
    gen_tool_purpose,
)


class GeneratorV3Tests(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def assert_five_unique(self, answers):
        self.assertEqual(5, len(answers))
        self.assertEqual(5, len({answer.strip().lower() for answer in answers}))

    def test_generic_forceps_is_union_of_subtypes(self):
        question, answers, key = gen_tool_presence(
            ["bipolar forceps"], target_tool="forceps", force_generic=True
        )
        self.assertEqual("forceps", key)
        self.assertEqual("Yes", answers[0])
        self.assertIn("install", question.lower())
        self.assert_five_unique(answers)

    def test_subtype_does_not_collapse_to_generic_forceps(self):
        question, answers, key = gen_tool_presence(
            ["bipolar forceps"], target_tool="cadiere forceps", force_generic=True
        )
        self.assertEqual("cadiere forceps", key)
        self.assertEqual("No", answers[0])
        self.assertIn("cadiere forceps", question.lower())
        self.assertNotRegex(question.lower(), r"\bbeing used\b|\bpresent\b|\binvolved\b")

    def test_commercial_variant_uses_commercial_record(self):
        question, answers, key = gen_tool_presence(
            ["needle driver"],
            active_commercial_tools={"large needle driver"},
            force_variant="large needle driver",
        )
        self.assertEqual("large needle driver", key)
        self.assertEqual("Yes", answers[0])
        self.assertIn("large needle driver", question.lower())

    def test_forceps_zero_one_and_multiple_labels(self):
        question, answers = gen_forceps_type([])
        self.assertIn("any forceps types", question.lower())
        self.assert_five_unique(answers)

        question, answers = gen_forceps_type(["prograsp forceps"])
        self.assertIn("type is", answers[1].lower())
        self.assertTrue(all("prograsp forceps" in answer.lower() for answer in answers))
        self.assert_five_unique(answers)

        question, answers = gen_forceps_type(
            ["prograsp forceps", "cadiere forceps", "bipolar forceps"]
        )
        self.assertIn("types", question.lower())
        for answer in answers:
            lowered = answer.lower()
            self.assertLess(lowered.index("cadiere forceps"), lowered.index("bipolar forceps"))
            self.assertLess(lowered.index("bipolar forceps"), lowered.index("prograsp forceps"))
        self.assert_five_unique(answers)

    def test_tool_purpose_requires_and_names_active_tool(self):
        self.assertEqual((None, None), gen_tool_purpose([]))
        question, answers = gen_tool_purpose(["cadiere forceps", "needle driver"])
        self.assertTrue(
            "cadiere forceps" in question.lower() or "needle driver" in question.lower()
        )
        self.assertNotIn("a forceps", question.lower())
        self.assert_five_unique(answers)

    def test_description_cleanup_and_filtering(self):
        self.assertEqual(
            (None, None),
            gen_description_question("Activity outside of the structured training."),
        )
        question, answers = gen_description_question(
            "  The surgeon   retracts tissue and exposes the field  "
        )
        self.assertTrue(question)
        self.assertTrue(all(answer.endswith(".") for answer in answers))
        self.assertTrue(all("  " not in answer for answer in answers))
        self.assert_five_unique(answers)

    def test_semantic_organ_has_five_references(self):
        answers = semantic_organ_answers("Colon")
        self.assert_five_unique(answers)
        self.assertTrue(all("colon" in answer.lower() for answer in answers))

    def test_case_number_accepts_id_and_video(self):
        self.assertEqual(147, case_number({"id": "case147_p1_t0_organ_123"}))
        self.assertEqual(
            154,
            case_number({"id": "missing", "video": "/data/case_154/part_1.mp4"}),
        )


if __name__ == "__main__":
    unittest.main()
