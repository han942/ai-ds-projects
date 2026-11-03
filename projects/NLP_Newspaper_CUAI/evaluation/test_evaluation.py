"""Focused regressions for metric parity, batched pairing, and judge persistence."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from . import judge, neural, report
from .korean import numbers
from .sari import sari_sentence


class EvaluationTests(unittest.TestCase):
    def test_nli_respects_roberta_position_offset(self):
        tok = MagicMock(model_max_length=512)
        model = MagicMock()
        model.config.max_position_embeddings = 512
        model.config.pad_token_id = 1
        model.config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        model.return_value.logits = torch.tensor([[3.0, 1.0, 0.0]])
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok), patch(
                "transformers.AutoModelForSequenceClassification.from_pretrained", return_value=MagicMock(eval=lambda: model)):
            nli = neural._NLI(32)
        self.assertEqual(nli.max_length, 510)
        tok.return_value = {"input_ids": torch.ones((1, 510), dtype=torch.long)}
        e, c = nli.probs(["긴 원문"], ["출력"])
        self.assertEqual(tok.call_args.kwargs["max_length"], 510)
        self.assertGreater(e[0], c[0])

    def test_sari_matches_original_metric(self):
        files = list((Path.home() / ".cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--sari").glob("*/sari.py"))
        if not files:
            self.skipTest("original Hugging Face SARI implementation is not cached")
        original = runpy.run_path(str(files[0]))
        norm = original["normalize"]
        examples = [
            ("정부가 3만 명에게 지원금을 지급한다.", "정부가 3만 명에게 돈을 줘요.", ["정부가 3만 명에게 돈을 줘요."]),
            ("a a b c", "a b b", ["a b", "b c"]),
            ("같은 문장", "같은 문장", ["같은 문장"]),
            ("관련 없는 원문입니다.", "동물이 잠을 자요.", ["원문을 쉽게 바꿔요."]),
        ]
        for source, output, refs in examples:
            expected = original["SARIsent"](norm(source), norm(output), [norm(r) for r in refs]) * 100
            self.assertAlmostEqual(sari_sentence(source, output, refs)["sari"], expected, places=10)

    def test_numbers_normalize_equivalent_forms(self):
        self.assertEqual(numbers("100만원"), numbers("100만 원"))
        self.assertEqual(numbers("2000억"), numbers("2천억"))
        self.assertEqual(numbers("세 곳"), numbers("3곳"))
        self.assertNotEqual(numbers("3만 명"), numbers("30만 명"))

    def test_bertscore_combined_call_preserves_pair_order(self):
        scorer = MagicMock()
        scorer.score.return_value = tuple(torch.arange(3, dtype=torch.float) + i for i in range(3))
        with patch("bert_score.BERTScorer", return_value=scorer):
            result = neural.bertscore({"a": (["a1", "a2"], ["r1", "r2"]), "b": (["b1"], ["r3"])})
        args = scorer.score.call_args.args
        self.assertEqual(args, (["a1", "a2", "b1"], ["r1", "r2", "r3"]))
        np.testing.assert_array_equal(result["a"]["p"], [0, 1])
        np.testing.assert_array_equal(result["b"]["f"], [4])

    def test_judge_rejects_string_flags_and_out_of_range_scores(self):
        valid = dict(meaning=5, completeness=4, simplicity=3, fluency=4, hallucination=False, number_error=False)
        self.assertEqual(judge._parse(json.dumps(valid))["judge_hallucination"], 0)
        for change in ({"hallucination": "false"}, {"meaning": 6}):
            with self.assertRaises(ValueError):
                judge._parse(json.dumps({**valid, **change}))

    def test_judge_checkpoint_reuses_only_matching_inputs(self):
        client = MagicMock()
        response = dict(meaning=5, completeness=4, simplicity=3, fluency=4, hallucination=False, number_error=False)
        client.chat.completions.create.return_value.choices[0].message.content = json.dumps(response)
        with tempfile.TemporaryDirectory() as d, patch.object(judge, "_client", return_value=client):
            checkpoint = Path(d) / "judge.jsonl"
            judge.judge(["원문"], {"model": ["출력"]}, [0], "test-model", checkpoint=checkpoint)
            judge.judge(["원문"], {"model": ["출력"]}, [0], "test-model", checkpoint=checkpoint)
            self.assertEqual(client.chat.completions.create.call_count, 1)
            judge.judge(["수정한 원문"], {"model": ["출력"]}, [0], "test-model", checkpoint=checkpoint)
            self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_paired_difference_uses_matched_rows(self):
        data = pd.DataFrame({"idx": [0, 1, 0, 1], "system": ["model", "model", "copy", "copy"], "score": [5, 9, 3, np.nan]})
        self.assertEqual(report.paired_diff(data, "score", "model", "copy"), (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
