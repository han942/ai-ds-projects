"""Unit and mock end-to-end tests for C0 OpenRouter runner, agent fixes, splits, and comparison."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from agent import prompts, tracing
from agent.agent import NewsSimplifierAgent
from agent.c0 import make_cache_key, run_c0_experiment
from agent.config import Settings
from agent.evaluation import Evaluator, _rouge_l
from agent.llm import extract_usage, strip_thinking
from agent.splits import STRESS_INDICES, build_splits
from evaluation import compare

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_V1 = PROJECT_ROOT / "evaluation" / "splits" / "stratified_v1.json"
RESULT_CSV = PROJECT_ROOT / "result.csv"


class _FakeAIMessage:
    def __init__(self, content: str, usage_metadata: dict | None = None):
        self.content = content
        self.usage_metadata = usage_metadata or {
            "input_tokens": 120,
            "output_tokens": 45,
            "total_tokens": 165,
        }
        self.response_metadata = {}


class _RateLimitError(RuntimeError):
    status_code = 429


class ConfigAndFixesTests(unittest.TestCase):
    def test_openrouter_requires_explicit_model_and_sets_base_url(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENROUTER_", "OPENAI_", "JUDGE_"))}
        with patch.dict(os.environ, env, clear=True):
            s = Settings(openrouter_api_key="or-key")
            self.assertEqual(s.provider, "openrouter")
            self.assertEqual(s.api_key, "or-key")
            self.assertEqual(s.base_url, "https://openrouter.ai/api/v1")
            self.assertEqual(s.model, "")
            with self.assertRaises(RuntimeError):
                s.require_llm(require_model=True)

    def test_prompt_config_does_not_override_settings_models(self):
        s = Settings(
            openrouter_api_key="or-key",
            model="qwen/qwen3-32b:free",
            judge_model="google/gemini-2.5-flash",
        )
        with patch("agent.agent.build_chat") as mock_build_chat:
            ag = NewsSimplifierAgent(s)
            self.assertEqual(ag._cfg("news-simplify", "main")[0], "qwen/qwen3-32b:free")
            self.assertEqual(ag._cfg("news-critique", "judge")[0], "google/gemini-2.5-flash")
            self.assertEqual(ag._cfg("news-analyze", "judge")[0], "google/gemini-2.5-flash")
            mock_build_chat.assert_not_called()

        with patch("agent.evaluation.build_chat") as mock_eval_chat:
            Evaluator(s)
            mock_eval_chat.assert_called_once()
            self.assertEqual(mock_eval_chat.call_args.kwargs["model"], "google/gemini-2.5-flash")

    def test_tracing_is_enabled_evaluates_dynamically(self):
        with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}):
            self.assertFalse(tracing.is_enabled())
        with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}):
            self.assertEqual(tracing.is_enabled(), tracing._IMPORT_OK)

    def test_strip_thinking_and_extract_usage(self):
        cleaned, had = strip_thinking("<think>내부 추론</think>\n```\n쉬운 문장입니다.\n```")
        self.assertTrue(had)
        self.assertEqual(cleaned, "쉬운 문장입니다.")

        empty_after, had_empty = strip_thinking("<think>only reasoning</think>")
        self.assertTrue(had_empty)
        self.assertEqual(empty_after, "")

        msg = _FakeAIMessage("ok", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "total_cost": 0.0})
        usage = extract_usage(msg)
        self.assertEqual(usage, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_usd": 0.0})

    def test_korean_morpheme_rouge_l_regression(self):
        same = _rouge_l("정부가 3만 명에게 지원금을 줘요.", "정부가 3만 명에게 지원금을 줘요.")
        self.assertEqual(same, 1.0)
        partial = _rouge_l("정부가 3만 명에게 지원금을 줘요.", "정부가 시민들에게 돈을 줘요.")
        self.assertIsNotNone(partial)
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, 1.0)

    def test_critique_parse_error_surfaces_instead_of_silent_empty_fallback(self):
        s = Settings(openrouter_api_key="or-key", model="test/model:free")
        ag = NewsSimplifierAgent(s)
        with patch.object(ag, "_complete", return_value="invalid non-json text"):
            res = ag._node_critique({"article": "원문", "draft": "초안", "critiques": []})
        self.assertFalse(res["approved"])
        self.assertTrue(res["critiques"][-1]["parse_error"])
        self.assertIn("JSON 파싱 실패", res["critiques"][-1]["issues"][0])


class SplitsTests(unittest.TestCase):
    def test_stratified_v1_split_invariants(self):
        self.assertTrue(SPLIT_V1.exists(), "stratified_v1.json must be generated")
        data = json.loads(SPLIT_V1.read_text(encoding="utf-8"))
        eval_idx = data["splits"]["eval"]
        dev_idx = data["splits"]["dev"]
        stress_idx = data["splits"]["stress"]

        self.assertEqual(len(eval_idx), 150)
        self.assertEqual(len(dev_idx), 100)
        self.assertEqual(stress_idx, list(STRESS_INDICES))
        self.assertEqual(set(eval_idx) & set(dev_idx), set())
        self.assertEqual(set(dev_idx) & set(stress_idx), set())

        with RESULT_CSV.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        eval_orig = [rows[i]["original"] for i in eval_idx]
        dev_orig = [rows[i]["original"] for i in dev_idx]
        stress_orig = [rows[i]["original"] for i in stress_idx]
        self.assertEqual(len(eval_orig), len(set(eval_orig)))
        self.assertEqual(len(dev_orig), len(set(dev_orig)))
        self.assertEqual(set(eval_orig) & set(dev_orig), set())
        self.assertEqual(set(dev_orig) & set(stress_orig), set())
        self.assertEqual(len(data["strata_counts"]["eval"]), 6)
        self.assertEqual(len(data["strata_counts"]["dev"]), 6)


class C0RunnerAndCompareTests(unittest.TestCase):
    def _make_mini_dataset(self, tmp: Path) -> tuple[Path, Path]:
        with RESULT_CSV.open(encoding="utf-8-sig") as f:
            all_rows = list(csv.DictReader(f))
        csv_path = tmp / "mini.csv"
        # Pick 12 rows spanning short, medium, long + stress #0 and #1.
        chosen = [0, 1, 5, 24, 25, 65, 100, 167, 617, 1285, 1497, 2008]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["original", "simplified_human", "simplified_by_model"])
            writer.writeheader()
            for i in chosen:
                writer.writerow(all_rows[i])

        split_payload = {
            "version": "mini_test",
            "tertile_bounds": {"short_max": 70, "medium_max": 118, "long_min": 119},
            "splits": {
                # Intentionally overlap index 0 across eval and stress to test deduplication.
                "eval": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "stress": [0, 9, 10],
            },
        }
        split_path = tmp / "splits.json"
        split_path.write_text(json.dumps(split_payload), encoding="utf-8")
        return csv_path, split_path

    def test_c0_runner_retry_resume_dedup_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            csv_path, split_path = self._make_mini_dataset(tmp)
            out_dir = tmp / "run"
            settings = Settings(openrouter_api_key="or-key", model="test/free-a:free")

            calls: list[str] = []
            attempt_counts: dict[str, int] = {}

            class _FakeChat:
                def invoke(self, messages, config=None):
                    prompt_text = messages[-1].content
                    calls.append(prompt_text)
                    attempt_counts[prompt_text] = attempt_counts.get(prompt_text, 0) + 1
                    # Row idx 1 contains "생활권" -> fail once with 429 then succeed.
                    if "생활권" in prompt_text and attempt_counts[prompt_text] == 1:
                        raise _RateLimitError("429 Too Many Requests")
                    # Row idx 2 ("부천시") -> always fail on first run to test errors.jsonl + resume.
                    if "부천시" in prompt_text and os.environ.get("TEST_FAIL_BUCHEON") == "1":
                        raise _RateLimitError("429 persistent")
                    return _FakeAIMessage("<think>고민 중</think>\n쉬운 뉴스 문장 100개예요.")

            with patch.dict(os.environ, {"TEST_FAIL_BUCHEON": "1"}):
                m1 = run_c0_experiment(
                    settings=settings,
                    csv_path=csv_path,
                    split_file=split_path,
                    split_names=("eval", "stress"),
                    out_dir=out_dir,
                    temperature=0.0,
                    min_interval=0.0,
                    max_retries=2,
                    retry_base_delay=0.0,
                    chat_factory=lambda **_: _FakeChat(),
                    sleep_fn=lambda _: None,
                )

            # Union of eval (0..8) and stress (0, 9, 10) has 11 unique rows; idx 0 is called once.
            self.assertEqual(m1["counts"]["requested"], 11)
            self.assertEqual(m1["counts"]["total_succeeded"], 10)
            self.assertEqual(m1["counts"]["failed"], 1)
            self.assertEqual(m1["failed_indices"], [2])
            self.assertTrue((out_dir / "errors.jsonl").exists())

            # Re-run with failure fixed: 10 cached rows must not be re-called; only idx 2 runs.
            calls.clear()
            with patch.dict(os.environ, {"TEST_FAIL_BUCHEON": "0"}):
                m2 = run_c0_experiment(
                    settings=settings,
                    csv_path=csv_path,
                    split_file=split_path,
                    split_names=("eval", "stress"),
                    out_dir=out_dir,
                    temperature=0.0,
                    min_interval=0.0,
                    max_retries=2,
                    retry_base_delay=0.0,
                    chat_factory=lambda **_: _FakeChat(),
                    sleep_fn=lambda _: None,
                )

            self.assertEqual(m2["counts"]["cached_reused"], 10)
            self.assertEqual(m2["counts"]["new_succeeded"], 1)
            self.assertEqual(m2["counts"]["total_succeeded"], 11)
            self.assertEqual(m2["counts"]["failed"], 0)
            self.assertEqual(len(calls), 1)
            self.assertIn("부천시", calls[0])

            # Changing model invalidates cache key and re-runs all 11 rows.
            calls.clear()
            m3 = run_c0_experiment(
                settings=settings,
                model="test/free-b:free",
                csv_path=csv_path,
                split_file=split_path,
                split_names=("eval", "stress"),
                out_dir=out_dir,
                temperature=0.0,
                min_interval=0.0,
                max_retries=2,
                retry_base_delay=0.0,
                chat_factory=lambda **_: _FakeChat(),
                sleep_fn=lambda _: None,
            )
            self.assertEqual(m3["counts"]["cached_reused"], 0)
            self.assertEqual(m3["counts"]["new_succeeded"], 11)
            self.assertEqual(len(calls), 11)

            # Verify cache key changes on temperature change too.
            k0 = make_cache_key("m", "v1", "sha", 0.0, 0, "원문")
            k1 = make_cache_key("m", "v1", "sha", 0.5, 0, "원문")
            self.assertNotEqual(k0, k1)

    def test_compare_end_to_end_with_mocked_neural(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            csv_path, split_path = self._make_mini_dataset(tmp)
            pred_dir = tmp / "run"
            settings = Settings(openrouter_api_key="or-key", model="test/mock-free:free")

            class _EchoSimpleChat:
                def invoke(self, messages, config=None):
                    return _FakeAIMessage("쉬운 문장으로 바꿨어요. 숫자 29와 3곳을 지켰어요.")

            run_c0_experiment(
                settings=settings,
                csv_path=csv_path,
                split_file=split_path,
                split_names=("eval", "stress"),
                out_dir=pred_dir,
                temperature=0.0,
                min_interval=0.0,
                chat_factory=lambda **_: _EchoSimpleChat(),
                sleep_fn=lambda _: None,
            )

            def fake_sbert(pairs):
                return {k: np.full(len(a), 0.85) for k, (a, _) in pairs.items()}

            def fake_bertscore(pairs):
                return {
                    k: {
                        "p": np.full(len(c), 0.82),
                        "r": np.full(len(c), 0.81),
                        "f": np.full(len(c), 0.815),
                    }
                    for k, (c, _) in pairs.items()
                }

            def fake_nli(systems, checkpoint_dir=None):
                return {
                    name: {
                        "nli_entail_min": np.full(len(srcs), 0.8),
                        "nli_contra_max": np.full(len(srcs), 0.1),
                        "nli_supported": np.ones(len(srcs)),
                        "nli_contradiction": np.zeros(len(srcs)),
                        "nli_coverage": np.full(len(srcs), 0.7),
                    }
                    for name, (srcs, _) in systems.items()
                }

            out_report = tmp / "C0_REPORT.md"
            with (
                patch.object(compare.neural, "sbert_similarity", side_effect=fake_sbert),
                patch.object(compare.neural, "bertscore", side_effect=fake_bertscore),
                patch.object(compare.neural, "nli_faithfulness", side_effect=fake_nli),
            ):
                summary = compare.run_comparison(
                    pred_dir=pred_dir,
                    csv_path=csv_path,
                    split_file=split_path,
                    split="eval",
                    stress_split="stress",
                    out_path=out_report,
                    skip_neural=False,
                    threads=2,
                )

            self.assertEqual(summary["matched"], 9)
            self.assertTrue(out_report.exists())
            self.assertTrue((pred_dir / "eval_cache" / "per_sample.csv").exists())
            self.assertTrue((pred_dir / "eval_cache" / "summary.csv").exists())
            text = out_report.read_text(encoding="utf-8")
            self.assertIn("C0 (LLM Single-shot) − Model (Gemma 3-1B)", text)
            self.assertIn("Stress #0", text)
            self.assertIn("## By Source Length", text)
            self.assertIn("## Operational Efficiency (C0)", text)


if __name__ == "__main__":
    unittest.main()
