"""Optional LLM-as-a-judge stage (G-Eval-style rubric) through an OpenAI-compatible API.

Runs only when requested. Uses OPENROUTER_API_KEY (OpenRouter) or OPENAI_API_KEY. Outputs are judged
one at a time without revealing whether they came from the model or the reference.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROMPT_VERSION = "v1"
AXES = ("meaning", "completeness", "simplicity", "fluency")
FLAGS = ("hallucination", "number_error")

PROMPT = """당신은 어린이·청소년용 뉴스 쉬운 글 변환을 평가하는 한국어 전문가다.
아래 [원문]을 초등학교 고학년~중학생이 읽기 쉽게 바꾼 [변환문]을 평가하라.

[원문]
{source}

[변환문]
{output}

다음 기준으로 1~5점(5가 가장 좋음)을 매겨라.
- meaning: 원문의 사실(누가, 무엇을, 왜, 수치, 인과관계)을 왜곡 없이 전달하는가
- completeness: 원문의 핵심 정보가 빠지지 않았는가
- simplicity: 대상 독자가 이해하기 쉬운 단어와 문장 구조인가
- fluency: 문법적으로 자연스러운 한국어인가
그리고 다음을 true/false로 판단하라.
- hallucination: 원문에 없는 사실을 추가하거나 원문의 사실을 바꿨는가
- number_error: 숫자, 금액, 비율, 날짜가 원문과 다르게 바뀌었는가

JSON 하나만 출력하라: {{"meaning": 1-5, "completeness": 1-5, "simplicity": 1-5, "fluency": 1-5, "hallucination": true/false, "number_error": true/false, "reason": "한 문장 근거"}}"""


def _client():
    from openai import OpenAI

    if os.getenv("OPENROUTER_API_KEY"):
        return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", timeout=60, max_retries=0)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60, max_retries=0)
    raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY to run the judge stage.")


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"no JSON in judge reply: {text[:200]}")
    data = json.loads(match.group(0))
    if any(type(data[a]) not in (int, float) or not 1 <= data[a] <= 5 for a in AXES):
        raise ValueError("judge scores must be numbers between 1 and 5")
    if any(type(data[f]) is not bool for f in FLAGS):
        raise ValueError("judge flags must be JSON booleans")
    row = {f"judge_{a}": float(data[a]) for a in AXES}
    row.update({f"judge_{f}": float(bool(data[f])) for f in FLAGS})
    row["judge_reason"] = str(data.get("reason", ""))
    return row


def stratified_sample(lengths: pd.Series, n: int, seed: int) -> list[int]:
    """Equal draws from short / medium / long source-length tertiles."""
    bins = pd.qcut(lengths, 3, labels=False)
    rng = np.random.default_rng(seed)
    picked = []
    for b in range(3):
        idx = lengths.index[bins == b].to_numpy()
        picked.extend(rng.choice(idx, size=min(len(idx), n // 3 + (b < n % 3)), replace=False).tolist())
    return sorted(picked)


def judge(sources: list[str], outputs: dict[str, list[str]], indices: list[int], model: str,
          workers: int = 4, retries: int = 3, checkpoint: Path | None = None) -> pd.DataFrame:
    client = _client()
    saved = {}
    if checkpoint and checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            try:
                item = json.loads(line)
                saved[item["key"]] = item["row"]
            except (ValueError, KeyError):
                continue

    def key(task):
        s, i = task
        return hashlib.sha256(json.dumps([model, PROMPT, i, s, sources[i], outputs[s][i]], ensure_ascii=False).encode()).hexdigest()

    def call(task):
        system, i = task
        if key(task) in saved:
            return saved[key(task)]
        prompt = PROMPT.format(source=sources[i], output=outputs[system][i])
        for attempt in range(retries):
            try:
                reply = client.chat.completions.create(
                    model=model, temperature=0, max_tokens=600, messages=[{"role": "user", "content": prompt}])
                return {"system": system, "idx": i, **_parse(reply.choices[0].message.content)}
            except Exception as exc:
                if attempt == retries - 1:
                    return {"system": system, "idx": i, "judge_error": str(exc)[:200]}
                time.sleep(2 * (attempt + 1))

    tasks = [(s, i) for i in indices for s in outputs]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if checkpoint and "judge_error" not in row and key(futures[future]) not in saved:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint.open("a") as stream:
                    stream.write(json.dumps({"key": key(futures[future]), "row": row}, ensure_ascii=False) + "\n")
            if len(rows) % 20 == 0 or len(rows) == len(tasks):
                print(f"[judge] {len(rows)}/{len(tasks)} outputs", flush=True)
    df = pd.DataFrame(rows)
    df["judge_model"], df["judge_prompt"] = model, PROMPT_VERSION
    return df


def main() -> None:
    """Run the optional judge independently while local CPU metrics execute."""
    import argparse
    from dotenv import load_dotenv

    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--csv", type=Path, default=project / "result.csv")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--work-dir", type=Path, default=project / "evaluation" / "runs" / "full")
    args = parser.parse_args()
    load_dotenv(project / ".env", override=False)
    model = args.model or ("openai/gpt-4o-mini" if os.getenv("OPENROUTER_API_KEY") else "gpt-4o-mini")
    data = pd.read_csv(args.csv).fillna("").astype(str)
    if args.n < 1 or args.n > len(data):
        parser.error("--n must be between 1 and the dataset size")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    stem = f"judge_{model.replace('/', '_')}_{args.n}"
    indices = stratified_sample(data.original.str.len(), args.n, args.seed)
    result = judge(data.original.tolist(), {"model": data.simplified_by_model.tolist(),
        "reference": data.simplified_human.tolist()}, indices, model,
        checkpoint=args.work_dir / f"{stem}.jsonl")
    result.to_csv(args.work_dir / "judge.csv", index=False)
    errors = int(result.get("judge_error", pd.Series(dtype=object)).notna().sum())
    if errors:
        print(f"{errors}/{len(result)} judge calls failed; successful calls were checkpointed", flush=True)
        raise SystemExit(1)
    pd.to_pickle(result, args.work_dir / f"{stem}.pkl")
    print(f"Saved {len(result)} judge ratings for {len(indices)} sources", flush=True)


if __name__ == "__main__":
    main()
