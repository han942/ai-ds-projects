"""Controlled BM25 review-cleanup comparison using the shared candidate runner.

Each arm has one fixed config and all three candidate policies. Test results
are diagnostic; preprocessing and policy preferences are chosen on validation.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.experiments.artifacts import read_json, read_jsonl, write_json
from rating_recsys.experiments.candidate_models import BM25Candidate
from rating_recsys.experiments.comparison import policies, run_candidate_comparison
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import paired_bootstrap
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import load_snapshot, load_review_texts
from rating_recsys.retrieval.bm25 import BM25Config
from rating_recsys.retrieval.review_preprocessing import REVIEW_STOPWORDS


ARMS = ("baseline", "clean", "clean_stopwords")
LABELS = {"baseline": "추가 전처리 없음(기존 Kiwi)", "clean": "잡음 정리",
          "clean_stopwords": "잡음 정리 + 공통 표현 제거"}
STAGES = {"bm25": "BM25 단독", "rrf_c1_bm25": "C1+BM25",
          "rrf_c1_c4_bm25": "C1+LightGCN+BM25"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(folder: Path, result: dict) -> None:
    rows = []
    for stage, label in STAGES.items():
        for arm in ARMS:
            run = result["arms"][arm]
            val, test = run["metrics"]["validation"][stage], run["metrics"]["test"][stage]
            rows.append(f"| {label} | {LABELS[arm]} | {100*val['recall_at_100']:.2f}% | {100*test['recall_at_100']:.2f}% | {100*test['recall_at_10']:.2f}% | {test['ndcg_at_10']:.6f} | {test['mrr_at_10']:.6f} |")
    contrasts = []
    for arm, stages in result["test_contrasts_vs_baseline"].items():
        for stage, scored in stages.items():
            rc, nd = scored["at_100"]["recall_at_100"], scored["at_10"]["ndcg_at_10"]
            contrasts.append(f"| {LABELS[arm]} | {STAGES[stage]} | {100*rc['delta']:+.3f}%p | [{100*rc['ci95'][0]:+.3f}, {100*rc['ci95'][1]:+.3f}]%p | {nd['delta']:+.6f} | [{nd['ci95'][0]:+.6f}, {nd['ci95'][1]:+.6f}] |")
    profiles = []
    pair_rows = []
    for arm in ARMS:
        meta = result["arms"][arm]["metrics"]["refit"]["retriever_preprocessing"]
        profiles.append(f"| {LABELS[arm]} | {meta['users_with_query_tokens']} | {meta['users_without_query_tokens']} | {meta['indexed_restaurants']} | {meta['indexed_tokens']:,} | {meta['vocabulary_size']:,} | {meta['profiles_changed_by_cleanup']} |")
        for stage,label in STAGES.items():
            p = result['arms'][arm]['metrics']['observed_pair_diagnostics']['test'][stage]
            acc = '미정의' if p['accuracy'] is None else f"{100*p['accuracy']:.2f}%"
            pair_rows.append(f"| {LABELS[arm]} | {label} | {acc} | {p['evaluated_queries']} | {p['compared_pairs']} |")
    chosen = result['validation_selection']
    reference = result['arms']['baseline']['metrics']['test']['c5_c1_lightgcn_rrf']
    report = f'''# Kiwi + BM25 리뷰 추가 전처리 비교

질문: 동일 리뷰에 추가 전처리를 적용하면 BM25 후보 회수와 상위 순위가 개선되는가?

**기존 Kiwi 처리를 대조군으로 유지하고, 잡음 정리와 공통 표현 제거를 단계적으로 비교했다.** 모델·정답·리뷰 선택·평가 모수는 고정했고, 세 조건을 사전에 정했다. C5 후보와 기본 baseline은 변경하지 않았다.

## 조건별 전체 결과

| 후보 방식 | 전처리 | Val Recall@100 | Test Recall@100 | Test Recall@10 | Test NDCG@10 | Test MRR@10 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

공통 C5 기준: Test Recall@100 {100*reference['recall_at_100']:.4f}%, Recall@10 {100*reference['recall_at_10']:.4f}%, NDCG@10 {reference['ndcg_at_10']:.6f}.

## 실제 적용한 전처리

1. **baseline:** 기존 공백 정리·리뷰당 240자 선택·Kiwi cong 형태소 분석·내용 품사 선택·소문자화만 유지한다. 이것도 기본 토큰 전처리를 포함하며, '원문을 아무 처리 없이 BM25에 넣었다'는 뜻이 아니다.
2. **clean:** 같은 리뷰 선택과 240자 잘림 이후 HTML entity를 복원하고 URL·HTML tag·이모티콘·반복 웃음/울음 문자를 정리한 뒤 Unicode NFKC를 적용한다. 앞의 잡음을 제거했다고 240자 이후 내용을 더 읽거나 다음 리뷰로 보충하지 않는다.
3. **clean_stopwords:** clean에 고정된 공통 칭찬·방문 표현을 추가 제거한다. Kiwi의 완전한 형태 단위로만 제거하며 단어 일부를 지우지 않는다. 불용어: {', '.join(sorted(REVIEW_STOPWORDS))}.

음식·메뉴·친절·서비스·가격 같은 표현은 사용자 정의 불용어로 제거하지 않는다. 다만 기존 Kiwi 내용 품사 필터는 모든 조건에 그대로 적용되어, 문장 전체의 부정·감정 의미를 이해하는 모델은 아니다. 지역명·상호명 마스킹, 사용자별 불용어 학습, 감정 모델, 동의어 확장은 이 비교에 포함하지 않았다.

## 고정 조건과 선택 원칙

- 사용자: cutoff 이전 최근 4점 이상 리뷰 최대 5개. 식당: cutoff 이전 최근 4점 이상 리뷰 최대 10개. 공백 정리 후 리뷰당 240자. 세 조건의 **원본 선택 프로필 hash가 validation/test 각각 동일함을 확인**했다.
- `k1=1.2`, `b=0.75`, Lucene 방식, seed=42, CPU threads=6. 사용자 query 중복 단어 제거, 식당 문서 빈도 유지. 같은 ID 동점 처리와 방문 식당 제외를 유지한다.
- validation 입력 ≤2025-12-19, test 입력 ≤2026-05-04. 각 cutoff 식당 문서로 IDF·문서 길이를 새로 계산한다. 미래 리뷰는 입력에 포함하지 않았다.
- 원본 baseline과 동일한 snapshot·개인별 만족도 label·평가 사용자·C5를 사용한다. Query 전체 validation 1,645명/test 1,580명, 양성 있는 주 지표 모수 1,639명/1,573명이다. 빈 query 사용자를 추가로 제외하거나 빈 후보를 임의로 채우지 않는다.
- BM25 단독, C1+BM25 RRF, C1+LightGCN+BM25 RRF를 **각각 같은 정책끼리 비교**한다. RRF constant=60, 각 출처 Top-100를 합쳐 Top-100를 유지한다. LambdaRank는 이번에 재학습하지 않았다.
- 전처리 효과의 주 비교는 clean_stopwords 대 baseline이다. clean은 잡음 정리만의 영향을 분리하는 보조 조건이다. 모든 조건의 test는 사전 정의된 ablation 평가이며, test로 불용어·파라미터를 고르지 않았다.
- validation BM25 단독 Recall@100 최대 조건: **{LABELS[chosen['standalone_preprocessing']]}**. validation의 3조건×3정책 중 최고 텍스트 정책: **{LABELS[chosen['preprocessing']]} / {STAGES[chosen['policy']]}**. 동률은 validation NDCG@10, 사전 조건 순서로 처리했다. 이 선택은 기본 모델 변경을 의미하지 않는다.

## 같은 사용자별 전처리 차이와 불확실성

동일 사용자의 점수 차이에 대해 paired bootstrap {result['config']['bootstrap_samples']:,}회, seed 42, 95% 구간이다. 차이는 '적용 − 추가 전처리 없음'이다. 아래 비교들은 탐색적이며 여러 정책·지표를 확인했다. 개별 구간만으로 일반적인 개선을 선언하지 않는다.

| 전처리 | 후보 방식 | Recall@100 차이 | Recall 95% 구간 | NDCG@10 차이 | NDCG 95% 구간 |
|---|---|---:|---|---:|---|
{chr(10).join(contrasts)}

## 원본 리뷰 선택은 같고 토큰만 달라졌는가

Test의 query 전체 모수는 1,580명이다. 잡음 때문에 변경된 프로필 수는 사용자·식당의 중복 없는 원본 문자열 단위이며 식당 수가 아니다.

| 전처리 | query 토큰 있음 | query 토큰 없음 | 색인 식당 | 식당 색인 토큰 | 어휘 수 | 잡음 정리로 바뀐 프로필 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(profiles)}

정확한 원본·처리 프로필 hash, 토큰 hash, 불용어별 제거 수, 빈 프로필과 전처리 버전은 각 실행의 metrics.json → refit → retriever_preprocessing에 기록했다.

## 관측 만족도 쌍 순서 정확도 — 보조 진단

각 후보에 함께 들어온 실제 미래 관측 식당 중 relevance가 다른 쌍만 비교하는 사용자 평균이다. 전처리에 따라 비교 가능한 사용자·쌍이 달라지므로 표의 정확도를 직접적인 전체 추천 개선으로 해석하지 않는다. 선택에는 사용하지 않는다.

| 전처리 | 후보 방식 | pair 정확도 | 비교 사용자 | 비교 쌍 |
|---|---|---:|---:|---:|
{chr(10).join(pair_rows)}

## 검증·재현·해석 범위

기존 BM25 대조군의 validation/test 후보와 점수를 과거 실행과 대조했다. 각 단계의 원본 프로필 hash와 C5 후보가 조건 간 동일한지 확인했다. 입력 snapshot·리뷰 본문 SHA-256와 전처리 소스 hash, 각 arm 실행 경로는 이 폴더 manifest.json에 기록했다. 각 실행에는 test 색인과 query 토큰을 저장한다. 과거의 평가 API와 cutoff를 그대로 사용한 후보 비교이며, review 피처를 넣은 LambdaRank 성능을 측정한 실험은 아니다.

BM25는 유사 단어가 나오는 문서를 찾아주는 방식이다. 리뷰가 평점을 준 이유를 담더라도 문장 의미·부정·조건부 선호를 직접 학습하지는 않는다. 따라서 리뷰가 중요한 정보라는 가설과 이 BM25 표현이 그 정보를 잘 활용한다는 가설은 구분해야 한다. 이번 결과는 선택한 전처리 범위에 대한 증거이며 모든 텍스트 활용 방식에 대한 결론이 아니다.

```sh
.venv/bin/python -m rating_recsys.experiments.bm25_preprocessing \\
  --baseline-run artifacts/runs/20261004T125314601899Z-e7896add \\
  --reference-bm25-run artifacts/comparisons/bm25/20261005T132419374140Z-e7896add
```

각 arm 경로:
{chr(10).join('- '+LABELS[arm]+': `'+result['arms'][arm]['run_dir']+'`' for arm in ARMS)}
'''
    (folder / "report.md").write_text(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--reference-bm25-run", type=Path)
    args = parser.parse_args(argv)
    started = perf_counter()
    baseline = read_json(args.baseline_run / "manifest.json")
    config = ExperimentConfig(**baseline["config"])
    snapshot_path = PROJECT_ROOT / baseline["snapshot"]["path"]
    interactions = load_snapshot(snapshot_path)
    texts_path = snapshot_path.with_name(snapshot_path.stem + ".reviews.jsonl")
    texts, texts_meta = load_review_texts(texts_path, interactions)
    texts_meta["path"] = str(texts_path.relative_to(PROJECT_ROOT))
    assert sha(snapshot_path) == baseline["snapshot"]["artifact_sha256"]
    model = BM25Candidate()
    base_config = BM25Config()
    if args.reference_bm25_run:
        ref_manifest = read_json(args.reference_bm25_run / "manifest.json")
        differences = {key for key in baseline["config"]
                       if ref_manifest["config"].get(key) != baseline["config"][key]}
        # The historical BM25 runner did not train a ranker. Its prefix/window
        # setting differs from R1 and does not alter candidate or label rules.
        assert differences <= {"ranker_training_mode"}, f"Candidate/evaluation settings differ: {differences}"
        config = ExperimentConfig(**ref_manifest["config"])
        assert ref_manifest["review_texts"]["artifact_sha256"] == texts_meta["artifact_sha256"]
        values = read_json(args.reference_bm25_run / "metrics.json")["refit"]["config"]
        base_config = BM25Config(**{k:v for k,v in values.items() if k not in ("name","profile_version","method","preprocessing")})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + baseline["snapshot"]["dataset_snapshot_id"][:8]
    folder = PROJECT_ROOT / "artifacts/comparisons/bm25_preprocessing" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    result = {"config": config.to_dict(), "baseline_run": str(args.baseline_run),
              "arms": {}, "test_contrasts_vs_baseline": {}, "checks": {}}
    # One fixed config per arm. Reuse the original candidate runner/caches.
    from dataclasses import replace
    for arm in ARMS:
        print(f"=== BM25 preprocessing: {arm} ===", flush=True)
        run = run_candidate_comparison(model, interactions, project_root=PROJECT_ROOT,
                artifacts_root=PROJECT_ROOT / "artifacts", texts=texts, texts_meta=texts_meta,
                config=config, grid=(replace(base_config, preprocessing=arm),),
                baseline_run=args.baseline_run, label=f"preprocessing-{arm}", plot=False)
        result["arms"][arm] = {"run_dir":str(run.run_dir.relative_to(PROJECT_ROOT)), "metrics":run.metrics}
        write_json(folder / "results.json", result)
    lists = {arm: {phase: {r['query_id']:r for r in read_jsonl(PROJECT_ROOT/result['arms'][arm]['run_dir']/f'candidates_{phase}.jsonl')}
                   for phase in ('validation','test')} for arm in ARMS}
    for phase in ('validation','test'):
        summaries = [result['arms'][arm]['metrics']['selection']['grid'][0] if phase=='validation' else result['arms'][arm]['metrics']['refit'] for arm in ARMS]
        assert len({s['retriever_preprocessing']['source_profile_sha256'] for s in summaries}) == 1
        reference_ids = lists['baseline'][phase]
        for arm in ARMS:
            assert lists[arm][phase].keys() == reference_ids.keys()
            assert all(lists[arm][phase][q]['c5_c1_lightgcn_rrf'] == reference_ids[q]['c5_c1_lightgcn_rrf'] for q in reference_ids)
        if args.reference_bm25_run:
            prior = {r['query_id']:r for r in read_jsonl(args.reference_bm25_run/f'candidates_{phase}.jsonl')}
            assert reference_ids == prior
            old = read_json(args.reference_bm25_run/'metrics.json')[phase]
            for stage in (*policies(model), 'c5_c1_lightgcn_rrf'):
                for key,value in old[stage].items():
                    assert abs(value-result['arms']['baseline']['metrics'][phase][stage][key]) < 1e-12
    result['checks'] = {'same_source_profiles_both_cutoffs':True, 'same_c5_candidates_both_windows':True,
                        'original_bm25_exactly_reproduced':bool(args.reference_bm25_run)}
    split = build_global_temporal_split(interactions, train_fraction=config.train_fraction, validation_fraction=config.validation_fraction)
    queries,_ = build_window_queries(split.train+split.validation,split.test,config=config,phase='test',cutoff=split.validation_cutoff)
    for arm in ARMS[1:]:
        result['test_contrasts_vs_baseline'][arm] = {stage: {f'at_{k}': paired_bootstrap(queries,
            {q:tuple(r[stage]) for q,r in lists[arm]['test'].items()},
            {q:tuple(r[stage]) for q,r in lists['baseline']['test'].items()},
            cutoff=k,samples=config.bootstrap_samples,seed=config.random_seed) for k in (10,100)} for stage in policies(model)}
    candidates = [(arm,stage) for arm in ARMS for stage in policies(model)]
    def preference(pair):
        arm, stage=pair
        val=result['arms'][arm]['metrics']['validation'][stage]
        return val['recall_at_100'],val['ndcg_at_10']
    chosen = max(candidates,key=preference)
    standalone = max(ARMS,key=lambda arm:preference((arm,'bm25')))
    result['validation_selection'] = {'standalone_preprocessing':standalone,'preprocessing':chosen[0],'policy':chosen[1],
                                     'rule':'validation Recall@100, NDCG@10, prespecified order'}
    result['elapsed_seconds']=perf_counter()-started
    write_json(folder/'results.json',result)
    write_json(folder/'manifest.json', {'kind':'controlled-bm25-preprocessing', 'arms':list(ARMS),
        'snapshot_sha256':sha(snapshot_path), 'review_texts_sha256':sha(texts_path),
        'baseline_run':str(args.baseline_run), 'reference_bm25_run':str(args.reference_bm25_run) if args.reference_bm25_run else None,
        'source_sha256':{name:sha(PROJECT_ROOT/'src/rating_recsys'/name) for name in
            ('experiments/bm25_preprocessing.py','retrieval/bm25.py','retrieval/review_preprocessing.py','retrieval/review_profiles.py')},
        'fixed_source_profiles':True, 'no_test_based_selection':True, 'ranker_retrained':False,
        'arm_runs':{arm:r['run_dir'] for arm,r in result['arms'].items()}})
    write_report(folder,result)
    print(f"Complete: {folder}",flush=True)
    return folder


if __name__ == '__main__':
    main()
