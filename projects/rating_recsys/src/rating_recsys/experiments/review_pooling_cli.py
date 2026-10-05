"""Prespecified review pooling ablations against an immutable concat run.

Run the two fixed strategies with the shared comparison runner. All selection
functions use validation only; reporting both fixed test ablations does not
adapt the configuration or tune a new candidate policy on test.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.experiments.candidate_models import ReviewEmbeddingsCandidate
from rating_recsys.experiments.compare_cli import build_parser, configs_from_args, load_or_fetch_texts
from rating_recsys.experiments.comparison import run_candidate_comparison
from rating_recsys.experiments.pipeline import paired_bootstrap
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import dataset_digest, review_texts_path


def _read(path):
    return json.loads(path.read_text())


def _candidate_lists(path, stage='review_embeddings'):
    return {row['query_id']:tuple(row[stage]) for row in
            (json.loads(line) for line in path.read_text().splitlines())}


def _macro_recall(queries, candidates):
    values=[]
    for query in queries:
        positive={iid for iid, grade in query.relevance_by_item.items() if grade>0}
        if positive:
            values.append(len(positive & set(candidates[query.query_id]))/len(positive))
    return sum(values)/len(values)


def main(argv=None):
    model=ReviewEmbeddingsCandidate()
    parser=build_parser(model)
    parser.prog='rating-recsys-review-pooling'
    parser.set_defaults(plot=False)
    parser.add_argument('--concat-run',type=Path,required=True,
                        help='Same-snapshot original concat comparison run')
    args=parser.parse_args(argv)
    config,grid=configs_from_args(model,args)
    if len(grid)!=1 or args.max_epochs!=1:
        parser.error('pooling ablations use one fixed config and one fitting step')
    prior_manifest=_read(args.concat_run/'manifest.json')
    prior_metrics=_read(args.concat_run/'metrics.json')
    if json.loads(json.dumps(config.to_dict()))!=prior_manifest['config']:
        raise ValueError('Evaluation config must exactly match the concat run')
    from rating_recsys.experiments.cli import _load_interactions
    interactions=_load_interactions(args.snapshot)
    snapshot_id=dataset_digest(interactions)
    if snapshot_id!=prior_manifest['snapshot']['dataset_snapshot_id']:
        raise ValueError('Snapshot must match the concat run')
    texts_path=args.review_texts or review_texts_path(args.snapshot)
    texts,text_meta=load_or_fetch_texts(texts_path,interactions)
    if text_meta['artifact_sha256']!=prior_manifest['review_texts']['artifact_sha256']:
        raise ValueError('Review texts must match the concat run')
    original=prior_manifest['model']['grid'][0]
    for field in ('model','dimensions','max_user_reviews','max_item_reviews',
                  'max_review_chars','max_document_tokens','min_rating','tokenizer_revision'):
        if getattr(grid[0],field)!=original[field]:
            raise ValueError(f'Input setting {field} must match concat')
    split=build_global_temporal_split(interactions,train_fraction=config.train_fraction,
                                     validation_fraction=config.validation_fraction)
    queries,_=build_window_queries(split.train+split.validation,split.test,config=config,
                                  phase='test',cutoff=split.validation_cutoff)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+snapshot_id[:8]
    suite=args.artifacts_dir/'comparisons/review_profile_pooling'/run_id
    suite.mkdir(parents=True)
    variants=(('own_reviews_mean','own_reviews'),('liked_items_mean','liked_items'))
    state={'status':'running','created_at':datetime.now(timezone.utc).isoformat(),
           'original_run':str(args.concat_run.resolve()),'snapshot_id':snapshot_id,
           'config':config.to_dict(),'prespecified_variants':[name for name,_ in variants],
           'selection_rule':'highest validation standalone Recall@100; ties NDCG@10; never test',
           'variants':{}}
    def save():
        temp=suite/'progress.tmp'
        temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
        temp.replace(suite/'comparison.json')
    save()
    print(f'[suite] {suite}',file=sys.stderr,flush=True)
    old_candidates=_candidate_lists(args.concat_run/'candidates_test.jsonl')
    old_c5=_candidate_lists(args.concat_run/'candidates_test.jsonl','c5_c1_lightgcn_rrf')
    sources={'concat':old_candidates}
    results={}
    try:
        for name,user_profile in variants:
            state['active_variant']=name; save()
            print(f'[suite] starting {name}',file=sys.stderr,flush=True)
            variant_config=replace(grid[0],aggregation='review_mean',user_profile=user_profile)
            result=run_candidate_comparison(
                ReviewEmbeddingsCandidate(),interactions,project_root=PROJECT_ROOT,
                artifacts_root=args.artifacts_dir,texts=texts,texts_meta=text_meta,
                config=config,grid=(variant_config,),eval_every=1,patience=1,
                baseline_run=args.baseline_run,label=name,
                command=[parser.prog,*(argv if argv is not None else sys.argv[1:])],
                log=lambda message:print(message,file=sys.stderr,flush=True),plot=args.plot,
                use_cache=not args.no_cache,rebuild_cache=args.rebuild_cache,
            )
            results[name]=result
            sources[name]=_candidate_lists(result.run_dir/'candidates_test.jsonl')
            c5=_candidate_lists(result.run_dir/'candidates_test.jsonl','c5_c1_lightgcn_rrf')
            assert c5==old_c5, 'C5 candidate orders must match exactly'
            assert set(sources[name])=={q.query_id for q in queries}
            catalog={row.restaurant_id for row in split.train+split.validation}
            for query in queries:
                ranked=sources[name][query.query_id]
                assert len(ranked)<=config.candidate_k and len(ranked)==len(set(ranked))
                assert set(ranked)<=catalog
                assert not (set(ranked)&{row.restaurant_id for row in query.history})
            expected=result.metrics['test']['review_embeddings'][f'recall_at_{config.candidate_k}']
            assert math.isclose(_macro_recall(queries,sources[name]),expected,abs_tol=1e-12)
            row={'run_dir':str(result.run_dir.resolve()),'validation':result.metrics['validation'],
                 'test':result.metrics['test'],'selection':result.metrics['selection'],
                 'refit':result.metrics['refit'],
                 'audit':{'same_query_ids':True,'c5_orders_equal_original':True,
                          'no_duplicates_or_visited_or_future_items':True,'manual_recall_matches':True}}
            state['variants'][name]=row; save()
            print(f'[suite] completed {name}',file=sys.stderr,flush=True)
        comparisons={}
        for left,right in [('own_reviews_mean','concat'),('liked_items_mean','concat'),
                           ('liked_items_mean','own_reviews_mean')]:
            comparisons[f'{left}_vs_{right}']=paired_bootstrap(
                queries,sources[left],sources[right],cutoff=config.candidate_k,
                samples=config.bootstrap_samples,seed=config.random_seed)
        state['paired_comparisons']=comparisons
        candidates=[('concat',prior_metrics['validation']['review_embeddings'])]+[
            (name,result.metrics['validation']['review_embeddings']) for name,result in results.items()]
        winner=max(candidates,key=lambda row:(row[1][f'recall_at_{config.candidate_k}'],row[1][f'ndcg_at_{config.ranking_k}']))[0]
        state['validation_selected_user_representation']=winner
        state['original_metrics']=prior_metrics
        state['status']='complete';state.pop('active_variant',None);save()
        _write_report(suite,state,config)
    except BaseException as error:
        state['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        state['error_type']=type(error).__name__;state['error']=str(error)
        save();raise
    print(json.dumps({'report':str(suite/'report.md'),'selected_on_validation':winner},ensure_ascii=False))
    return state


def _write_report(suite,state,config):
    k=config.candidate_k;rk=config.ranking_k
    original=state['original_metrics']
    rows=[('기존 C5',original['validation']['c5_c1_lightgcn_rrf'],original['test']['c5_c1_lightgcn_rrf']),
          ('본문 결합: 내 리뷰',original['validation']['review_embeddings'],original['test']['review_embeddings'])]
    labels={'own_reviews_mean':'리뷰별 평균: 내 리뷰','liked_items_mean':'리뷰별 평균: 좋아한 식당'}
    for name,row in state['variants'].items():
        rows.append((labels[name],row['validation']['review_embeddings'],row['test']['review_embeddings']))
    lines=['# 리뷰 임베딩 집계 방식 비교','',
           f"Validation 기준 선택: **{state['validation_selected_user_representation']}**. 두 새 설정을 실험 전에 고정하고 test를 이용한 추가 설정 변경은 하지 않았다.",'',
           f'| 방법 | Validation Recall@{k} | Test Recall@{k} | Test 후보순서 NDCG@{rk} |',
           '|---|---:|---:|---:|']
    for label,val,test in rows:
        lines.append(f"| {label} | {val[f'recall_at_{k}']:.4%} | {test[f'recall_at_{k}']:.4%} | {test[f'ndcg_at_{rk}']:.6f} |")
    lines += ['', '## 고정한 조건과 달라진 부분','',
              '- 같은 스냅샷·리뷰 본문·시간 분할·history-aware relevance·후보 100개·Liquid 1,024차원 모델을 사용했다. 식당 이름·메뉴 필드·지역 필터는 추가하지 않았다.',
              '- 사용자 최근 4점 이상·본문 있는 리뷰 이벤트 최대 5개, 식당 최근 같은 조건의 리뷰 최대 10개를 유지했다. 리뷰당 앞 240자를 사용했다.',
              '- 본문 결합은 전체 프로필을 500토큰으로 제한한다. 리뷰별 평균은 각 리뷰를 500토큰으로 제한하고 각 벡터를 L2 정규화 → 단순 평균 → 다시 L2 정규화한다. 따라서 집계와 함께 보존되는 본문 양도 달라진다. 순수 평균 연산의 효과만 분리한 실험은 아니다.',
              '- 내 리뷰 평균: 사용자 리뷰는 query:, 식당 리뷰는 document:로 인코딩한다.',
              '- 좋아한 식당 평균: 위와 동일한 사용자 리뷰 이벤트에 해당하는 식당의 document 벡터를 평균한다. 식당 벡터는 전체 과거 사용자들의 최근 리뷰 최대 10개로 구성한다. 같은 식당의 반복 방문은 이벤트별 가중치를 유지한다. 이 비교는 사용자 정보 원천과 역할(query/document)도 함께 바꾼다.',
              '- API 배치 최대 128개, 동시 요청 2개, 분당 18회이며 성공한 배치는 즉시 재사용 가능한 SQLite 캐시에 저장한다.',
              '- 후보 순서의 NDCG이며 ranker 재학습 결과가 아니다. 이미 관찰한 test에 대한 후속 탐색이므로 최종 일반화 검증에는 새 holdout이 필요하다.',
              '', '## RRF 결합 결과','',
              f'| 표현 | Validation 선택 정책 | Validation Recall@{k} | Test Recall@{k} |',
              '|---|---|---:|---:|']
    for name,row in state['variants'].items():
        policy=row['selection']['chosen_policy']
        lines.append(f"| {labels[name]} | `{policy}` | {row['validation'][policy][f'recall_at_{k}']:.4%} | {row['test'][policy][f'recall_at_{k}']:.4%} |")
    lines += ['', '## 검증과 근거','',
              '- 두 실행의 모든 test query ID와 C5 후보 순서가 원 실험과 일치한다. 후보 중복·방문 식당·시점 이후 카탈로그 식당이 없으며 Recall을 수동으로 재계산해 확인했다.',
              '- `comparison.json`에 설정, validation/test 전체 지표, 사용자 단위 paired bootstrap 비교(2,000회), API 사용량과 각 실행 경로를 보존했다.',
              f"- 원 실험: {state['original_run']}"]
    for name,row in state['variants'].items():
        lines.append(f"- {labels[name]}: {row['run_dir']}/report.md")
    (suite/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
