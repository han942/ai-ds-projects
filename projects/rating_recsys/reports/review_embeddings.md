# 리뷰 임베딩: E5와 LTR

**현재 판단: E5 Small과 전체 캐시는 유지하되, 이번 리뷰 피처 LTR은 기존 ranker를 대체하지 않는다.** CPU·저장 비용은 실용적이었지만 추천 개선은 확인되지 않았다. 이는 리뷰 자체나 E5 모델 전체가 무효라는 결론은 아니다. 측정·평가·독립 재검증은 2026-10-07 완료했다.

## LTR 결과

같은 C5 후보 100개에서 기존 16피처 LambdaRank와 리뷰 피처를 더한 22피처 모델을 비교했다.

| 모델 | Validation NDCG@10 | Test NDCG@10 | Test Recall@10 |
|---|---:|---:|---:|
| C5 원래 후보 순서 | 0.02331129 | 0.02659011 | 4.027937% |
| 기존 LTR | 0.02769438 | 0.02546657 | 3.650223% |
| E5 리뷰 LTR | 0.02622946 | 0.02557046 | 3.517040% |

Test 유효 사용자 1,573명에서 paired bootstrap 2,000회, seed 42로 계산한 리뷰 LTR − 기존 LTR 차이다.

| 지표 | 차이 | 95% 구간 |
|---|---:|---|
| NDCG@10 | +0.00010389 | [-0.00392160, +0.00429821] |
| Recall@10 | -0.133183%p | [-0.809860, +0.571015]%p |

두 구간 모두 0을 포함해 개선·악화를 확정하지 않는다. 두 LTR 모두 test에서는 C5 순서보다 점 추정이 낮지만 이 차이의 구간도 0을 포함한다. Validation에서는 두 LTR 모두 C5보다 높았다. 한 기간·한 학습 seed의 결과이며 신규 사용자 cold-start 효과는 검증하지 않았다.

## 실험 조건

- Snapshot `e7896add5b4b5939`, 80%/10% 시간 분할, validation 입력 ≤2025-12-19, test 입력 ≤2026-05-04, seed 42. Test 이력 사용자 1,580명 중 관련 정답이 있는 1,573명을 정확도 집계에 사용한다.
- 같은 C5 후보·기존 피처·학습 행·group·label을 유지했다. 3개월 Window 시작 이전의 리뷰만 입력으로 쓰며, T1/T2 프로필을 과거 학습 행에 재사용하지 않는다.
- 두 모델에 같은 validation grid 7개와 선택 규칙 NDCG@10 → Recall@10을 적용했다. 모두 `num_leaves=63`, `min_child_samples=100`, tree 수는 기존 76개·리뷰 37개를 선택했다. T2 전체 Window를 8,043 group·804,232행으로 다시 학습한 뒤 test 예측·평가는 모델별 1회 수행했다.
- 리뷰는 4점 이상 최신 사용자 5개·식당 10개, 리뷰당 240자를 연결(concat)하고 최대 500토큰으로 제한한다. 사용자 `query:` / 식당 `passage:` 접두어를 사용한다.
- 추가 피처는 cosine 유사도, 사용자·식당·쌍의 프로필 존재 여부, 사용자·식당의 선택 리뷰 수다. 누락 cosine 0은 존재 여부로 구분하고 음수도 보존한다. 리뷰 수는 토큰 잘림 전 개수다.

피처 연결·벡터 순서·동일 후보·cutoff 점검에서 오류나 미래 입력 누수를 발견하지 못했다. 최종 모델이 cosine을 실제 사용했지만 사용 여부나 중요도는 성능 개선의 증거가 아니다. 이번 비교는 6피처 묶음과 모델 선택을 함께 평가했으며 cosine만의 효과는 분리하지 않았다.

## 진단과 후속

세 서브 에이전트의 독립 검토 후 저장 순위·읽기 전용 validation 캐시로 재확인했다. **진단에는 새 학습·ranker 예측·임베딩 생성·API 호출이 없다.**

| 확인한 사실 | 의미와 한계 |
|---|---|
| 사용자 내 cosine AUC 평균 0.515650, 725명 | 정답과 기타 후보의 구분력이 약하다. 각 사용자에서 모든 정답·기타 쌍을 비교하고 동점은 0.5로 센 평균이다. 기타 후보는 실제 비선호가 아니라 미관측 또는 relevance 0이다. 인과 효과·유의성 검정은 아니다. |
| Validation 벡터쌍 존재율 98.5015%, test 97.3690% | 대량 누락이나 피처 미연결만으로 결과를 설명하기 어렵다. |
| Early stopping은 후보 내부 IDCG, 보고 지표는 전체 정답 IDCG | 최적 정렬 점수로 정규화하는 기준이 달라 사용자를 재가중한다. Validation callback 대상 730명 중 607명에서 불일치한다. 저하의 원인인지는 미검증이다. |
| Validation 관련 사용자 909/1,639명, 55.46%는 후보에 정답 없음 | 후보 고정 ranker만으로 회복할 수 없다. 후보 안 정답을 완벽하게 정렬한 oracle NDCG@10은 0.242739로, 검색뿐 아니라 재정렬 여지도 있다. Oracle은 미래 정답을 아는 진단 상한이다. |
| Validation 후보 99.253%가 미래 방문 미관측 | Label 0 대부분은 실제로 싫어한 식당이 아니다. 방문 예측과 만족도 예측을 구별해야 한다. |

Early stopping에서 정답 없는 사용자를 빼는 것 자체는 최적 iteration을 바꾸지 않는다. 그 사용자는 항상 0이기 때문이다. 핵심은 사용자마다 다른 후보/전체 IDCG다. 같은 미튜닝 150트리 설정에서는 validation NDCG가 리뷰 0.025030 > 기존 0.024387이었다. 따라서 리뷰가 항상 나쁘다고 단정할 근거도 없다.

**다음 실험은 미실행**이며 아래 순서로 진행한다.

1. Early stopping의 전체 정답 IDCG를 보고 metric과 맞추고 synthetic regression test로 일치 여부를 검증한다.
2. 같은 캐시·C5·학습 예산으로 validation에서 `baseline`, `cosine+pair mask`, `존재 여부·개수만`, `전체 6피처`를 비교한다. 고정 tree 비교도 보조로 두어 모델 선택의 영향을 구분한다.
3. 필요하면 개별 리뷰 평균·max/mean matching·긍정/부정 프로필을 소규모로 비교한다. 새 임베딩이 필요한 단계이므로 먼저 비용을 확인한다. E5 후보 검색의 C5 보완 효과는 별도 실험 축으로 둔다.

이미 확인한 test를 새 설정 선택에 재사용하지 않는다. 새 holdout이나 과거 rolling validation이 없으면 후속 결과는 탐색 결과로 표시한다. 리뷰를 한 문서·cosine 하나로 압축하는 표현 손실은 가능한 원인이지만 아직 검증된 원인은 아니다.

## CPU와 저장 공간

현재 모델은 `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`다. 동결된 CPU float32·384차원·배치 8·6스레드로 로컬 파일만 읽으며 API 키가 필요 없다.

| 실제 전체 실행 | 결과 |
|---|---|
| LTR용 고유 입력 / 캐시 누락 | 59,455개 / 0개 |
| 배치 commit / 누적 encoder 추론 | 7,432회 / 60.1분, 16.48개/초 |
| 첫 배치 추론부터 마지막 저장 | 66.1분; 중간 재개 점검 포함, 최초 입력 구성 제외 |
| 벡터 캐시 / 모델 | 약 500MiB / 471MiB |
| LTR 비교 CLI | 24.4분; 앞선 임베딩 생성 제외, 새 벡터 생성 0개 |
| LTR 관측 최대 child RSS | 1.264GiB, 0.25초 간격 측정 |
| 임베딩 API 호출 / 비용 | 0회 / $0 |

59,455개는 cutoff별로 구성한 고유 프로필 입력이다. 수집 리뷰 96,922개를 각각 임베딩했다는 뜻이 아니다. SQLite 전체 무결성과 모든 벡터의 차원·유한값·norm을 확인했다. 캐시는 모델·revision·전처리·본문 hash로 구분하고 배치별 저장 후 누락분만 재개한다.

별도 i5-10400 표본 시험에서는 Gemma보다 약 7배 빨랐다. 같은 본문 256개 리뷰·64개 프로필, float32·배치 8·4/6스레드·2회 반복에서 E5는 리뷰 60.98~66.56개/초, 프로필 19.67~22.21개/초였다. Gemma는 각각 8.59~9.57개/초, 2.82~3.19개/초였다. 표본은 70%/15% 분할의 과거 입력이며 LTR의 80%/10% 평가와 다르다. 라이브러리 환경도 달라 순수 모델 구조의 배율로 일반화하지 않는다.

표본 최대 RSS는 E5 1.52GiB(배치 32 시험 포함), Gemma 2.54GiB였다. 표본 기반 전체 프로필 환산 44~51분과 실제 66.1분은 구분한다. 빠른 추론은 추천 품질 개선의 증거가 아니다.

## 과거 모델

Liquid는 **후보 검색만** 평가했으며 아래 결과는 새 후보용 LTR을 학습한 성능이 아니다. 같은 snapshot·cutoff·K=100·seed 42·test 유효 사용자 1,573명 조건이다.

| 과거 후보 조건 | Test Recall@100 | 후보 순서 NDCG@10 |
|---|---:|---:|
| 기준 C5 | 20.1626% | 0.026590 |
| Liquid 리뷰 임베딩 단독 | 3.2757% | 0.002979 |
| C1+LightGCN+Liquid RRF | 16.5637% | 0.028920 |

선택 결합의 C5 대비 Recall 차이는 -3.5989%p, 95% 구간 [-4.4422, -2.8003]%p로 미채택했다. 개별 리뷰 평균·선호 식당 평균 비교는 API 한도로 미완료다. 이후 Liquid LTR 추출 재개도 첫 호출 429로 중단돼 새 벡터가 없었다. 사용자 정리 요청으로 Liquid 캐시·설치·토크나이저·`embeddings.zip`을 삭제했으며 현재 해당 벡터·후보를 재검산할 수 없다.

Nemotron은 연결·2,048차원·API 역할 검증과 부분 추출 기록만 있고 전체 LTR 평가는 미완료다. 당시 8,960/59,453개 기록과 달리 현재 로컬 부분 캐시는 없다. Gemma는 CPU 표본 추론만 측정했으며 전체 추출·LTR 품질 비교는 하지 않았다. 설치와 측정 코드는 삭제했고 소량의 집계 JSON만 남겼다. E5 본 캐시·모델·원본 데이터·현재 평가 결과는 보존한다.

## 재현

프로젝트 디렉터리에서 실행한다. **저장 결과 재검증**은 학습·예측·API 없이 validation 순위와 E5 캐시를 읽는다. 원 지표·후보 순서·snapshot·tokenizer hash가 다르면 실패한다.

```sh
.venv/bin/python -m rating_recsys.experiments.review_ltr_diagnostics \
  --run-dir artifacts/comparisons/review_ltr/20261007T060016724659Z-e7896add \
  --output /tmp/rating-e5-check.json \
  --review-cache artifacts/e5_review_embedding_cache.sqlite \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --tokenizer artifacts/tokenizers/614241f622f53c4eeff9890bdc4f31cfecc418b3.tokenizer.json
```

`--include-test`는 저장된 test 순위의 검산에만 쓰며 모델 선택에 사용하지 않는다.

**원래 LTR 조건 재실행**은 아래 명령이며 CPU 학습 시간이 필요하다. 위 미실행 후속 조건이나 새 독립 test를 구성하는 명령은 아니다. `--dry-run`은 입력·캐시 점검만, `--embed-only`는 누락분 로컬 추출만 수행한다.

```sh
.venv/bin/python -m rating_recsys.experiments.review_ltr \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl
```

표본 CPU 측정은 [benchmark_e5_cpu.py](../scripts/benchmark_e5_cpu.py)에 `--snapshot`, `--output`, `--cache-dir artifacts/local_models`를 지정한다. 이 측정도 추천 성능 평가와 구분한다.

## 근거

- [실제 E5 LTR run](../artifacts/comparisons/review_ltr/20261007T060016724659Z-e7896add/report.md): 같은 폴더의 `metrics.json`, `manifest.json`, `selection.json`, 모델·저장 순위가 원본이다.
- `artifacts/diagnostics/review_ltr_e5_followup.json`: 저장 순위·validation cosine·후보 회수·IDCG 진단.
- `artifacts/diagnostics/e5_ltr_cache_verification.json`, `review_ltr_e5_cpu_run.json`, `e5_ltr_result_verification.json`: 전체 캐시·실행 시간·모델/순위 검산.
- `artifacts/diagnostics/e5_small_cpu_benchmark.json`, `embeddinggemma2_cpu_benchmark.json`: CPU 표본 집계. Liquid의 과거 요약은 위 표와 보존된 상세 리포트 ZIP에만 남는다.

정리 후 전체 테스트 199개와 실제 E5 캐시·저장 결과 재검산을 통과했다. 이번 리포트 통합은 학습·모델·캐시·원 실험 artifact를 변경하지 않았다. 로컬 결과·원본 데이터·모델은 Git에 포함되지 않으므로 다른 환경에서 재현하려면 별도로 옮겨야 한다. [현재 파일 보존 범위](../artifacts/README.md).
