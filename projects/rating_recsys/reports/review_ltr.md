# 리뷰 임베딩을 LTR 피처로 사용하는 비교

C5 후보 100개를 고정하고 Nemotron 리뷰 유사도를 LambdaRank 피처로 추가하는 비교다. **2026-10-07 사용자 요청으로 실제 임베딩 추출을 보류했다.** 구현·로컬 검증은 완료했으며 부분 캐시를 보존했다. 추천 성능은 아직 평가하지 않았다.

## 비교 조건

| 항목 | 기존 LTR | 리뷰 LTR |
|---|---|---|
| 후보 생성 | C5: item-item + LightGCN RRF | 동일 후보·순서·수량 |
| 입력 피처 | 기존 16개 | 기존 16개 + 리뷰 피처 6개 |
| 학습·정답 | Window, 개인 이력 기반 만족도 | 동일 학습 행·query group·label |
| 모델 선택 | 동일 validation grid, NDCG@10 → Recall@10 | 동일 규칙 |
| 최종 학습 | T2까지 전체 Window 재구성 | 동일 기간, 부분 Window 중복 없음 |
| Test | 선택한 모델로 1회 | 선택한 모델로 1회 |

Snapshot `e7896add5b4b5939`, validation 입력 ≤2025-12-19, test 입력 ≤2026-05-04, seed 42를 유지한다. 3개월 학습 Window마다 시작일 이전 리뷰로 프로필을 만든다. T1/T2 프로필을 과거 학습 행에 넣지 않는다.

공유 C5 후보는 실제 데이터로 재생성했다. Test 1,580개 query·158,000개 행의 Recall@100 20.162567%, 후보 순서 Graded NDCG@10 0.026590111이 기존 기준과 일치한다. 이는 후보 검증이며 리뷰 LTR의 성능 결과가 아니다. 근거는 로컬 `artifacts/diagnostics/review_ltr_candidate_verification.json`이다.

## 임베딩 모델과 리뷰 처리

사용자 요청에 따라 OpenRouter `nvidia/nemotron-3-embed-1b:free`를 사용한다. 실제 응답은 2,048차원이며 확인한 호출 비용은 $0이다. Encoder를 학습하거나 미세조정하지 않는다.

최근 4점 이상 리뷰 중 사용자 최대 5개·식당 최대 10개를 묶는다. 리뷰당 240자, 문서당 최대 500토큰을 유지한다. 공식 토크나이저 `nvidia/Nemotron-3-Embed-1B-BF16`, revision `c0c9fea93ea424587517f2c59e20db9f1d6bf615`를 사용하고 파일 hash를 기록한다.

[공식 NVIDIA API](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-embed-1b-infer)의 역할 구분에 맞춰 사용자 입력은 `input_type=query`, 식당 입력은 `input_type=passage`로 분리해서 호출한다. 캐시 식별용 문자열은 역할 접두어를 포함하지만 실제 요청에서는 접두어를 제거하고 API 역할을 명시해 중복 접두어를 피한다. 숫자형 사용자·식당 ID는 API로 보내지 않는다.

2026-10-06 접두어만 붙인 시험 입력과 API 역할을 명시한 입력이 다른 벡터를 반환하는 것을 확인했다. 접두어만 쓴 시험 리뷰 벡터 128개는 캐시에 남아 있지만, 새 전처리 버전 `positive-recent-token-budget-api-roles-v3`의 key와 달라 최종 실험에 사용되지 않는다. Liquid의 기존 v2 key는 유지한다.

## 추가 피처

| 피처 | 의미 |
|---|---|
| `review_cosine` | L2 정규화 사용자·식당 벡터의 내적 |
| `review_user_present` | 사용자 프로필 존재 여부 |
| `review_item_present` | 식당 프로필 존재 여부 |
| `review_pair_present` | 양쪽 프로필이 모두 존재하는지 |
| `review_user_count` | 선택된 사용자 리뷰 수 |
| `review_item_count` | 선택된 식당 리뷰 수 |

누락 유사도 0은 존재 여부 피처로 실제 cosine 0과 구분하며 음의 cosine도 보존한다. 후보 식당 ID에 맞춰 행을 구성하고 기존 피처·label·후보를 변경하지 않는다.

리뷰 수는 토큰 잘림 전 선택된 리뷰 수다. 정보가 충분하다는 증거나 최소 corpus threshold가 아니다. 이번 비교는 긍정 리뷰 concat 조건이며 부정 리뷰·개별 리뷰 평균·속성 근거 결합은 추가하지 않는다.

## 실행 상태와 한도

| 단계 | 상태 |
|---|---|
| 2026-10-06 연결·차원·입력 역할 검증 | 완료, 실제 출력 2,048차원 |
| 전체 cutoff별 고유 프로필 입력 | 59,453개, 학습 Window 시작 40개 + validation/test |
| 2026-10-06 실제 추출 | 접두어 시험 128개; API 역할 수정 후 일일 한도 429로 중단 |
| 2026-10-07 재개 후 보류 | v3 입력 8,960/59,453개 저장, 50,493개 미추출; 사용자 요청으로 중단 |
| 큰 배치 검증 | 2,048개는 413, 1,024개는 400; 검증된 128개 배치 사용 |
| 전체 임베딩 완료·LTR 학습·성능 평가 | 아직 미완료 |

128개 배치에서 전체 최초 추출은 약 465회다. 재개 전 키 조회에는 일일 무료 요청 한도 50회·잔여 50회가 표시됐지만, 이번 실행에서는 70개 배치가 성공했다. 표시된 쿼터만으로 실제 임베딩 호출 상한을 확정하지 않는다. 이번 중단 사유는 사용자 요청이며 일일 한도 오류가 아니다.

전처리·모델·본문 hash가 같은 입력은 재호출하지 않는다. 캐시에는 v3 입력 8,960개와 시험용 v2 벡터 128개가 남아 있으므로 전체 SQLite 행 수를 v3 완료 수로 해석하지 않는다. 전체 프로필을 다시 점검해 실제 key 기준 누락량을 확인했다.

추출은 전체 벡터를 메모리에 쌓지 않고 배치마다 SQLite에 저장한다. Validation/test 입력을 먼저 처리하고 query/passage 배치를 번갈아 호출해 중단 시 양쪽 역할의 벡터를 남긴다. 호출 순서는 피처 내용·학습 label·평가 규칙을 바꾸지 않는다. 일일 한도 오류에서는 재시도하지 않으며, 수동 중단 시 남은 누락량을 진단 파일에 기록한다.

최신 누락량·API 사용량은 로컬 `artifacts/diagnostics/review_ltr_nemotron_preflight.json`, 캐시는 `artifacts/nemotron_review_embedding_cache.sqlite`에 저장한다. 진단 JSON에는 리뷰 본문을 넣지 않는다. 이전 Liquid의 10,505개 입력 캐시는 이 작업 환경에 없고, Git에서 제외된 로컬 파일이라는 점도 확인했다.

## 검증과 재현

임베딩·LTR·기존 pipeline/ranking 테스트 51개가 통과했다. 음/0/누락 cosine 구분, 후보 ID 정렬, 미래 리뷰 차단, 모델·전처리별 캐시 분리, query/passage API payload, 캐시 재개·메모리 절약 저장·수동 중단 기록을 확인했다. 합성 데이터에서는 두 모델의 학습·validation 선택·최종 refit·test·bootstrap·artifact 저장까지 실행했다. 합성 벡터의 수치를 실제 추천 성능으로 해석하지 않는다.

아래 명령은 보류 해제 후 재개할 때 사용한다. 프로젝트 디렉터리의 실험 의존성이 설치된 환경에서 실행한다. 첫 명령은 API 호출 없이 점검하고, 두 번째는 누락 벡터만 추출한다. 마지막 명령은 캐시를 읽으며 누락이 있으면 학습 전에 종료한다.

리뷰 본문·API 응답·벡터 캐시·모델·진단 결과는 Git에서 제외된 로컬 산출물이다. 커밋에는 코드·테스트·요약 문서만 포함되므로, 다른 환경에서 재개하려면 별도로 캐시와 snapshot을 옮겨야 한다.

```bash
python -m rating_recsys.experiments.review_ltr \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --embedding-model nvidia/nemotron-3-embed-1b:free --dry-run

python -m rating_recsys.experiments.review_ltr \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --embedding-model nvidia/nemotron-3-embed-1b:free \
  --embedding-batch-size 128 --embedding-concurrency 2 --embed-only

python -m rating_recsys.experiments.review_ltr \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --embedding-model nvidia/nemotron-3-embed-1b:free
```

결과는 `artifacts/comparisons/review_ltr/<run_id>/`에 저장한다. 기존 LTR 대비 Graded NDCG@10·Recall@10과 사용자별 paired bootstrap 2,000회 구간, 피처 중요도·프로필 coverage·실제 비용을 남긴다. Test 결과로 피처나 리뷰 수 threshold를 재선택하지 않는다.
