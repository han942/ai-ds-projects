# 리뷰 임베딩: retrieval 구조 검토와 실행 결과

## 현재 판단

**텍스트를 넣은 여러 검색 구조를 구현·비교했지만, 강한 행동 기반 후보 검색을 안정적으로 개선한 조건은 아직 없다.** 현재 기준은 공동 방문 기반 item-item(C1)과 방문 그래프 LightGCN(C4)을 RRF로 합친 **C5**다. RRF는 각 검색기의 순위를 합치는 방식이며, 최종 재정렬은 기존 LambdaRank를 보존한다.

1. 완료 범위

   - E5 리뷰 LTR, 초기 Two-Tower, 리뷰 Transformer 10epoch
   - RLMRec/E5, 동결 review mean/late, 속성·감정 10채널, 취향 K1/K3
   - 동일 Two-Tower 3seed, 텍스트 식당 그래프, 학습형 mean/late 3seed
   - 본문 존재 균형 교정

2. 현재 범위

   - 리뷰 텍스트를 이용한 전체 식당 후보 검색
   - 재정렬 후속 실험은 보류

3. 후속 질문

   - 미래 방문·만족도를 구분하는 텍스트 학습이 행동 검색을 보완하는가?

4. 최근 변경

   - 2026-10-12 코드 오류 6개 수정
   - 전체 테스트 512개 통과
   - 수정 후 실데이터 재학습·재임베딩·성능 평가는 미실행

본문 날짜는 KST 기준이다. Artifact의 `20261010T...Z` 같은 run ID는 UTC라 본문의 실행 날짜와 하루 차이가 날 수 있다. 아래 성능은 모두 **10월 12일 오류 수정 전 실행 결과**다.

<a id="지금까지-무엇을-시도했는가"></a>

## 실험 이력

1. E5 리뷰 LTR — 10/7

   - 변경: C5 재정렬에 E5 유사도·존재·리뷰 수 6피처 추가
   - 범위: validation 선택, test refit, seed 42
   - 결과: Test NDCG `0.025467 → 0.025570`
   - 판단: 차이 구간이 0을 포함해 개선 미확인

2. 초기 Two-Tower — 10/8

   - 변경: E5·ID·수치 입력의 독립 검색
   - 범위: 12epoch 중 9 선택, test refit, seed 42
   - 결과: Recall@100 validation 9.4419%, test 9.1193%
   - 판단: C5보다 낮아 미채택

3. 리뷰 Transformer — 10/9~10

   - 변경: 리뷰 블록 attention, Transformer, Top-200 cross-attention
   - 범위: 같은 설정으로 10epoch, validation, seed 42
   - 결과: 선택 epoch 2 검색 Recall 9.6714%, 재정렬 NDCG 0.007007
   - 판단: 학습 연장 후에도 최선 불변

4. RLMRec/E5 — 10/10

   - 변경: LightGCN 방문 학습에 E5 평균 프로필 정렬 추가
   - 범위: 5조건×20epoch, 과거 graph 38개와 ranker 7개 재학습
   - 결과: C1 결합 Recall 19.6248%, 최종 NDCG 0.028022
   - 판단: 두 개선 구간이 0을 포함

5. 동결 review mean/late — 10/11

   - 변경: 리뷰 평균과 리뷰쌍 Top-3 matching 비교
   - 범위: validation, 새 학습 없음
   - 결과: Recall 4.5032%→4.7866%
   - 판단: 차이 불확실, C5 quota는 낮아짐

6. 속성·감정 10채널 — 10/11

   - 변경: 맛·가격·서비스·분위기·양 × 긍정/부정
   - 범위: 4조건×20epoch, seed 42
   - 결과: 채널+C1 19.4210%, 평균 대조 19.6042%
   - 판단: 분리 이득 미확인

7. 여러 취향 K1/K3 — 10/11

   - 변경: 사용자 취향을 최대 3개로 분리
   - 범위: 동결 군집 검색, 새 신경망 없음
   - 결과: 텍스트 단독 4.5032%→5.5003%, 그래프 quota 19.4779%
   - 판단: 텍스트 단독 신호는 있으나 C5 보완 미확인

8. 구간별 진단 — 10/11

   - 변경: 기존 22조건을 사용자 이력·식당 방문 수별 재계산
   - 결과: 일부 탐색 신호
   - 한계: 신규 사용자 1,139명 제외

9. 동일 Two-Tower 대조 — 10/11

   - 변경: 같은 구조에서 E5만 0/실제 값으로 변경
   - 범위: seed 42/43/44, 고정 12epoch
   - 결과: 평균 Recall 9.1229%→9.3854%
   - 판단: 차이 구간이 0을 포함

10. 텍스트 그래프·학습형 리뷰 검색 — 10/11

   - 범위: 6조건×3seed, 균형 교정 review 9모델
   - 결과: 행동 C4 18.4481%, 실제 그래프 18.2001%, 교정 late 18.4514%
   - 판단: 두 텍스트 조건 모두 미채택

11. 코드 오류 수정 — 10/12

   - 범위: 회귀·전체 테스트, 새 성능 실험 없음
   - 결과: 512개 통과, 1개 skip
   - 판단: 과거 지표는 수정 전 결과로 유지

위 Recall은 Recall@100, NDCG는 Graded NDCG@10이다. 후보 순서와 LambdaRank/cross-attention 재정렬 후 NDCG를 구분한다. 구조 전체 비교와 같은 구조의 텍스트 추가효과 비교도 구분한다.

## 비교 기준

1. 데이터

   - Snapshot `e7896add5b4b5939`
   - Train cutoff T1: 2025-12-19
   - Validation 끝 T2: 2026-05-04

2. 검색 범위

   - 과거 catalog 4,404곳 전체 점수화
   - 이미 방문한 식당 제외
   - 최대 후보 100개
   - Transformer 상세 비교만 Top-200

3. 정확도 분모

   - Query 1,645명 중 관련 정답 사용자 1,639명
   - 본문 없는 사용자도 유지
   - Catalog 밖 정답 233쌍도 분모에 유지

4. 신규 사용자

   - 과거 이력이 없는 1,139명 제외
   - 이력 1–4건 결과는 기존 사용자 결과
   - Cold-start 검증으로 해석하지 않음

5. 과거 test

   - 이력 사용자 1,580명, 관련 정답 사용자 1,573명
   - 초기 LTR·Two-Tower·Liquid가 이미 조회
   - 새 설계의 독립 최종 증거로 재사용하지 않음

6. 신뢰구간

   - 사용자 paired bootstrap 2,000회
   - 3seed는 사용자별 seed 평균 차이를 재표본추출
   - 사용자×seed를 독립 표본으로 합치지 않음

7. 결론 범위

   - 반복 조회한 validation의 탐색 결과
   - 고정 학습 seed에 조건부인 구간
   - 다중 비교 보정과 새 독립 holdout 검증 없음

**본문 encoder와 후단 학습은 별개다.** 모든 E5 실험의 원문→384차원 변환기는 동결했다. 이후 Two-Tower MLP·Transformer·LightGCN 정렬·리뷰 projector는 학습했으므로 “텍스트를 전혀 학습하지 않았다”는 설명은 맞지 않는다. 원문 encoder 자체의 추천 목적 fine-tuning은 미실행이다.

## LTR 결과

**질문:** 이미 찾은 C5 후보 100개에서 리뷰가 순서를 개선하는가? 기존 16피처 LambdaRank에 E5 cosine·사용자/식당/쌍의 프로필 존재·선택 리뷰 수 6피처를 더했다. 후보·학습 행·정답·7개 설정의 선택 규칙을 같게 유지했다.

| 모델 | Validation NDCG@10 | Test NDCG@10 | Test Recall@10 |
|---|---:|---:|---:|
| C5 후보 순서 | 0.023311 | 0.026590 | 4.0279% |
| 기존 LTR | 0.027694 | 0.025467 | 3.6502% |
| E5 리뷰 LTR | 0.026229 | 0.025570 | 3.5170% |

| 리뷰 LTR − 기존 LTR, test | 차이 | 95% 구간 |
|---|---:|---|
| NDCG@10 | +0.000104 | [−0.003922, +0.004298] |
| Recall@10 | −0.1332%p | [−0.8099, +0.5710]%p |

**개선·악화를 확정하지 않는다.** 같은 C5 후보에서 6피처 묶음과 모델 선택을 함께 비교했으며 cosine만의 효과는 분리하지 않았다. 두 모델 모두 leaves 63/minchild 100을 선택했고 tree는 기존 76·리뷰 37개였다. T2의 8,043그룹·804,232행으로 refit 후 모델별 test 평가는 1회였다.

입력은 과거 4점 이상 리뷰 중 사용자 최신 5개·식당 10개를 각 240자까지 연결한 concat 프로필이다. 사용자 `query:`·식당 `passage:`, 최대 500토큰의 E5 벡터를 사용했다. 이 선택·압축·절대 4점 기준의 효과는 별도 통제하지 않았다.

근거: [실제 run·보고서][e5-ltr], [원 지표][e5-ltr-metrics], [저장 순위·IDCG·cosine 진단][ltr-diagnostics]. Early stopping과 보고 IDCG의 차이는 [10월 12일 코드 수정](#코드-오류-수정-2026-10-12)에서 바로잡았지만, 수정 후 tree 재선택·실데이터 비교는 미실행이다.

## Two-Tower 후보 비교

**질문:** E5를 입력한 작은 독립 검색기가 행동 C5를 대체할 수 있는가? 사용자·식당의 `ID 32차원 + 동결 E5 + 프로필 존재·리뷰 수`를 MLP로 각각 64차원 벡터로 바꾸고 내적으로 전체 식당을 검색했다. 사용자에는 과거 방문 수도 추가했다.

| 후보 조건 | Validation Recall@100 | Test Recall@100 | Test 후보 NDCG@10 |
|---|---:|---:|---:|
| C5: C1+LightGCN RRF | 19.4769% | 20.1626% | 0.026590 |
| E5 Two-Tower 단독 | 9.4419% | 9.1193% | 0.009962 |
| C1+E5 Two-Tower RRF | 15.3290% | 16.0897% | 0.018460 |

Test의 C5 대비 Recall 차이는 단독 −11.0432%p [−12.6135, −9.4552], 결합 −4.0728%p [−5.2298, −2.9666]이었다. **이번 설정은 미채택**한다. 새 후보용 LTR은 학습하지 않았고, 이미 본 test의 탐색 결과다.

- **학습:** seed 42·sampled-softmax·배치 512·음성 16개·lr 0.001. 최대 12epoch 중 validation Recall로 9를 선택하고 T2에서 9epoch refit했다.
- **차이:** LightGCN은 모든 과거 방문 간선을 학습한다. 이 모델은 과거 38시간창의 적격 후속 방문 44,806개를 학습해 사용자 ID가 받는 신호 범위도 다르다.
- **한계:** 구조·목표·수치 피처까지 바뀐 전체 모델 비교다. 이 결과만으로 E5의 추가 효과를 판단할 수 없다. 이후 [동일 구조 대조](#같은-two-tower의-텍스트-추가효과)를 완료했다.
- **비용:** 전체 약 7분·최대 RSS 1.128GiB·신경망 학습 합계 73.35초. 새 E5·API·baseline 재생성은 0회다.

근거: [원 지표][initial-tt-metrics], [세 조건 상세 보고서][initial-tt-report], [독립 검산][initial-tt-audit]. 2026-10-12 미참여 ID 처리 수정은 이 과거 v1 checkpoint에 포함되지 않는다.

## 두 안의 구체적 구현 설계

당시 두 안은 **리뷰를 그대로 읽는 Transformer**와 **의견·속성별로 읽는 Transformer**였다. 첫 안은 구현·10epoch 비교를 완료했고, 둘째 안 전체 구조는 미실행이다. 뒤의 규칙 기반 10채널→LightGCN 실행과 둘째 안을 구분한다.

| 항목 | 1안: 구현·실행 | 2안: 보류 설계 |
|---|---|---|
| 입력 | 사용자 최근 리뷰 20개·식당 30개, 리뷰당 최대 4개 문맥 블록 | 대상·의견·극성·근거를 가진 여러 record |
| 표현 | 동결 E5 384차원 + 과거 평점·시간 → 128차원 | 근거 벡터 + category·극성·시간·mask |
| 학습 | 블록 attention → 리뷰 self-attention 2층/4 heads → pooling·ID 결합 | 같은 속성 내부 집계 → 속성 Transformer·후보별 matching |
| 전체 catalog 검색 | 사용자·식당을 각각 한 벡터로 만든 내적 | 전체 속성 Transformer 검색은 미실행 |
| 상세 비교 | 검색 Top-200에서 양방향 cross-attention | 같은 category의 의견 matching 설계만 존재 |

입력 계약은 cutoff 이전 리뷰만 join하고, 타깃 방문의 리뷰는 사용자·식당 양쪽 입력에서 제외하는 것이다. 동결 벡터는 재사용하되 cutoff별 리뷰 선택·mask를 새로 구성한다. Concat 프로필 캐시에서 리뷰별 벡터를 복원할 수는 없다.

### 1안 전체 validation 결과

**질문:** 리뷰 사이 문맥과 후보별 대응을 학습하면 독립 추천 점수가 좋아지는가? 검색·matching head를 공동 학습했으며, 검색에는 gain 가중 sampled-softmax, 상세 비교에는 미관측 BPR와 관측 방문 사이 등급 BPR를 사용했다.

| 같은 validation, 선택 epoch 2 | Recall@100 | NDCG@10 |
|---|---:|---:|
| C5 후보 순서 | 19.4769% | 0.023311 |
| 기존 LambdaRank | 19.4769% | 0.027694 |
| Transformer 검색 head | 9.6714% | 0.010893 |
| 검색 Top-200의 cross-attention 재정렬 | 7.7692% | 0.007007 |

선택 기준은 validation **재정렬** NDCG@10이다. 같은 200개 후보를 재정렬하므로 양쪽 Recall@200은 14.7609%로 같다. 검색만의 최선 epoch 3을 선택 행에 섞지 않았다.

재정렬−검색 NDCG 차이는 −0.003885 [−0.006613, −0.001270], 재정렬−기존 LambdaRank는 −0.020687 [−0.025615, −0.015863]이었다. 후보 200개와 C5 100개가 달라 전체 시스템 비교이며, 선택된 validation의 차이 구간이다.

### 3epoch 판단과 식당 쏠림의 원인 조사

초기 3epoch로 결론을 내리지 않고 **모델·입력·손실·lr·seed를 유지해 10epoch까지 확인**했다. 최소 10·최대 20epoch, 재정렬 NDCG patience 5에서 10epoch 종료 조건을 충족했다. 처음 세 epoch와 선택 checkpoint는 초기 실행과 정확히 같았다.

| Epoch | 총손실 | 검색 Recall@100 | 재정렬 NDCG@10 | Top-10 고유 식당 | 최다 노출 식당의 사용자 비율 |
|---|---:|---:|---:|---:|---:|
| 1 | 2.2839 | 8.9035% | 0.003378 | 31 | 98.84% |
| 2 · 선택 | 2.2097 | 9.6714% | **0.007007** | 31 | 98.48% |
| 3 | 2.1567 | 9.8930% | 0.004589 | 70 | 98.18% |
| 6 | 1.8771 | 9.0415% | 0.006735 | 259 | 86.75% |
| 10 · 마지막 | 1.4997 | 6.6514% | 0.002968 | 1,054 | 46.38% |
| 기존 LambdaRank | — | 19.4769% | 0.027694 | 2,186 | 20.85% |

![같은 설정 10epoch의 손실·정확도·노출 변화](../artifacts/comparisons/review_transformer/20261009T155837960566Z-e7896add/learning_curve.png)

**쏠림 감소가 정확도 개선으로 이어지지 않았다.** 마지막 열은 매 epoch 가장 많이 노출된 식당을 따로 찾은 값이며, 노출은 전체 1,645명·정확도는 정답 사용자 1,639명 기준이다. 생략한 epoch도 [전체 곡선][transformer-long]에 남아 있다.

| 독립 진단 | Epoch 2 → 10 | 해석의 한계 |
|---|---|---|
| 검색 점수의 식당 공통 효과 | 97.77%→60.05% | 전체 catalog의 사용자 상수 제거 후 분산 분해; 인과 기여율이 아님 |
| 재정렬의 식당 공통 효과 | 99.03%→73.74% | 고정 96명×158개 식당 표본의 진단 |
| 사용자 E5를 0으로 바꾼 Top-10 유지율 | 97.71%→88.85% | 입력 개입이며 텍스트 없이 재학습한 대조군이 아님 |
| 사용자 ID를 unknown으로 바꾼 유지율 | 94.90%→63.54% | 긴 학습에서 ID 민감도도 커짐 |
| Cross-attention을 0으로 바꾼 유지율 | 90.52%→77.08% | 내부 연산이 쓰였다는 진단이며 성능 이득 증거가 아님 |

44,806개 양성 중 관측 등급 비교가 있는 예제는 11,118개(24.8%)였다. Uniform 미관측 식당으로 학습한 matching과 검색 Top-200의 추론 분포 차이, 손실 균형·개인별 비교 신호 부족이 원인 가설이다. 원인별 재학습 대조는 하지 않았다.

실행은 38개 과거 시간창·3,279,745개 학습 파라미터다. 초기 새 E5 171,382개 추론을 포함해 59.39분·최대 RSS 3.60GiB였고, 연장에서는 모두 cache HIT였다. 확정 10epoch 학습 합계 56.93분·평가 6.17분은 중단된 일부 epoch 재계산을 제외한 값이다.

근거: [초기 progress][transformer-initial], [연장·선택·전체 epoch][transformer-long], [학습 신호 감사][transformer-signal], [입력 개입 감사][transformer-representation], [선택 모델 동등성][transformer-equivalence], [순위·입력 재검산][transformer-verification]. Test 평가는 없었다.

## RLMRec 방식으로 베이스라인에 텍스트를 결합한 비교

**질문:** 독립 리뷰 추천기를 새로 만드는 대신, 강한 방문 그래프를 유지하면서 텍스트를 보완하면 어떤가? 기존 LightGCN의 BPR 방문 학습에 `λ × 동일 엔티티의 그래프–텍스트 대조 정렬 손실`을 추가했다. BPR는 방문 식당을 미방문 식당보다 높이는 학습이고, 대조 정렬은 같은 사용자/식당의 두 표현을 가깝게 만드는 학습이다.

[원 RLMRec](https://arxiv.org/html/2310.15950v5)은 LLM 프로필을 생성한다. 여기서는 **저장 E5 리뷰 평균을 사용하는 RLMRec-Con 적응판**이며 완전 재현은 아니다. 같은 초기값에서 통제 재학습했고, 추론은 학습한 LightGCN 내적→C1 RRF→LambdaRank 순서다.

### 입력과 비교 조건

- **행동:** 동일 70,883개 방문·64차원·3층·batch 2,048·lr 0.005·seed 42·고정 20epoch.
- **텍스트:** 모든 평점의 사용자 최근 리뷰 20개·식당 30개. E5 블록→리뷰별 평균→리뷰 동등 평균·L2 정규화로 엔티티마다 한 벡터.
- **학습:** 공유 384→224→64 MLP·InfoNCE, λ=0/0.001/0.01/0.1과 λ=0.001의 role별 profile shuffle. BPR 표본·초기화는 같고 shuffle은 coverage·벡터 집합·norm을 유지한다.
- **확인:** λ=0의 기존 C4·C5 순위가 정확히 재현됐다. 원 코드의 전체 negative pool·layer sum 대신 capped in-batch negative·기존 layer mean을 쓰므로 λ를 논문과 직접 비교하지 않는다.

| 고정 20epoch | 그래프 Recall@100 | C1 결합 Recall@100 | 결합 후보 NDCG@10 |
|---|---:|---:|---:|
| 기존 LightGCN, λ=0 | 19.2869% | 19.4769% | 0.023311 |
| E5 정렬, λ=0.001 | **19.4880%** | **19.6248%** | 0.024108 |
| E5 정렬, λ=0.01 | 18.4395% | 19.1412% | 0.023402 |
| E5 정렬, λ=0.1 | 10.3178% | 15.4712% | 0.018000 |
| Profile shuffle, λ=0.001 | 19.2921% | 19.4391% | 0.023293 |

λ=0.001 결합−C5의 Recall 차이는 +0.1479%p [−0.1948, +0.4927], NDCG 차이는 +0.000796 [−0.000087, +0.001744]다. Shuffle 대비 NDCG 차이도 +0.000814 [−0.000099, +0.001748]로 0을 포함한다. 강한 λ=0.1은 낮았으며 방문 목적을 방해했을 가능성은 가설로 남는다.

선택 λ=0.001로 **과거 38개 graph를 각 cutoff에서 재학습하고 새 후보·16피처·7개 LambdaRank를 재생성**했다. 단순히 기존 graph 가중치나 기존 LTR 행을 재사용한 비교가 아니다.

| 전체 validation 시스템 | Recall@100 | Recall@10 | 최종 NDCG@10 |
|---|---:|---:|---:|
| C5 + 기존 LambdaRank | 19.4769% | 4.1667% | 0.027694 |
| RLMRec/E5 + C1 + 새 LambdaRank | 19.6248% | 4.0645% | 0.028022 |

최종 NDCG 차이는 +0.000328 [−0.003864, +0.004325]로 불확실하며 Recall@10은 낮았다. **방문 모델 성능을 유지한 결합은 확인했지만 텍스트 의미가 개선했다고 확정할 수 없어 baseline을 유지**한다.

근거: [지표·λ 선택·전체 시스템][rlmrec-metrics], [20epoch 곡선][rlmrec-curve], [38개 cutoff·저장 모델 재검산][rlmrec-verification], [독립 감사][rlmrec-audit]. 전체 890.32초 중 과거 graph 315.89초·ranker 학습 21.39초였으며 새 E5·LLM/API·test는 없었다.

## 리뷰별 다중 벡터 후보 검색

**질문:** 리뷰를 먼저 한 벡터로 평균하기보다, 개별 리뷰쌍의 점수를 합치면 검색이 좋아지는가? 동결 E5 블록 store만 재사용하고 Transformer·ID·cross-attention의 학습 가중치는 쓰지 않았다. 신경망 학습이 없어 epoch 비교도 없다.

사용자 리뷰마다 식당 리뷰의 cosine **Top-3 평균**을 구하고, 사용자 리뷰 사이에서 동등 평균한다. 사용자 최근 20개·식당 30개·모든 평점·같은 review bank를 사용하며 4,404곳 전체에서 직접 후보를 만든다. Mean 대조군은 그 bank를 검색 전에 평균·정규화한다.

| 조건 | Recall@100 | 후보 NDCG@10 |
|---|---:|---:|
| C5 | 19.4769% | 0.023311 |
| 같은 bank mean | 4.5032% | 0.004186 |
| 리뷰별 late Top-3 matching | 4.7866% | 0.001908 |
| C5 + late 신규 후보 최대 10개(quota) | 18.6186% | 0.023311 |
| C5 + late 동일 가중 RRF | 14.5240% | 0.015529 |

Late−mean Recall 차이는 +0.2834%p [−0.6025, +1.1705]로 불확실하고 NDCG는 −0.002277 [−0.003755, −0.000928]로 낮아졌다. C5 quota는 −0.8583%p [−1.3032, −0.4156]이며 새 정답 26쌍을 찾는 동안 기존 72쌍이 빠졌다. Quota NDCG가 같은 것은 **C5 앞 90개를 유지**하기 때문이며 재정렬 검증이 아니다.

Matching 노출의 97.15%가 리뷰 10~30개 식당에 몰렸다(전체 catalog의 51.32%). 리뷰 수가 많을 때 가까운 리뷰를 찾을 기회가 늘어나는 편향과 맞지만 원인 통제 실험은 없다. 긍정/불만을 함께 쓰는 평균, 리뷰 수 보정, encoder 조정은 별도 가설이다.

근거: [지표·비교 구간][frozen-late-metrics], [bank·입력 hash][frozen-late-profiles], [전체 순위 검산][frozen-late-verification], [독립 감사][frozen-late-audit]. 전체 54.16초 중 점수 계산 30.18초이며 새 E5·API·ranker·test는 없었다.

## 속성·감정 채널과 여러 취향 후보 비교

두 방식은 정보를 평균하기 전에 나누는 후속이다. **10채널은 새 텍스트 정렬 학습**, **여러 취향은 동결 군집 검색**이므로 효과와 학습 범위를 구분한다.

### 속성·감정 10채널

맛·가격·서비스·분위기·양 × 긍정/부정 의견을 한국어 규칙으로 추출하고, 대상·의견·근거를 E5로 인코딩해 채널별 LightGCN 대조 정렬에 사용했다. 51,984개 리뷰에서 의견 94,505개를 추출했으며 긍정 71,098·부정 8,600개만 정렬에 사용했다.

사용자 최근 본문 리뷰 20개·식당 30개를 **추출 여부와 무관하게 먼저 선택**했다. 같은 MLP·10개 head·관측 mask·초기값·방문 BPR 표본·seed 42·20epoch·700 update를 유지한다. 평균 대조군은 관측 채널에 평균 벡터를 반복하고, shuffle은 채널 안에서 벡터만 섞어 결측과 분포를 보존한다.

| 조건 | 그래프 Recall@100 | C1 결합 Recall@100 |
|---|---:|---:|
| 텍스트 없는 C4 / C5 | 19.2869% | 19.4769% |
| 의견 평균 대조, λ=0.001 | 19.4096% | 19.6042% |
| 속성·감정 채널, λ=0.001 | 19.2099% | 19.4210% |
| 채널, λ=0.01 | 18.2869% | 19.1864% |
| 채널 shuffle, λ=0.001 | 19.0368% | 19.2897% |

채널+C1의 차이는 C5 대비 −0.0559%p [−0.3986, +0.2677], 평균 대비 −0.1831%p [−0.5482, +0.1629], shuffle 대비 +0.1313%p [−0.1962, +0.4384]로 모두 불확실하다. 평균+C1도 C5보다 점 추정만 높다. **4조건 모두 20epoch를 완료했으며 분리 이득은 미확인**이다.

당시 v2 추출기는 일부 간접 부정을 수정했지만 전체 라벨의 사람 평가를 완료하지 않았다. 부정·희소 채널 부족, 대상 연결·감정 오류, 방문 BPR와 만족도 평가의 차이는 가설이다. 학습형 ABSA·LLM 추출이나 속성 Transformer 재현으로 표현하지 않는다.

### 여러 취향 K1/K3

같은 최신 리뷰를 spherical k-means로 1개/최대 3개 군집화한 뒤, 취향별 검색 순위를 번갈아 합쳐 최종 최대 100개를 만든다. 텍스트 공간은 군집 리뷰 평균 cosine, 그래프 공간은 해당 방문 식당의 **동결 LightGCN 벡터 평균** cosine이다. ComiRec 같은 신경망을 새로 학습한 것은 아니다.

| 조건 | K1 Recall@100 | K3 Recall@100 | 핵심 비교·한계 |
|---|---:|---:|---|
| 텍스트 단독 | 4.5032% | 5.5003% | +0.9971%p [+0.2029, +1.7946] |
| 텍스트 + C1 RRF | 12.0753% | 13.2025% | 동일 가중 결합 참고 |
| 텍스트 C5 quota | 18.7312% | 18.7600% | K3 새 정답 28쌍·탈락 72쌍 |
| 그래프 단독 | 16.0110% | 16.1444% | +0.1334%p [−0.8514, +1.0752] |
| 그래프 + C1 RRF | 17.6510% | 17.9534% | C4의 학습 사용자 벡터·raw dot과 다른 검색기 |
| 그래프 C5 quota | 19.2999% | 19.4779% | K3−C5 +0.0010%p [−0.5310, +0.5764] |

텍스트 단독의 여러 취향 효과는 관측됐지만 강한 C5를 보완하지 못했다. 그래프 K3 quota는 새 정답 70쌍·탈락 72쌍이다. Macro Recall의 분모가 사용자마다 달라 단순 쌍 개수와 차이 방향이 다를 수 있다. K3는 내부 검색 횟수도 늘어 계산량까지 동일한 비교는 아니다.

근거: [전체 22조건·20epoch·CI][views-metrics], [학습 곡선][views-curve], [원문·부정·토큰 감사][views-opinions], [독립 프로필 집계][views-profiles], [군집 검산][views-interests], [checkpoint 최종 순위 감사][views-checkpoints]. 완료 실행은 780.18초이며 중단분까지 로컬 E5 입력 총 138,856개를 추론했다(현 v2 고유 입력 138,798개와 구분). Ranker·test·외부 API는 없었다.

## 같은 Two-Tower의 텍스트 추가효과

**질문:** 구조·행동 입력·학습을 같게 두고 E5만 더하면 이득이 있는가? 초기 세 조건의 전체 모델 비교를 보완하려고 행동만/행동+텍스트 **2조건×seed 42/43/44×고정 12epoch**를 새로 학습했다.

| 고정한 부분 | 설정 |
|---|---|
| 행동 입력 | 사용자·식당 ID, 방문 수·평균 평점; 사용자 서로 다른 방문 식당 수. 낮은 평점 포함 전체 과거 행동 사용 |
| 구조 | ID 32·MLP hidden 128/output 64·637,856 parameter. 양쪽 모두 384차원 텍스트 자리를 두고 행동 조건만 0 입력 |
| 학습 | 같은 38개 시간창·44,806개 양성·seed별 초기화·부정 샘플·순서·12epoch |
| 바꾼 입력 | 기존 과거 4점 이상 concat E5 벡터의 0/실제 값. 본문 존재·선택 리뷰 수는 양쪽에서 제외 |

| 3seed 평균 | 행동만 | 행동+텍스트 | 차이·95% 구간 |
|---|---:|---:|---|
| Recall@100 | 9.1229% | 9.3854% | +0.2625%p [−0.2390, +0.7743] |
| 후보 NDCG@10 | 0.009757 | 0.010285 | +0.000528 [−0.000818, +0.002004] |

| Seed | 행동 Recall@100 | 텍스트 Recall@100 | 차이 (%p) |
|---|---:|---:|---:|
| 42 | 8.9387% | 9.6356% | +0.6969 |
| 43 | 9.2625% | 9.3722% | +0.1097 |
| 44 | 9.1675% | 9.1484% | −0.0191 |

전체·각 seed의 Recall 구간이 모두 0을 포함해 **텍스트 추가 이득은 미확인**이다. 마지막 12epoch를 고정했으며 중간 최고 epoch를 선택하지 않았다. 이력이 적은 사용자 348명의 1–4건 구간도 세 seed 모두 불확실했다. 희소 식당 1–4건은 seed 44에서만 양의 신호가 있어 부분집단 개선으로 확정하지 않는다.

근거: [6조건 곡선·평균 CI][ablation-metrics], [seed별 구간 파일이 있는 run][ablation-run], [초기화·입력·샘플 hash 독립 검증][ablation-audit], [checkpoint 전체 순위·9,921개 검사][ablation-root]. 전체 618.02초이며 새 E5·API/DB·ranker·test는 없었다.

## 텍스트 식당 그래프와 학습형 다중 벡터의 실제 비교

**질문:** 텍스트로 식당 간 관계를 추가하거나 리뷰별 matching을 추천 손실로 학습하면 행동 검색보다 좋아지는가? 같은 seed의 C4를 기준으로 고정 마지막 epoch를 비교했다. 최초 18모델과 리뷰 유무 균형을 교정한 후속 9모델을 모두 보존한다.

| 항목 | 텍스트 식당 그래프 | 학습형 리뷰 mean/late |
|---|---|---|
| 표현 | 동결 E5 식당 평균의 cosine Top-10·양방향 binary kNN | 동결 리뷰 E5 → 공유 선형 384→32 projector → L2 |
| 행동 결합 | 매 층 방문 전파 + 0.1×텍스트 전파, 0–3층 평균 | 동일 seed의 동결 C4 내적 + 학습 계수×텍스트 점수 |
| 학습 | ID 64차원·BPR·20epoch·batch 2,048·1,027,776 parameter | projector·계수 12,289 parameter만 BPR·12epoch·batch 512 |
| 대조 | 방문 그래프만 / 실제 간선 / 식당별 차수 보존 재배선 | 동일 bank·초기화·샘플의 mean / late / shuffled late |
| 검색 | 전체 catalog 내적 | CF+텍스트 점수를 전체 catalog에서 직접 계산 |

Review bank는 모든 평점·사용자 최근 최대 8개·식당 12개다. Late는 사용자 리뷰마다 식당 리뷰 Top-3를 평균한 뒤 사용자 리뷰를 동등 평균한다. 양성 식당의 본인 리뷰를 사용자 bag에서, 현재 사용자의 리뷰를 양성·음성 식당 bag에서 제거하며 **두 후보에 같은 사용자 bag**을 사용해 복사 일치를 막는다.

재배선은 간선 37,448개·노드별 차수·원래 단위 가중치·본문 결측을 유지한다. 연결 성분·정규화 가중치 분포까지 같거나 균일한 무작위 그래프를 뽑은 것은 아니다. Shuffle은 role별 유효 벡터 자리만 섞어 mask·리뷰 수·소유권 제외 규칙을 유지한다. 두 계열은 epoch·batch·parameter가 달라 계열 간 비교는 전체 시스템 효과다.

### 전체 catalog 검색 결과

아래 리뷰 조건은 **본문 존재 균형 교정 후** 결과다. 세 seed 각각의 마지막 epoch에서 평가하고 평균했다.

| 3seed 평균 | Recall@20 | Recall@50 | Recall@100 | 후보 NDCG@10 |
|---|---:|---:|---:|---:|
| 행동 그래프만: 동일 seed C4 | 5.2696% | 10.9483% | **18.4481%** | 0.021067 |
| 행동+실제 텍스트 그래프 | 5.3971% | 11.1246% | **18.2001%** | 0.020848 |
| 행동+재배선 그래프 | 5.3037% | 11.0353% | 18.2623% | 0.021190 |
| 동결 C4+학습 mean | 5.3411% | 10.9926% | 18.4003% | 0.020890 |
| 동결 C4+학습 late | 5.2862% | 10.9552% | **18.4514%** | 0.021068 |
| 동결 C4+shuffled late | 5.2780% | 10.9459% | 18.4414% | 0.021069 |

| Recall@100 대조 | 차이 (%p) | 사용자 paired 95% 구간 (%p) |
|---|---:|---:|
| 실제 그래프 − 행동 그래프 | −0.2481 | [−0.4546, −0.0497] |
| 실제 그래프 − 재배선 | −0.0622 | [−0.2904, +0.1615] |
| 학습 late − 행동 그래프 | +0.0032 | [−0.0681, +0.0748] |
| 실제 late − shuffled late | +0.0100 | [−0.0617, +0.0851] |
| late − mean | +0.0511 | [−0.1165, +0.2197] |
| late − 실제 그래프 | +0.2513 | [+0.0524, +0.4525] |

| Seed | 행동 C4 | 실제 텍스트 그래프 | 균형 교정 late |
|---|---:|---:|---:|
| 42 | 19.2869% | 18.9697% | 19.2967% |
| 43 | 17.9092% | 17.6545% | 17.9092% |
| 44 | 18.1483% | 17.9760% | 18.1483% |

**실제 그래프는 주 지표 Recall@100에서 낮았고, late는 행동·shuffle·mean 대비 개선이 불확실해 두 조건을 채택하지 않는다.** 기존 C5 19.4769%는 seed 42 단일 기준이라 위 3seed 평균과의 차이를 텍스트만의 효과로 해석하지 않는다. Seed 42 행동 순위는 기존 C4와 정확히 같았다.

### 표현 집중과 리뷰 유무 교정

| 점검 | 원 실행의 관측 | 교정·남은 한계 |
|---|---|---|
| 본문 presence 비대칭 | 첫 epoch 70,883쌍 중 225쌍은 자기 리뷰 제외 후 음성에만 본문 존재; 반대 14/15/18쌍 | 양성·음성 모두 usable text가 있을 때만 두 semantic score 활성화. 비대칭은 0 |
| Late 성능 | 평균 18.4297%, 행동 대비 −0.0185%p [−0.0610, +0.0037] | 교정 18.4514%, 원 실행 대비 +0.0217%p [−0.0566, +0.1008]. 개선 불확실 |
| 출력 방향 집중 | 4,096개 사용자 리뷰 표본의 평균 단위 방향 norm² 0.9980/0.9998/0.9971 | 교정 0.9231/0.9992/0.9981. Seed 42만 일부 완화 |
| 행동 Top-100 보존율 | 99.9210/99.9848/99.8565% | 교정 99.1660/99.9982/99.9757%. 많은 순위가 그대로 남음 |

교정은 CF·BPR 쌍·초기화·샘플링·parameter를 유지하고 리뷰 9모델만 새로 학습했다. Graph/C4 9모델은 파일 hash까지 같은 결과를 재사용했다. 계수가 음수에서 양수로 바뀌어도 Recall 이득은 불확실했다. Mean seed 43은 후보 보존율 90.1422%여서 모든 조건을 근상수로 단정하지 않으며, **본문 유무 비대칭을 유일한 실패 원인으로 확정하지 않는다.**

### 같은 item-item 검색과의 결합

저장 Top-100을 같은 C1과 동일 가중 RRF로 결합한 보조 결과다. 재정렬은 없다.

| 3seed 평균 C1 결합 | Recall@100 | 같은 C1+행동 대비 차이·95% 구간 (%p) |
|---|---:|---|
| C1+행동 C4 | 18.9429% | 기준 |
| C1+실제 텍스트 그래프 | 18.9730% | +0.0301 [−0.1283, +0.1919] |
| C1+재배선 그래프 | 19.0005% | 대조군 |
| C1+학습 mean | 18.9992% | 참고 |
| C1+학습 late | 18.9414% | −0.0015 [−0.0353, +0.0314] |
| C1+shuffled late | 18.9226% | 대조군 |

결합의 실제 그래프−재배선과 late−shuffle 구간도 0을 포함한다. C1 결합에서도 의미 대응의 이득을 확인하지 못했다.

근거: [원 실행][families-original], [교정 지표·곡선·CI][families-balanced], [C1 결합][families-fusion], [전체 순위·400개 지표 독립 평가][families-evaluation], [학습 mask][families-mask], [출력 방향·점수 진단][families-score], [9개 checkpoint 전체 점수 복원][families-reconstruction], [scalar 7,938쌍 검산][families-scalar]. 최대 복원 순위 불일치는 0·scalar 오차 1.94×10⁻⁶이었다. 새 E5·API/DB·ranker·test는 없었다.

## 사용자 이력·식당 방문 수별 기존 결과

**새 실험이 아니라 기존 22조건의 저장 validation 순위를 재계산한 진단**이다. 구간 경계를 결과에 맞춰 바꾸지 않았으며 방문 수는 T1까지의 첫 사용자–식당 상호작용 수다. 신규 사용자 1,139명은 계속 제외한다.

| 사용자 과거 이력 | 정답 사용자 | C5 Recall@100 | 채널+C1 | 텍스트 K3 quota | 그래프 K3 quota |
|---|---:|---:|---:|---:|---:|
| 1–4건 | 348 | 18.8238% | 19.1456% | 18.4646% | 20.0991% |
| 5–9건 | 317 | 19.4487% | 19.5071% | 18.7004% | 19.2730% |
| 10–19건 | 410 | 21.2245% | 20.1358% | 20.2091% | 20.1899% |
| 20건 이상 | 564 | 18.6253% | 19.0231% | 17.9223% | 18.6923% |

이력 1–4건의 그래프 K3 quota−C5는 +1.2753%p [+0.1737, +2.6430]의 탐색 신호다. 같은 구간 채널+C1은 +0.3217%p [−0.1368, +0.8566], 텍스트 K3 quota는 −0.3592%p [−0.7902, −0.0479]로 이력 희소성이 모든 텍스트 방식의 이득을 뜻하지 않는다.

| 식당 과거 방문 수 | 해당 정답 사용자 | 정답 쌍 | C5 Recall@100 | 채널+C1 | 텍스트 K3 quota | 그래프 K3 quota |
|---|---:|---:|---:|---:|---:|---:|
| 0건·catalog 밖 | 191 | 233 | 0% | 0% | 0% | 0% |
| 1–4건 | 510 | 849 | 9.0310% | 9.0310% | 8.4931% | 8.6225% |
| 5–9건 | 717 | 1,105 | 11.7650% | 11.7998% | 11.6703% | 12.6513% |
| 10–19건 | 920 | 1,583 | 15.1705% | 14.9658% | 15.0382% | 15.7719% |
| 20건 이상 | 1,125 | 2,435 | 31.4637% | 31.5290% | 30.0570% | 30.7371% |

희소 식당 1–4건에서 그래프 quota−C5는 −0.4085%p [−1.0621, +0.1148]로 보완 이득이 없다. 사용자 구간은 사용자 집합을, 식당 구간은 해당 정답이 있는 사용자를 분모로 하므로 두 표를 단순 합산하지 않는다. 식당별 NDCG 기여량은 원래 위치·사용자 전체 IDCG를 유지해 합계가 원 지표와 같다.

근거: [22조건·구간 지표][segments-metrics], [획득·탈락 정답][segments-changes], [주요 조건 비교][segments-comparisons]. 원 정답·순위·hash·가중 합계 검사 106개를 통과했다. 여러 구간을 살핀 탐색이며 cold-start 성능이나 새 holdout의 효과를 검증한 것은 아니다.

## 코드 오류 수정 (2026-10-12)

확인한 구현 오류를 수정했다. **위 성능 표와 수정 후 코드의 효과를 구분한다.**

| 수정 경로 | 수정한 동작 | 기존 결과·재사용 영향 |
|---|---|---|
| 의견 추출 | `친절하지도 않다`, `맛있지도 않다`의 부정 범위 보존 | v3. 기존 v2 의견·채널 학습 결과는 수정 전 결과 |
| E5 formatter | 최소 토큰 예산 거부·잘림 중 역할 접두어 보존 | 정상 예산 cache identity 유지. 새 encoder 추론 없음 |
| Retrieval 비교 CLI | 그래프/리뷰 중 긴 epoch 예산까지 샘플 검증 기록 생성 | 리뷰 epoch가 더 길어도 허위 불일치로 실패하지 않음 |
| Two-Tower | 추천 학습 미참여 사용자·식당 ID를 공통 0번으로 추론 | 참여 ID를 v2 checkpoint에 기록. 과거 v1은 수정 효과 없음 |
| LTR early stopping | 전체 정답 IDCG·관련 사용자 분모·C5 동점 순서와 보고 metric 일치 | Baseline·리뷰 LTR·RLMRec·shrinkage 적용. 수정 후 tree 재선택 없음 |
| BM25 | 품사 필터에서 부정 표현 보존 | 전처리 모든 모드 색인 v2. [BM25 수정 안내](./bm25.md#부정-표현-보존-수정-2026-10-12) |

수정 검증은 **전체 테스트 512개 통과·1개 skip**, `git diff --check` 통과다. Train 원문 49개 리뷰의 직접 부정 오분류 50건을 대조해 negative/unknown 교정을 확인했고, 로컬 E5 tokenizer의 1토큰 예산 거부도 확인했다. 실데이터 재학습·재임베딩·새 test 평가는 없다.

이 수정으로 모든 감정 변형·BM25 부정 의미·평균 압축·식당 쏠림·목표 불일치까지 해결했다고 해석하지 않는다. 오류 수정의 정확도 효과는 새 실험으로 확인해야 한다.

## Retrieval에 한정한 구조 검토

### 적용 범위와 후속 가설

**이번 고정 E5 조건의 미채택은 텍스트 추천 전체의 불가능성을 뜻하지 않는다.** 고정 binary kNN은 LATTICE의 학습형 그래프 갱신을, 리뷰 선형 projector·Top-3 평균은 ColBERT의 본문 encoder fine-tuning·토큰 MaxSim을 재현하지 않았다.

| 다음 순서 | 확인할 질문 | 현재 상태 |
|---|---|---|
| 1 | 미래 방문과 만족도를 구분하는 텍스트 학습 목표가 효과적인가? | 미실행. 전체 과거 방문 BPR와 평가 목표 차이부터 설계 |
| 2 | 같은 CF에서 text-only/CF-only/결합·실제/shuffle·방향 집중을 통제하면 의미 효과가 남는가? | 현재 run을 기준으로 과거 rolling validation 또는 새 미래 holdout 필요 |
| 3 | 원문 encoder 추천 목적 조정 또는 학습형 그래프 이웃이 개선하는가? | 미실행. 목표·encoder·이웃을 한 번에 하나씩 변경 |

### Retrieval 전용 평가 계약

후보는 cutoff catalog 전체에서 직접 생성하고 **Recall@100을 주 지표**, Recall@20/50·후보 Graded NDCG@10·노출·비용을 보조로 보고한다. C5 보완은 새 정답 회수와 기존 정답 탈락을 함께 본다. 재정렬 지표로 검색 모델을 선택하지 않는다.

새 목표 학습은 타깃 리뷰·미래 방문·미래 식당 프로필을 양쪽 입력에서 제외한다. 신규 사용자·식당을 평가하려면 추천 시점에 이용 가능한 정보와 fallback을 먼저 정의한다. 반복 조회한 validation 결과는 탐색으로 표시한다.

### 진단과 후속

| 실제 확인한 사실 | 해석·남은 검증 |
|---|---|
| Validation cosine AUC 평균 0.515650, 725명 | 이번 긍정 concat의 정답/기타 후보 구분력이 약함. 기타는 실제 비선호가 아니며 인과 분석이 아님 |
| 벡터쌍 존재율 validation 98.5015%·test 97.3690% | 대량 누락만으로 결과를 설명하기 어려움 |
| 관련 사용자 909/1,639명(55.46%)은 C5 후보에 정답 없음 | 고정 후보 재정렬로는 복구 불가. 후보 안 oracle NDCG 0.242739는 미래 정답을 아는 진단 상한 |
| Validation 후보 99.253%는 미래 방문 미관측 | 방문 예측과 만족도 예측을 구분할 필요 |
| 후보 내부 IDCG와 전체 IDCG의 callback 불일치 | 10월 12일 코드 수정 완료. 성능 영향과 cosine/존재 피처 분리 비교는 미실행 |

<a id="두-retrieval-방법을-선택한-근거"></a>

텍스트 이웃 그래프와 학습형 late는 각각 관계 구조와 리뷰 집계의 효과를 검사하려고 선택했고 실제 대조를 완료했다. 다음에는 구조 이름을 늘리기보다 위 목표 가설을 먼저 검증한다.

## 재현

**현재 문서 정리는 저장 결과 읽기와 Markdown 편집이다. 새 실험을 실행하지 않았다.** 재현에는 로컬 snapshot·본문 sidecar·E5 모델/토크나이저·SQLite cache·prepared validation이 필요하다. 결과·원본·모델은 Git에 포함되지 않아 다른 환경에는 [보존 범위](../artifacts/README.md)에 맞춰 별도로 옮겨야 한다.

저장 LTR 결과의 읽기 전용 validation 검산:

```sh
.venv/bin/python -m rating_recsys.experiments.review_ltr_diagnostics \
  --run-dir artifacts/comparisons/review_ltr/20261007T060016724659Z-e7896add \
  --output /tmp/rating-e5-check.json \
  --review-cache artifacts/e5_review_embedding_cache.sqlite \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --tokenizer artifacts/tokenizers/614241f622f53c4eeff9890bdc4f31cfecc418b3.tokenizer.json
```

`--include-test`는 저장 test 순위 검산이며 모델 선택에 쓰지 않는다. 구간 진단은 `.venv/bin/python -m rating_recsys.experiments.segment_diagnostics`로 저장 순위만 재계산한다.

| 새 학습/검색을 실행하는 CLI | 재현하는 조건 |
|---|---|
| `review_ltr --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl` | 원 LTR 비교. `--dry-run` 입력/캐시 점검·`--embed-only` 누락 추론 |
| `compare_cli two_tower --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl --max-epochs 12 --patience 3 --threads 4 --no-plot` | 초기 세 조건 후보 비교. [원 보고서][initial-tt-report]·run manifest에 전체 옵션 보존 |
| `review_transformer_cli --epochs 20 --min-epochs 10 --patience 5 --threads 4 --embedding-batch-size 32` | Transformer 연장 조건. `--resume-run <run>`은 optimizer·RNG에서 재개 |
| `rlmrec_cli` | 고정 20epoch 정렬 비교 + 과거 graph·ranker 재학습 |
| `review_multivector_cli` | 동결 같은 bank mean/late·quota/RRF 검색 |
| `text_candidate_views_cli` | 규칙 의견 추출·10채널 학습·K1/K3 군집 검색 |
| `two_tower_ablation_cli --epochs 12 --seeds 42 43 44 --threads 4` | 동일 구조 텍스트 유무 6모델 학습 |
| `text_retrieval_comparison_cli --seeds 42 43 44 --graph-epochs 20 --review-epochs 12 --threads 4` | 그래프·학습 mean/late/shuffle. 기본은 본문 존재 균형 교정 |

위 모듈은 `.venv/bin/python -m rating_recsys.experiments.<CLI>`로 실행한다. 현재 코드는 10월 12일 수정이 포함돼 원 artifact와 동일한 새 수치를 보장하지 않는다. 과거 실행 소스는 각 run의 `source/` 또는 보존 소스를 확인한다.

그래프를 재사용해 리뷰만 재학습할 때는 `--reuse-graph-run artifacts/comparisons/text_retrieval_families/20261011T105026423770Z-e7896add`, 원 presence 정책에는 `--no-balance-text-presence`를 쓴다. 의미·초기화·샘플 통제가 같은 run인지 먼저 검사한다.

### CPU와 저장 공간

시간은 배치 실험 비용이며 사용자당 운영 검색 지연·ANN 비용이 아니다. 준비·추론·평가 포함 범위가 달라 모델 간 순수 fit 속도 비교로 사용하지 않는다.

| 실행 | 측정 범위·비용 |
|---|---|
| E5 concat 캐시 | 고유 입력 59,455개, 누적 encoder 60.1분·첫 추론~마지막 저장 66.1분; cache 약 500MiB·모델 471MiB |
| E5 LTR | 캐시 생성 제외 CLI 24.4분·관측 최대 child RSS 1.264GiB |
| 초기 Two-Tower | 전체 약 7분·최대 RSS 1.128GiB·학습 파라미터 678,112 |
| Transformer | 초기 새 블록 추론 포함 59.39분; 10epoch 확정 학습 56.93분·평가 6.17분 |
| RLMRec | 전체 890.32초·과거 graph 315.89초·ranker 21.39초 |
| 동결 mean/late | 전체 54.16초·전체 점수 계산 30.18초 |
| 채널·취향 | 완료 780.18초; K1/K3 검색·저장 8.50/15.69초 |
| 동일 Two-Tower 3seed | 전체 618.02초, 텍스트 조건 시간에 과거 concat bank 준비 포함 |
| Graph·학습 review 3seed | 최초 18모델 601.54초 + 그래프 재사용·교정 리뷰 9모델 312.89초; 최대 RSS 각 약 1.27GiB |

E5는 pinned `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`·384차원이다. Cache identity는 모델·revision·역할·전처리·본문 hash를 포함한다. 새 cutoff마다 동일 리뷰를 다시 인코딩하는 것은 아니다. E5 관련 완료 비교의 외부 임베딩 API 비용은 $0이었다.

## 임베딩 입력 재점검

2026-10-08의 읽기 전용 감사에서 전체 59,455개 캐시의 차원·유한값·단위 norm, validation 5,977개 입력 identity·모델·역할·revision을 확인했다. 사용자 8개·식당 8개만 CPU 재추론한 표본은 공식 attention-mask 평균 pooling·저장 벡터와 최대 차이 5.59×10⁻⁸이었다. 전체 재임베딩 검증으로 표현하지 않는다.

| T1 concat 입력 | 과거 4점 이상 본문 리뷰 | 최신 선택 후 | 선택 유지율 | 500토큰 추가 잘림 |
|---|---:|---:|---:|---:|
| 검증 사용자 | 27,640개 | 7,103개·최대 5개씩 | 25.70% | 86/1,628프로필·5.28% |
| 전체 과거 식당 | 59,915개 | 31,336개·최대 10개씩 | 52.30% | 723/4,352프로필·16.61% |

추가 토큰 잘림보다 최신 리뷰 선택과 여러 리뷰의 한 벡터 압축이 더 큰 입력 축소였다. 성능 저하의 원인 크기는 미검증이며, 부정어·감정 표현을 지우는 일반 문자열 청소를 개선책으로 단정하지 않는다. E5의 양쪽 `query:` 역할, concat과 동일 원문 리뷰별 평균, 평점·최근성 선택 조건은 아직 통제 비교가 남아 있다.

근거: [입력·forward 감사][input-audit], [캐시 전체 검증][cache-audit]. 당시 추가 학습·API·새 캐시 저장은 없었고 기존 cache SHA는 유지됐다.

## 과거 모델과 자료 상태

| 방법 | 실제 진행 | 현재 판단·원본 상태 |
|---|---|---|
| Liquid | 후보 검색만: test 단독 Recall 3.2757%, C1+C4+Liquid RRF 16.5637%(C5 20.1626%) | 결합 차이 −3.5989%p [−4.4422, −2.8003], 미채택. API 한도로 일부 집계/LTR 미완료; 사용자 정리로 cache·후보 삭제, 현재 재검산 불가 |
| Nemotron | 2,048차원·역할 연결 검증·부분 추출 8,960/59,453 기록 | 전체 LTR 미완료, 로컬 부분 cache 없음 |
| Gemma | CPU 표본 추론 비교 | E5보다 느렸던 표본 집계만 보존. 전체 추천 품질 미측정·설치/측정 코드 삭제 |
| DeepCoNN | V1의 문자 embedding·CNN 평점 예측 변형 | 과거 이력·코드는 있으나 현재 원 metrics 없음. V2 시간창 retrieval 개선 근거로 사용하지 않음 |
| RQ-VAE | 리뷰 벡터 이산화·이웃/추천 진단 | [별도 리포트](./rqvae.md). 원 run이 로컬에 없어 일부 결과는 실행 당시 기록, 이번 재검산 아님 |

자료가 없는 실험은 보고서의 과거 기록과 현재 직접 재검산 가능한 결과를 구분한다. [Artifact 보관 안내](../artifacts/README.md)를 기준으로 복원·새 실행 여부를 결정한다.

## 보류 설계와 문헌 조사

### 베이스라인을 유지한 텍스트 추가효과 검증

과거 C5 후보에 `행동 점수 + 신경망 텍스트 보정`을 더하려던 **미실행 재정렬 설계**다. 같은 Transformer·metadata·ID를 유지한 본문 0/실제 입력의 재학습 대조, hard-negative 학습, 검색 점수 잔차가 포함됐다. 2026-10-11 이후 retrieval 범위에서 보류하며 RLMRec·새 학습형 전체 catalog 검색과 구분한다.

### 텍스트 임베딩 방법 조사

2026-10-09에 원문·모델 카드로 검토한 후보이며 우리 데이터의 새 추천 성능 순위가 아니다. Encoder, 리뷰 집계, 추천 목적 학습을 별도 변수로 둔다.

| 검토 축 | 후보·근거 | 상태·비교 조건 |
|---|---|---|
| 가벼운 표현 | [TF-IDF+SVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html), [fastText](https://fasttext.cc/docs/en/unsupervised-tutorial.html) | 미실행. 어휘·IDF·SVD도 과거 cutoff 텍스트만 학습 |
| 문장 encoder | [한국어 SBERT](https://huggingface.co/jhgan/ko-sroberta-multitask), [E5 Base/Large](https://huggingface.co/intfloat/multilingual-e5-base), [BGE-M3](https://huggingface.co/BAAI/bge-m3), [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | E5 Small 외 추천 비교 미실행. 동일 원문·집계·길이에서 모델 전용 tokenizer/pooling/prompt·CPU 비용 확인 필요 |
| API encoder | [OpenRouter](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request), [Voyage](https://docs.voyageai.com/docs/embeddings), [Cohere](https://docs.cohere.com/v2/docs/cohere-embed) | 제공 모델·가격·버전은 실행 시 확인. 직접 API와 OpenRouter 제공 여부는 별개 |
| 리뷰 집계 | Concat·동일 bank mean/late·속성 채널·여러 취향 | 일부 완료. 평점/최근성 가중·LLM 근거 요약·동일 원문 concat 대 mean은 미실행 |
| 추천 학습 | 후단 projection/attention/정렬·본문 encoder fine-tuning | 후단 학습 완료, 원문 encoder 조정은 미실행 |

다른 모델을 설정 이름만 바꿔 실행할 수는 없다. [현재 backend](../src/rating_recsys/retrieval/review_embeddings.py)는 pinned E5·384차원을 검증하며 모델별 입력·차원·revision 계약을 구현해야 한다. Vector DB는 저장·검색 운영 선택으로, 표현·목표의 개선을 대신하지 않는다.

### 리뷰 텍스트와 다른 추천 모델의 결합

아래는 **논문 내부 비교와 구조 설명**이다. 평점 MSE 개선, 무작위/샘플 후보 평가, 대규모 사전학습 결과를 우리 미래 전체 catalog Recall과 직접 비교하지 않는다.

| 연구 | 결합 방식 | 논문에서 확인한 예·프로젝트 적용 한계 |
|---|---|---|
| [HFT](https://www-cs.stanford.edu/~jure/pubs/reviews-recsys13.pdf) / [EFM](https://yongfeng.me/attach/efm-zhang.pdf) | MF 요인과 topic 또는 명시적 속성·품질을 공동 학습 | HFT MSE 개선 5~10% 보고; EFM Yelp RMSE HFT 1.221→1.212. 평점 회귀 증거 |
| [NARRE](https://doi.org/10.1145/3178876.3186070) | 리뷰별 CNN·attention + ID → 평점 회귀 | [저자 코드](https://raw.githubusercontent.com/chenchongthu/NARRE/master/model/NARRE.py) 확인. 원 PDF 표 접근 실패로 수치 재인용 없음 |
| [MPCN](https://arxiv.org/html/1801.09251) / [CARP](https://arxiv.org/html/1907.00687) | 리뷰쌍 co-attention 또는 관점/aspect·감정 capsule | MPCN Yelp17 MSE 1.385→1.349; CARP Yelp16–17 1.593→1.508. 타깃 리뷰 제외·평점 오차 결과 |
| [UniSRec](https://arxiv.org/html/2206.05941) | 동결 상품 BERT·adaptor·ID + 방문 순서 Transformer | Pantry Recall@10 SASRec 0.0501→0.0693. 상품 설명·다중 도메인 사전학습·다음 항목 과제 |
| [APH](https://smufang.github.io/paper/WSDM25_APH.pdf) | 사용자–상품–속성–감정 hypergraph + ID·FM | Yelp MSE 1.1655→1.1467, Recall@5 0.7248→0.6996. 모든 지표가 개선된 것은 아님 |
| [재현 연구](https://arxiv.org/html/2005.12210) | 데이터 희소성·ID/bias·리뷰 모델 대조 | 희소 데이터에서 HFT가 더 나은 경우 확인. 복잡한 리뷰 구조의 우위를 일반화하지 않음 |

리뷰 경계·문맥 블록·대상/의견/극성 record·토큰별 표현은 서로 다른 단위다. [ASTE](https://aclanthology.org/2020.emnlp-main.183/) 같은 의견 추출과 문장 분할도 별개다. **리뷰별 late**는 문장 벡터쌍, [ColBERT](https://arxiv.org/abs/2004.12832)는 토큰별 MaxSim이므로 같은 구현으로 표현하지 않는다.

### 그래프와 텍스트 결합 연구

| 연구 | 텍스트의 결합 위치 | 논문 내부 비교·우리 실행과 차이 |
|---|---|---|
| [TextGCN](https://cdn.amazon.science/19/8d/4ecc938f4453aa6f285bf0a05a48/augmenting-graph-convolutional-networks-with-textual-data-for-recommendations.pdf) | 동결 LightGCN 점수 + MiniLM 텍스트 점수의 학습 결합 | Books Recall@20 0.1700→0.1833. 무작위 분할·SELU BPR 변경, 우리 LTR와 별개 |
| [RLMRec](https://arxiv.org/html/2310.15950v5) | 그래프–LLM 선호 프로필의 보조 대조/복원 손실 | Amazon-book Recall@20 0.1411→0.1483, 의미 벡터 단독 0.0199. 우리 E5 평균 적응판은 실행 완료 |
| [LT-HGCF](https://arxiv.org/pdf/2010.07027) | 리뷰·상품 설명을 이종 그래프 노드로 추가 | Beauty HR@20 0.5195→0.6068. 정답 1+비정답 99개 평가로 전카탈로그와 다름 |
| [LATTICE](https://arxiv.org/pdf/2104.09036) | 텍스트·이미지 item-item 그래프를 학습하고 ID 전파 | Sports Recall@20 0.0782→0.0915. 텍스트+이미지·학습형 관계로 고정 E5 binary kNN과 다름 |

현재 LightGCN과 텍스트의 결합은 **RLMRec 정렬·10채널 정렬·텍스트 식당 간선 추가까지 실행**했다. 문헌 사례는 이 실행의 개선을 보장하지 않으며, 학습형 이웃·원문 fine-tuning은 남은 별도 가설이다.

## 근거

핵심 지표는 아래 로컬 원 artifact에 연결했다. JSON·저장 순위·checkpoint·입력 감사의 역할을 구분하며, 재검산과 새 학습을 혼동하지 않는다.

[e5-ltr]: ../artifacts/comparisons/review_ltr/20261007T060016724659Z-e7896add/report.md
[e5-ltr-metrics]: ../artifacts/comparisons/review_ltr/20261007T060016724659Z-e7896add/metrics.json
[ltr-diagnostics]: ../artifacts/diagnostics/review_ltr_e5_followup.json
[initial-tt-metrics]: ../artifacts/comparisons/two_tower/20261008T002625755462Z-e7896add/metrics.json
[initial-tt-report]: ../artifacts/comparisons/two_tower/20261008T002625755462Z-e7896add/report.md
[initial-tt-audit]: ../artifacts/diagnostics/two_tower_e5_verification.json
[transformer-initial]: ../artifacts/comparisons/review_transformer/20261009T023235302075Z-e7896add/progress.json
[transformer-long]: ../artifacts/comparisons/review_transformer/20261009T155837960566Z-e7896add/progress.json
[transformer-signal]: ../artifacts/comparisons/review_transformer/20261009T023235302075Z-e7896add/training_signal_audit.json
[transformer-representation]: ../artifacts/comparisons/review_transformer/20261009T023235302075Z-e7896add/representation_audit.json
[transformer-equivalence]: ../artifacts/comparisons/review_transformer/20261009T155837960566Z-e7896add/diagnostic_checkpoint_equivalence.json
[transformer-verification]: ../artifacts/comparisons/review_transformer/20261009T155837960566Z-e7896add/verification.json
[rlmrec-metrics]: ../artifacts/comparisons/rlmrec/20261010T003719015651Z-e7896add/metrics.json
[rlmrec-curve]: ../artifacts/comparisons/rlmrec/20261010T003719015651Z-e7896add/learning_curve.png
[rlmrec-verification]: ../artifacts/comparisons/rlmrec/20261010T003719015651Z-e7896add/verification.json
[rlmrec-audit]: ../artifacts/comparisons/rlmrec/20261010T003719015651Z-e7896add/independent_audit.json
[frozen-late-metrics]: ../artifacts/comparisons/review_multivector/20261010T153953859126Z-e7896add/metrics.json
[frozen-late-profiles]: ../artifacts/comparisons/review_multivector/20261010T153953859126Z-e7896add/profiles_validation.json
[frozen-late-verification]: ../artifacts/comparisons/review_multivector/20261010T153953859126Z-e7896add/verification.json
[frozen-late-audit]: ../artifacts/comparisons/review_multivector/20261010T153953859126Z-e7896add/independent_audit.json
[views-metrics]: ../artifacts/comparisons/text_candidate_views/20261010T234421199403Z-e7896add/metrics.json
[views-curve]: ../artifacts/comparisons/text_candidate_views/20261010T234421199403Z-e7896add/training_recall.png
[views-opinions]: ../artifacts/comparisons/text_candidate_views/20261010T234421199403Z-e7896add/opinion_token_audit.json
[views-profiles]: ../artifacts/comparisons/text_candidate_views/20261010T234421199403Z-e7896add/opinion_profiles_independent_audit.json
[views-interests]: ../artifacts/comparisons/text_candidate_views/20261010T234421199403Z-e7896add/multi_interest_independent_audit.json
[views-checkpoints]: ../artifacts/comparisons/text_candidate_views/20261010T234421199403Z-e7896add/method1_independent_audit.json
[ablation-metrics]: ../artifacts/comparisons/two_tower_ablation/20261011T084326075960Z-e7896add/metrics.json
[ablation-run]: ../artifacts/comparisons/two_tower_ablation/20261011T084326075960Z-e7896add
[ablation-audit]: ../artifacts/comparisons/two_tower_ablation/20261011T084326075960Z-e7896add/independent_verification.json
[ablation-root]: ../artifacts/comparisons/two_tower_ablation/20261011T084326075960Z-e7896add/root_verification.json
[families-original]: ../artifacts/comparisons/text_retrieval_families/20261011T105026423770Z-e7896add/metrics.json
[families-balanced]: ../artifacts/comparisons/text_retrieval_families/20261011T110232785914Z-e7896add/metrics.json
[families-fusion]: ../artifacts/comparisons/text_retrieval_families/20261011T110232785914Z-e7896add/candidate_fusion.json
[families-evaluation]: ../artifacts/comparisons/text_retrieval_families/20261011T110232785914Z-e7896add/independent_evaluation.json
[families-mask]: ../artifacts/comparisons/text_retrieval_families/20261011T110232785914Z-e7896add/independent_training_mask_audit.json
[families-score]: ../artifacts/comparisons/text_retrieval_families/20261011T110232785914Z-e7896add/independent_score_diagnostics.json
[families-reconstruction]: ../artifacts/comparisons/text_retrieval_families/20261011T110232785914Z-e7896add/independent_balanced_score_verification.json
[families-scalar]: ../artifacts/comparisons/text_retrieval_families/20261011T110232785914Z-e7896add/root_scalar_verification.json
[segments-metrics]: ../artifacts/diagnostics/segments/20261011T084211161233Z-e7896add/metrics.json
[segments-changes]: ../artifacts/diagnostics/segments/20261011T084211161233Z-e7896add/positive_changes.jsonl
[segments-comparisons]: ../artifacts/diagnostics/segments/20261011T084211161233Z-e7896add/selected_comparisons.json
[input-audit]: ../artifacts/diagnostics/e5_embedding_input_audit.json
[cache-audit]: ../artifacts/diagnostics/e5_ltr_cache_verification.json
