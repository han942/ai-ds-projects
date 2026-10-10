# RQ-VAE: 리뷰 임베딩을 이산 코드로 바꾼 효과

질문: 리뷰의 연속 임베딩을 짧은 이산 코드로 바꿨을 때, 저장량과 표현 보존, 미래 식당 검색·순위가 어떻게 달라지는가?

**현재 판단: Supabase 본문 리뷰 94,301개 전체의 이산 코드 생성과 validation 비교를 완료했다. 코드와 공유 모델의 tensor 저장량은 크게 줄었지만, 원래 이웃 구조의 손실이 크고 추천 개선은 확인되지 않아 기존 추천 모델을 유지한다.** 학습·추천 비교는 과거 리뷰만 사용했고, 이후 리뷰에는 같은 모델을 고정 적용했다.

## 실행 결과

실행은 `20261010T065109Z`다. 과거 본문 리뷰 **69,359개** 중 실제 E5 입력이 같은 그룹을 함께 분리해 fit 62,456개·audit 6,903개를 사용했다. 아래 표현 보존 수치는 학습에 넣지 않은 audit 리뷰 기준이다.

| 표현 보존·저장량 | 결과 | 해석 |
|---|---:|---|
| 코드 복원 cosine | 0.953475 | fit 평균만 복원해도 0.932875이므로 이 값만으로 세부 의미 보존을 판단하지 않음 |
| 코드 복원 MSE | 0.000236301 | 평균 복원 오차 0.000337876보다 30.06% 낮음 |
| 원래 Top-10 이웃 overlap | 10.42% | audit 1,000개 × fit reference 10,000개 비교. 실제 의미 정답·추천 Recall과 다른 진단 |
| 같은 RQ 모델의 양자화 우회 overlap | 20.37% | 학습된 encoder/decoder에서도 원래 이웃 변화가 큼. 독립 AE 대조가 아님 |
| 단계별 codebook 사용 | 213 / 211 / 223개, 각 256개 중 | 사용률 83.20% / 82.42% / 87.11%; 가장 큰 단일 코드 비중은 1.17% 이하 |
| 과거 리뷰의 3단계 고유 코드 | 67,563개 | 리뷰 69,359개 중 초과 중복 1,796개. 코드가 고유 review ID라는 뜻은 아님 |
| 서로 다른 실제 입력 그룹의 초과 중복 | 1,212개 / 68,775그룹 (1.76%) | 같은 본문 반복·512토큰 절단 후 동일 입력을 제외한 코드 중복. 의미 충돌 주석은 미실행 |
| 과거 float32 벡터 payload | 106,535,424 bytes | 69,359 × 384 × 4 |
| 코드 + 공유 모델 payload | 736,593 bytes | 코드 208,077 + 모든 state tensor 528,516; **144.63배 작음** |

저장량은 ID·컨테이너·라이브러리·인덱스·연구용 원래 벡터 캐시를 제외한 계산이다. 이번 작업에서는 비교를 위해 원래 벡터도 보존하므로 실제 프로젝트 디스크 사용량이 144.63배 줄었다는 의미는 아니다. 추천 평가는 코드를 복원한 dense 벡터를 사용했으며, 코드 직접 검색의 속도 개선은 측정하지 않았다.

추천 validation은 query 1,645명 중 정답이 있는 **1,639명**의 동일 분모를 사용했다. 과거 이력이 없는 신규 validation 사용자 1,139명은 기존 프로토콜에 따라 제외했다. 텍스트 프로필 coverage는 세 표현 모두 사용자 10,961명·식당 4,403곳·query 사용자 1,640명으로 같다.

| validation 경로 | 원래 E5 NDCG@10 | 코드 복원 NDCG@10 | 원래 E5 Recall@100 | 코드 복원 Recall@100 |
|---|---:|---:|---:|---:|
| 텍스트 dense 검색 | 0.003213 | 0.003456 | 0.045222 | 0.035930 |
| C5 후보의 텍스트 cosine 재정렬 | 0.016649 | 0.016368 | 0.194769 | 0.194769 |
| C5 + 텍스트 dense RRF | 0.015637 | 0.017623 | 0.146405 | 0.144810 |
| C5 자체, LambdaRank 이전 | 0.023311 | — | 0.194769 | — |

같은 경로에서 코드 복원−원래 E5의 NDCG@10 차이와 paired bootstrap 95% 구간은 dense `+0.000243 [-0.001934, 0.002637]`, C5 텍스트 순서 `−0.000281 [-0.003885, 0.003524]`, RRF `+0.001986 [-0.001651, 0.005851]`이다. **세 구간 모두 0을 포함해 추천 개선을 확인하지 못했다.** Dense Recall@100은 점 추정상 20.55% 낮지만 해당 차이의 신뢰구간은 이번에 계산하지 않았다.

원래 E5와 코드 복원의 모든 경로는 C5 자체보다 NDCG@10이 낮았고, 각각의 paired 구간도 0보다 아래였다. C5 텍스트 재정렬의 Recall@100이 같은 것은 후보 집합을 유지하기 때문이다. RRF 결합은 K=100에서 행동 후보 일부를 대체하므로 후보 Recall이 낮아질 수 있다. 기존 **C5 + LambdaRank의 최종 지표**와 이번 C5 후보 순서를 혼동하지 않는다.

같은 RQ 모델의 연속 경로 NDCG@10은 dense 0.002747·C5 텍스트 순서 0.015819·RRF 0.016192였다. 원래 E5 대비 paired 구간은 이 경로도 모두 0을 포함한다. 반복 seed·다른 코드 예산 비교는 실행하지 않았다.

### Supabase 전체 코드 적용

2026-10-10 15:54 KST의 `READ ONLY / REPEATABLE READ` 조회를 사용했다. DB 전체 96,922개 중 **본문 94,301개**에 `uint8[3]` 코드를 생성했고 본문 없는 2,621개는 제외했다. 과거 모델 hash를 유지했고 추가 RQ 학습은 0회, DB 쓰기도 0회다. 리뷰 ID와 코드의 연결은 로컬 NPZ에 보존했다.

| 전체 적용 항목 | 결과 |
|---|---:|
| 고유 3단계 코드 | 90,569개 |
| 리뷰 기준 초과 코드 중복 | 3,732개 / 94,301개 (3.96%) |
| 고유 전체 본문 | 93,449종 |
| 고유 전체 본문 기준 초과 코드 중복 | 2,880종 (3.08%); token 절단 후 같은 입력도 포함 |
| 384차원 float32 payload | 144,846,336 bytes |
| 코드 + 공유 모델 tensor payload | 811,419 bytes; 원래 벡터보다 약 178.51배 작음 |
| 실제 ID 포함 압축 NPZ | 421,091 bytes; 공유 model.pt는 별도 파일 |

전체 고유 본문 93,449종의 접두어 포함 토큰 길이 P50/P90/P95/P99는 41/134/193/305.52이고 512토큰 초과는 102종 (0.11%)이다. 과거에 새로 만든 E5 입력 68,777종은 정확히 같은 본문 hash 정책으로 재사용하고 나머지 24,672종을 추가 인코딩했다. 전체 적용의 embedding 단계 423.39초는 재사용 경로를 포함하므로 처음부터 전체를 임베딩하는 시간은 아니다.

이 전체 적용 통계에는 미래 리뷰와 재방문도 있다. **추천 validation·표현 audit 결과는 위의 과거 입력에서 측정한 수치**이며, 미래 리뷰의 의미 보존이나 미래 추천 효과를 전체 적용 건수만으로 검증했다고 해석하지 않는다.

## 해석과 미실행 후속

확인한 문제는 **평균 cosine과 실제 이웃 보존의 차이**, 그리고 **텍스트 유사도만으로 행동 기반 후보 순서를 바꿀 때의 성능 손실**이다. 원래 E5 경로 자체도 C5보다 낮으므로 이번 추천 결과 전체를 이산화만의 문제로 설명할 수 없다.

원인 가설은 복원 MSE가 공통 문장 성분의 보존을 선호하고, 32차원 bottleneck·양자화·리뷰 평균이 선호에 중요한 차이를 충분히 보존하지 못한다는 것이다. 양자화를 우회해도 이웃 overlap이 낮은 점은 학습된 변환도 추가 조사할 근거다. 속성·긍정/부정 의미 손실을 주석으로 확인하거나 독립 AE를 학습하지 않았으므로 인과 원인으로 확정하지 않는다.

다음은 이번에 실행하지 않은 비교다.

1. 같은 코드 예산에서 원래 이웃 또는 추천 선호 거리를 보존하는 contrastive loss를 추가한다.
2. latent 차원 또는 단계·코드 수를 한 요인씩 바꾸고 표현 손실과 validation을 함께 비교한다.
3. C5의 행동 점수를 유지한 채 리뷰 코드를 보조 feature로 학습한다. 코드 자체가 순위 점수를 대체할 이유는 아직 없다.
4. 리뷰 코드의 시간 순서를 사용자 sequence encoder에 넣는 설계를 검토한다. 식당 ID 생성 모델은 별도 실험이다.

## 무엇을 코드로 바꾸는가

`리뷰 본문 → 고정 E5 임베딩 384차원 → 학습 encoder 32차원 → 잔차 양자화 3단계 → 코드 3개` 순서다. 예를 들어 `(17, 82, 4)`는 각 단계 코드북에서 선택한 벡터의 번호다. 코드 벡터를 합친 뒤 decoder를 통과하면 원래 384차원 공간의 근사 벡터를 복원할 수 있다.

각 단계는 이전 단계가 남긴 오차를 양자화한다. 숫자를 연속 실수처럼 평균하거나 코드 번호의 차이를 의미 거리로 쓰지 않는다. 첫 코드가 맛, 둘째가 가격 같은 해석을 자동으로 갖는 것도 아니다.

[TIGER](https://arxiv.org/abs/2305.05065)는 **식당/상품 같은 추천 대상**의 표현을 Semantic ID로 바꾸고, 별도의 생성 모델이 그 ID를 예측한다. 이번은 사용자가 지정한 **리뷰별 코드** 실험이다. 코드가 같은 리뷰를 하나로 삭제하지 않으며, 코드 자체는 고유 review ID나 식당 ID가 아니다. 생성형 추천 Transformer는 이번에 학습하지 않는다.

RQ-VAE의 잔차 양자화·encoder/decoder·codebook/commitment loss는 [원 RQ-VAE 연구](https://arxiv.org/abs/2203.01941)와 [TIGER의 Semantic ID 학습](https://arxiv.org/html/2305.05065v3#S3.SS1)에 따른다. 모델 크기·입력·optimizer·학습 예산은 이 데이터에 맞춘 변경이다.

## 입력과 비교 조건

| 항목 | 이번 조건 |
|---|---|
| 과거 입력 | 기존 interaction snapshot의 첫 상호작용 중 작성 event_date ≤ 2025-12-19, 모든 평점 유지 |
| 리뷰 임베딩 | `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`; 384차원 float32·L2 정규화 |
| 역할·길이 | 전체 리뷰에 `passage:` 한 역할; 리뷰별 최대 512토큰. 이전 RLMRec의 query/passage·문장 블록 캐시와 다른 입력 정책 |
| 기존 캐시 | 대부분 합성 프로필이라 재사용하지 않음. 새 리뷰별 로컬 E5를 생성, 외부 embedding API 호출 0회 |
| RQ 학습·검증 | 실제 접두어/512토큰 절단 후 E5 token IDs가 같은 리뷰를 하나의 그룹으로 묶어 과거 내부 fit 90% / audit 10% 분리 |
| 모델 | encoder 384→128→32, decoder 32→128→384, 단계별 별도 codebook 256개 × 3단계 |
| scaling | fit 리뷰의 평균과 전역 RMS만 사용. 평균 벡터 복원 기준도 비교해 공통 성분만 복원하는 착시를 줄임 |
| 최적화 | AE warmup 5epoch → fit latent 잔차 KMeans 초기화 → RQ 학습 40epoch; Adam 0.001, batch 512, commitment β=0.25, seed 42, CPU 4threads |
| 모델 선택 | 정해진 예산의 마지막 epoch 사용. 추천 validation 지표로 checkpoint·하이퍼파라미터 선택하지 않음 |
| 추천 평가 | T1 이후~2026-05-04 validation. history-aware 만족도, min history 10·평균 가중치 0.5·이동 한도 0.5. Test 성능 미계산 |
| 프로필 | 모든 평점의 최근 사용자 리뷰 최대 20개·식당 30개. 각 리뷰를 먼저 L2 정규화한 동등 평균 후 프로필 L2 정규화 |
| 후보 | 과거 catalog·방문 식당 제외·K=100 동일. 사용자 본문 없으면 인기 fallback; C5 텍스트 재정렬에서 결측 식당의 위치 유지 |
| DB 전체 적용 | 과거 fit에 고정한 RQ-VAE를 이후 리뷰에도 업데이트 없이 적용. 미래 리뷰 코드는 앞선 추천 평가 입력에 넣지 않음 |

역사적 event_date로 과거 경험을 재구성한 비교다. 리뷰가 실제 DB에 적재된 시점은 2026년 9월이므로, 당시 DB에서 이용 가능했던 텍스트라고 주장하지 않는다.

## 효과를 어떻게 구분하는가

| 비교 | 측정 | 답하는 질문 |
|---|---|---|
| 원래 E5 / 코드에서 복원 | MSE, 원래 벡터 cosine, 학습 평균 복원 대비 오차 | 압축 후 표현이 얼마나 남는가? 높은 cosine이 단순 공통 성분 때문인지 확인 |
| RQ 모델의 연속 경로 / 코드 복원 | 같은 encoder/decoder에서 양자화만 우회한 진단 | 학습된 경로와 이산 코드 경로의 출력 차이. 독립 AE 실험이나 전체 양자화 인과 효과는 아님 |
| 원래 이웃 / 복원 이웃 | 과거 audit 최대 1,000 query × fit 최대 10,000 reference에서 Top-10 overlap | 원래 E5 근접 구조가 보존되는가? 전체 추천 Recall이나 실제 의미 정답과 다름 |
| 코드 사용·충돌 | 단계별 사용률·perplexity·최대 버킷, prefix 길이별 중복 | 일부 코드에 쏠리는지, 실제 사용량과 분해능이 충분한지 확인. 같은 본문 반복도 충돌에 포함 |
| 저장량 | float32 벡터 vs 코드 + encoder/decoder/codebook/scaling tensor | 공유 모델까지 포함해 압축하는지 확인. ID·파일 포맷·원 벡터를 별도로 보존하는 연구용 디스크 비용은 제외 |
| 추천 검색 | 같은 raw / continuous / decoded 프로필의 Recall@100·NDCG@10 | 코드 복원이 실제 미래 식당 검색에 손실 또는 이득을 주는가? |
| C5 결합 | 같은 C5를 cosine으로 재정렬 또는 dense와 RRF 결합 | 행동 후보 안의 순위와 후보 보완 효과를 구분 |

C5 비교는 **LambdaRank 이전 후보 순서**다. cosine 재정렬을 기존 학습된 LambdaRank의 개선으로 부르지 않는다. 동일 경로 raw와 RQ, C5와 각 경로의 NDCG@10 차이는 동일 사용자 paired bootstrap 2,000회로 계산한다. 이번 validation 결과는 탐색이며 독립 미래 holdout의 최종 증거가 아니다.

## 재현과 artifact

프로젝트 루트에서 실행한다. Python 3.12의 별도 `.venv-rqvae`를 사용했다. 공개 E5 가중치와 CPU 패키지는 프로젝트 내에서 내려받았으며 DB·기존 임베딩 캐시·기존 모델을 변경하지 않는다.

```bash
PYTHONPATH=src .venv-rqvae/bin/python -m rating_recsys.experiments.rqvae_cli --download-model
```

임베딩 캐시가 준비된 다음 재실행할 때는 `--download-model` 없이 실행할 수 있다. 실제 DB 전체에 완료한 모델을 적용한다.

```bash
.venv-rqvae/bin/python scripts/encode_supabase_review_codes.py --run-dir artifacts/comparisons/rqvae/<run>
```

저장된 코드·순위를 원래 임베딩을 읽지 않고 검증한다. 이 명령은 새 학습·예측·DB 조회를 하지 않는다.

```bash
.venv-rqvae/bin/python scripts/verify_review_codes.py --run-dir artifacts/comparisons/rqvae/20261010T065109Z
```

- [실제 수치](../artifacts/comparisons/rqvae/20261010T065109Z/metrics.json), [독립 재검산](../artifacts/comparisons/rqvae/20261010T065109Z/verification.json): 5개 artifact·2개 구현 hash, 리뷰 ID·fit/audit 분리·과거 catalog/방문 제외를 확인. 코드 복원 최대 오차 `1.68e-8`; 저장 순위의 NDCG/Recall@10/100 재계산 일치.
- [전체 DB 적용 근거](../artifacts/comparisons/rqvae/20261010T065109Z/supabase_application.json), [전체 리뷰 코드](../artifacts/comparisons/rqvae/20261010T065109Z/supabase_review_codes.npz), [고정 모델](../artifacts/comparisons/rqvae/20261010T065109Z/model.pt).
- [전체 코드 독립 검증](../artifacts/comparisons/rqvae/20261010T065109Z/supabase_verification.json): ID 94,301개 고유·embedding cache 연결·model/code/cache/구현 hash 일치. 무작위 1,000개는 저장 E5로 재인코딩한 코드가 일치했고, 코드 복원 최대 오차는 `1.49e-8`이다. 새 DB 조회·학습 없이 검산했다.
- [검산 companion notebook](../artifacts/comparisons/rqvae/20261010T065109Z/analysis.ipynb): 표와 hash·집계 확인용 Python 5셀을 순서대로 실행해 출력을 저장했다. Jupyter/nbformat/nbclient가 없어 Jupyter kernel 실행·viewer 검사는 미실행이다. 프로젝트 루트 또는 notebook 폴더에서 표준 Python으로 계산을 재현할 수 있다.
- 핵심 RQ 검증 8개, 기존 데이터·만족도·순위 검증 46개가 통과했고 소형 합성 데이터 학습·코드 복원도 확인했다. 기존 환경 OpenMP 라이브러리 경로 문제는 검증 명령에만 임시 경로를 제공해 해결했다.
- CPU4threads에서 과거 E5 추론 1,069.36초·RQ 학습 57.71초·두 출력 경로 생성 1.24초. 다운로드·초기화는 E5 추론 시간에 제외되며, 각 추천 경로 실행 시간은 순서·캐시 영향을 통제한 속도 비교가 아니다.

- [RQ-VAE 구현](../src/rating_recsys/retrieval/rqvae.py), [학습·비교 CLI](../src/rating_recsys/experiments/rqvae_cli.py), [DB 전체 코드 적용](../scripts/encode_supabase_review_codes.py).
- [핵심 검증](../tests/test_rqvae.py): 잔차 최근접 선택·STE/codebook gradient·코드 복원/저장·동일 입력 분리·동등 리뷰 pooling·결측/방문 제외.
- 새 E5 캐시는 `artifacts/rqvae_review_embeddings`, DB 전체 적용 캐시는 `artifacts/rqvae_all_db_embeddings`다. 원문은 저장하지 않고 ID와 벡터·집계·입력 hash를 유지한다.

기존 [리뷰 텍스트 프로파일](./text_data.md)과 [E5·RLMRec 실험](./review_embeddings.md)의 데이터 단위·입력 정책 차이를 함께 확인한다.
