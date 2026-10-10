# 실험 결과

현재 기준은 **C1 + LightGCN 후보 검색(C5) + LambdaRank**다. E5 Small 캐시는 유지한다. RLMRec/E5 결합까지 비교했지만 최종 정확도의 일관된 개선은 확인하지 못해 기존 모델을 유지한다.

## 완료한 실험

| 주제 | 결과와 현재 판단 | 리포트 |
|---|---|---|
| Baseline | C5 Recall@100 20.1626%; LTR NDCG@10 0.025467. Test 재정렬 이득은 미확인 | [Baseline](./baseline.md) |
| 리뷰 임베딩 | RLMRec-Con/E5를 기존 LightGCN에 추가해 20epoch·가중치 3개·shuffle·전체 LambdaRank 재학습 완료. 최종 NDCG@10 0.027694→0.028022이나 Recall@10 감소·paired 구간 0 포함으로 미채택. 이전 LTR·Two-Tower·Transformer도 개선 미확인 | [E5·Transformer·RLMRec](./review_embeddings.md) |
| 리뷰 검색·순위 결합 | BM25 단독·결합·전처리는 미채택. VDB·E5 하이브리드 설계와 RRF 한계 검토 | [BM25·하이브리드](./bm25.md) |
| 리뷰 속성 추출 | 같은 리뷰와 평점의 표현 일치성 진단 완료. 미래 추천 효과는 미검증 | [리뷰 속성](./review_aspects.md) |

텍스트 표현의 다음 비교는 [임베딩 방법 조사](./review_embeddings.md#텍스트-임베딩-방법-조사)에서 확인한다. TF-IDF·단어 벡터·한국어 SBERT·E5·BGE·Qwen·API encoder, 리뷰 집계·속성 표현·추천 목적 학습을 나눠 정리한 문헌 조사이며 새 추천 실험은 아니다.

[다른 추천 모델과의 결합](./review_embeddings.md#리뷰-텍스트와-다른-추천-모델의-결합)에는 HFT·EFM·NARRE·MPCN·CARP·UniSRec·APH의 원문 결과와 조건, 리뷰·문장·의견별 분할, 근거 추출, 여러 벡터의 실제 입력 구조를 정리했다. C5에 의존하지 않는 설계안이며 프로젝트 성능은 아직 측정하지 않았다.

[두 안의 구현 설계](./review_embeddings.md#두-안의-구체적-구현-설계)는 입력 bank, attention·ID 결합, 학습·검색 순서를 설명한다. 첫 안의 코드·초기 validation은 [실행 결과](./review_embeddings.md#1안-전체-validation-결과), 동일 설정 10epoch의 정확도·표현 변화와 다음 검증은 [원인 조사](./review_embeddings.md#3epoch-판단과-식당-쏠림의-원인-조사)에 정리했다. 둘째 안은 미실행 설계다.

## 다음 실험

아래는 **아직 실행하지 않은 작업**이다. 순서는 [리뷰 임베딩 후속 계획](./review_embeddings.md#진단과-후속)을 따른다.

[RLMRec/E5 비교](./review_embeddings.md#rlmrec-방식으로-베이스라인에-텍스트를-결합한-비교)는 완료했다. 같은 그래프 결합에서 긍정·부정·속성별 프로필을 분리하는 후속은 미실행이며, [행동 후보·점수를 유지한 본문 유무 대조](./review_embeddings.md#베이스라인을-유지한-텍스트-추가효과-검증)도 미실행 설계로 유지한다. 독립 리뷰 Transformer의 학습 분포 개선은 [학습 후보·검색 점수·표현 분리](./review_embeddings.md#다음에-바꿀-것은-무엇인가)를 따른다. 이 학습 분포 변경도 미실행이다. 기준 모델의 신호와 보완 근거는 [baseline 점검](./baseline.md#왜-리뷰-모델보다-강하게-나타나는가)에 있다.

1. Early stopping의 NDCG 정규화 기준을 최종 평가와 맞춘다.
2. Validation에서 리뷰 유사도 피처와 존재 여부·개수 피처의 효과를 분리한다.
3. 이후 리뷰 집계·긍정/부정 표현이나 후보 검색 보완을 한 조건씩 검증한다.

## 문서 역할

리포트는 주제마다 하나만 유지하고 **현재 판단 → 결과 → 한계·다음 실험 → 근거** 순서로 갱신한다. 모델별 도입 검토·CPU 측정·재검증을 별도 리포트로 계속 늘리지 않는다.

[모델·평가 정의](../BASELINE_MODEL.md), [전체 계획](../PLAN.md), [로컬 파일 보존 범위](../artifacts/README.md)는 별도 문서가 담당한다. 실행별 상세 수치·모델·추천 순위는 해당 artifact에 두며, 기존 결과는 보존하고 새 실험은 별도 run에 저장한다.
