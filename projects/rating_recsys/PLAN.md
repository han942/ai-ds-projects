# V2 Plan & Experiment Notes

기준일: 2026-10-04. 현재 모델은 [BASELINE_MODEL.md](./BASELINE_MODEL.md)를 따른다.
이 문서는 다음 실행, 실험 후보, 지난 판단과 운영 방법을 함께 관리한다.

## 다음 실행

| 순서 | 작업 | 상태 / 판단 기준 |
|---|---|---|
| 0 | Baseline 후보·학습 행을 고정 파일로 재사용 | 구현 완료. 기본 자동 사용, 데이터 생성만 별도 실행 가능 |
| 1 | LTR: 방문 하나씩 → 기간 안 여러 방문 → 원래 평점 보존 (P/W/WR) | 옵션 구현 완료, 전체 snapshot 비교 미실행 |
| 2 | 외부 encoder 없이 리뷰+평점 후보 비교 | 설계 단계. 후보 Recall 개선부터 확인 |
| 3 | 개인별 평점 성향·관측 선호 쌍 평가 | 모델 학습 변경과 평가 정의 변경을 분리 |
| 4 | 새 미래 holdout·seed 반복 | 반복 확인한 test의 탐색 결과를 검증 |

실험 실행 방식도 개선 대상으로 둔다. 사용자가 비교할 후보 조합과 학습 예산을 직접
지정하고, validation에서 검토한 조건만 최종 refit/test로 넘길 수 있어야 한다.
아래의 설정 파일·조건 선택·validation 전용 실행은 **설계이며 아직 구현되지 않았다.**
Baseline 데이터의 실행 간 재사용은 구현되어 기본으로 적용된다.

후속 서비스 작업은 추천 API, 후보/모델 버전 추적, 노출·클릭·재방문 로그다.
DB integration test, 날짜 정밀도·수집 완전성 확인, cold-start 평가는 남아 있다.
후보 K=200/300 실험에 앞서 설정 K가 평가 cutoff에 포함되도록 수정해야 한다.
현재는 20/50/100, 5/10 중 설정값 이하만 지표를 만들어 다른 K에서 실행이 실패할 수 있다.

## 새 후보 방법론을 비교하는 범위와 비용

### 비교 조건 수와 모델 학습 횟수는 다르다

현재 후보 baseline은 C5 = RRF(C1 item-item, C4 LightGCN)이고, 전체 추천 baseline은
그 후보를 R1 LambdaRank로 재정렬하는 구성이다. 후보 변경을 검토할 때는 먼저 후보
비교만 진행하고, 개선된 조건에 대해 LTR까지 재학습하는 두 단계로 나눈다.

현재 구성의 역할을 확인하려면 C1 단독 / LightGCN 단독 / C1+LightGCN의 **3조건**이면 된다.
새 그래프 모델 X로 LightGCN을 교체하려면 다음 **3조건**부터 비교한다.

| 조건 | 후보 구성 | 확인할 효과 |
|---|---|---|
| 기준 | C1 + LightGCN | 현재 baseline |
| 새 모델 단독 | X | X 자체의 검색 성능 |
| 교체 | C1 + X | 같은 C1과 결합했을 때 그래프 모델 교체 효과 |

새 모델을 기존 baseline에 추가하려는 목적이면 교체 조건 대신 C1+LightGCN+X를 둔다.
교체와 추가를 모두 확인하면 4조건이다. C1·LightGCN·X의 모든 비어 있지 않은 조합은
7개지만, 첫 비교에 전부 필요하지는 않다. C1 단독 등을 별도 진단 조건으로 더할 수 있다.

동일한 cutoff·설정·seed에서 X의 후보 목록을 한 번 생성하면 X 단독, C1+X,
C1+LightGCN+X에 공유할 수 있다. RRF는 후보 순위 목록의 결합 계산이므로
조합마다 X를 다시 학습하지 않는다. 여러 지표 계산이나 bootstrap 반복도 모델 학습이 아니다.

### 현재 후보 비교 명령이 실제로 하는 일

`rating-recsys-compare <model>`은 다음 순서로 실행하며 **LTR은 학습하지 않는다.**

1. 같은 snapshot과 전역 T1/T2로 validation/test query를 만든다.
2. 저장된 validation baseline 후보를 읽는다. 없는 조건일 때만 고정 LightGCN을
   T1까지 학습하고 C0~C5 참고 후보를 만들어 저장한다.
3. 새 모델의 grid 설정을 각각 T1까지 학습한다. Validation의 새 모델 단독 Recall@100으로
   epoch와 설정을 먼저 선택한다. 이후 선택된 설정의 단독 / C1과 RRF / C1+C4와 RRF
   3가지 중 validation Recall@100이 가장 높은 결합을 고른다.
4. 선택된 새 모델을 T2까지 한 번 refit한다. Test baseline 후보도 저장 파일을 읽고,
   없는 조건일 때만 고정 baseline LightGCN을 T2까지 학습해 생성한다.
   같은 test window에서 후보 목록들을 평가한다.

따라서 현재는 모든 grid 설정 × 모든 결합을 학습하는 구조가 아니다. 설정을 단독 검색
성능으로 먼저 고르기 때문에 **특정 결합의 성능을 기준으로 설정까지 최적화하는 기능은 없다.**
Validation에서 최종 정책을 선택하고, test의 다른 조건 수치는 진단용으로 기록한다.

| 현재 명령의 기본값 | 새 모델 설정 수 | 새 모델 학습 + 최종 refit | 고정 baseline LightGCN | LTR 학습 |
|---|---:|---:|---:|---:|
| `compare lightgcn` | layers 3종 × L2 2종 = 6 | 6 + 1 | 첫 생성 2, 이후 0 | 0 |
| `compare deepconn` | objective/activation 3종 | 3 + 1 | 첫 생성 2, 이후 0 | 0 |
| 후보 설정 1개로 제한 | 1 | 1 + 1 | 첫 생성 2, 이후 0 | 0 |

LightGCN 비교 기본값은 최대 200 epochs, 5 epochs마다 평가, 개선 없는 평가 6회면 종료다.
현재 baseline 내부 LightGCN은 고정 20 epochs이므로 두 실행의 학습 예산이 다르다.
표의 횟수는 학습 호출 수이며 epoch별 평가 로그를 별도 실험으로 세지 않는다.
결과 표는 baseline·참고 6조건과 새 모델 관련 3조건을 함께 보여주므로 총 9행이 생긴다.

`--baseline-run`은 저장된 R1 추천 결과를 읽어 비교하는 옵션이다. 후보·학습 행의
재사용은 이 옵션과 별개로 `prepared/`에서 수행한다. 이 비교는 X의 후보를 새 LTR로
재정렬한 최종 성능이 아니다.
현재 등록된 후보 모델은 `lightgcn`, `deepconn`이다. 다른 GCN이나 새 방법론 X는
모델 구현과 공통 후보 모델 adapter 등록이 필요하다.

### 전체 파이프라인이 오래 걸리는 실제 이유

2026-09-30 표준 baseline run의 manifest에 기록된 총 시간은 약 23.6분이다.

| 단계 | 기록된 시간 |
|---|---:|
| Train의 과거 시점별 후보·feature·학습 행 생성 | 16.4분 |
| Validation LightGCN 학습 + 후보·feature 생성 | 1.0분 |
| LambdaRank 7개 설정 탐색 | 41초 |
| 추가 학습 행 생성 + 최종 LambdaRank refit | 4.1분 |
| Test LightGCN·후보 생성과 평가 | 1.3분 |

각 학습 방문의 당시 이력으로 후보를 만드는 과정이 가장 오래 걸린다. 과거 graph도
3개월 구간별로 다시 학습한다. 해당 run의 학습 graph 기록에는 41개 구간이 있으며
첫 구간은 데이터 부족으로 skip했다. 이는 41개 하이퍼파라미터를 탐색한 것이 아니다.
미래 방문 정보를 과거 후보 생성에 사용하지 않기 위한 시간 구간 구분이다.

전체 파이프라인은 LambdaRank 7개 설정 학습 + 선택된 설정의 최종 refit 1회를 수행한다.
이 시간은 저장된 학습 행을 사용하기 전 기록이다. 2026-10-04부터 같은 조건의 학습 행과
validation/test 후보를 `prepared/`에서 읽으므로, 파일이 있으면 과거 LightGCN 학습과
후보·feature 생성 단계를 생략한다. Graph checkpoint 자체는 최근 모델을 메모리에만
보관하지만, 이미 완성된 학습 행을 읽을 때는 checkpoint 모델을 다시 만들 필요가 없다.
2026-10-02 shrinkage 비교는 실행 내부 공유만 했고 약 20.6분이었다. 이제 이 runner도
표준 baseline과 같은 저장 데이터를 사용한다.

### 지금 직접 바꿀 수 있는 설정

후보 모델의 grid·최대 epochs·평가 간격·patience·seed는 CLI로 지정할 수 있다.
예를 들어 아래는 **LightGCN 후보 비교의 설정을 1개로 제한**한다. 기존 C5 내부의
고정 LightGCN과 비교하는 명령이며 새 GCN 구현을 실행하는 예시는 아니다.

```bash
rating-recsys-compare lightgcn \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --layers-grid 3 --regularization-grid 0.0001 \
  --max-epochs 20 --eval-every 5 --patience 3
```

현재 이 명령도 자동 3가지 결합과 최종 refit/test를 실행한다. Validation 전용 옵션은 없다.
전체 파이프라인에서 `--num-leaves-grid 15 --min-child-samples-grid 10`을 지정하면
ranker grid 1개 **외에 고정 150-tree 기준 설정도 남아 총 2개**를 비교한다.
Shrinkage 전용 CLI는 snapshot·보정 강도·출력 폴더·cache 사용 옵션을 받는다.
튜닝 grid를 바꾸는 옵션은 아직 없다.

### Baseline 데이터를 한 번 저장하고 반복 사용한다

원본 interaction snapshot과 모델 학습 데이터는 서로 다른 파일이다. 원본은
`artifacts/snapshots/`에 이미 고정되어 있고, 여기서 계산한 학습 행과 후보 목록은
`artifacts/prepared/<snapshot-id>/`에 저장한다. 크롤링·DB 적재를 다시 실행하지 않는다.

| 저장 구간 | 내용 |
|---|---|
| `train-<key>/` | T1까지 LTR feature 행·label·group 크기·과거 평점 prior |
| `refit-<key>/` | Prefix에서는 T1 이후 T2까지 추가 학습 행, window에서는 T2까지 전체 학습 행 |
| `validation-<key>/` | T1 시점의 C0~C5 후보 순위·feature·공통 평가 label |
| `test-<key>/` | T2 시점의 C0~C5 후보 순위·feature·공통 평가 label |

각 폴더는 `data.npz`와 `manifest.json`을 가진다. NPZ는 압축된 숫자 배열이라 CSV보다
작고 float32 feature·정수 label·group의 자료형과 순서를 보존한다. JSON에는 후보 목록,
학습 행 수, graph의 과거 데이터 범위, 생성 조건과 파일 checksum을 기록한다.
미래 평가 label과 과거 feature는 저장해도 경로를 구분하며, test label을 모델 선택이나
학습에 사용하지 않는다. 파일로 고정했다고 전체 기간으로 후보를 만들지는 않는다.

표준 실험, 새 후보 비교, shrinkage 비교는 기본으로 자동 재사용한다. 저장된 데이터가
없으면 해당 구간만 한 번 생성한다. 실행 manifest의 `prepared_data`에는 읽은 파일의
key·경로와 `hit`(재사용)/`built`(생성)를 기록한다.
Graph의 `fit_seconds`는 해당 파일을 생성할 때의 기록이다. 현재 실행에 걸린 시간은
run manifest의 `timings_seconds`를 보고, 실제 재생성 여부는 `prepared_data`로 확인한다.

2026-10-04 실제 snapshot `e7896add5b4b5939`의 baseline 자료를 생성하고 재사용을 확인했다.

| 확인 항목 | 결과 |
|---|---:|
| 최초 전체 데이터 준비 | 1,277.010초, 약 21.3분 |
| 같은 조건의 재실행 | 7.155초, 4구간 모두 `hit` |
| Train 학습 데이터 | 14,018 groups / 1,383,764 rows |
| 추가 refit 데이터 | 1,911 groups / 190,888 rows |
| Validation / test 후보 query | 1,645 / 1,580 |
| 저장 크기 | 약 77MB |

위 시간은 **데이터 준비·읽기만** 측정한 값이며 LambdaRank 학습을 포함한 전체 실험 시간이
아니다. Train/refit의 summary와 validation/test의 모든 C5 후보 목록이 2026-10-02
baseline과 일치했다. 평가 label과 feature 행의 식당 순서도 확인했다.
원본 측정은 `prepared/<snapshot-id>/preparation_initial.json`, `preparation_reuse.json`,
`verification.json`에 남겼다. 새 성능 비교 run이나 MLflow run은 생성하지 않았다.

LTR 학습 없이 미리 생성하려면 아래 명령을 실행한다. 현재 컴퓨터에서는 준비·실험·비교
명령도 사용자 PATH에 등록되어 환경 활성화 없이 실행할 수 있다.
이 명령은 후보용 LightGCN과 학습 행만 만들며 LambdaRank 학습·성능 평가는 하지 않는다.

```bash
rating-recsys-prepare --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl
# 후보 비교만 할 때: train/refit의 LTR 학습 행 생성은 생략
rating-recsys-prepare --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl --scope candidates
```

| 바뀐 조건 | 재사용 범위 |
|---|---|
| LambdaRank 트리 수·leaves·learning rate·bootstrap 횟수 | 기존 학습 행과 평가 후보 모두 재사용 |
| 새 후보 모델 X의 구조·파라미터·리뷰 입력 | 고정 baseline 자료는 재사용, X의 학습·후보만 별도 계산 |
| Prefix/window 또는 relevance/rating 학습 label | 해당 train/refit 자료 새 생성, baseline 평가 후보 재사용 |
| Baseline LightGCN 설정·seed·RRF·후보 수·feature 정의·평균 보정 | 영향을 받는 baseline 자료 새 생성 |
| Snapshot·전역 cutoff·평가 정답 기준·전처리 코드/NumPy/SciPy 버전 | 조건에 맞는 자료 새 생성 |

현재 key는 보수적으로 구성한다. 예를 들어 전체 snapshot이 바뀌면 과거 구간의 내용이
같더라도 별도 key를 쓴다. 설명 문서나 LambdaRank 튜닝 코드만 바뀌면 전처리 key는 유지한다.
새 feature가 추가됐다고 이전 행을 새 모델의 학습 데이터로 그대로 쓰지는 않는다.
Baseline 자료를 보존하면서 변경 조건의 데이터를 별도로 생성한다.

필요하면 기존 실험 명령에 `--no-cache`(사용·저장 생략) 또는 `--rebuild-cache`(같은
조건도 다시 생성)를 지정한다. 두 옵션은 함께 사용할 수 없다. 손상되거나 checksum이
맞지 않는 파일은 해당 구간을 다시 만든다. 파일 저장 중 다른 실행이 불완전한 데이터를
읽지 않도록 구간별 lock과 완료 manifest를 사용한다.

### 사용자가 선택할 수 있도록 추가할 실행 설정

하나의 설정 파일에서 다음 항목을 독립적으로 지정하는 구조를 목표로 한다.

| 선택 항목 | 예시 | 현재 상태 |
|---|---|---|
| 비교 조건 | baseline / X 단독 / C1+X만 선택 | 결합 목록이 코드에 고정 |
| 평가 단계 | 후보만 / LTR까지 | 현재 서로 다른 runner, 임의 X의 LTR 연결 미구현 |
| 학습 예산 | 고정 설정 1개 / 지정 grid 탐색 | 모델별 CLI grid 지원, 통합 설정 파일 없음 |
| 종료 단계 | validation에서 종료 / 선택된 조건의 최종 test | validation 전용 실행 없음 |
| 재사용 | 동일 조건의 baseline 후보·학습 행 | `prepared/` 자동 재사용 구현. Graph 가중치 자체의 영구 저장은 아님 |
| 실행 전 확인 | 조건 수·학습 호출 수·epoch 상한 출력 | 공통 dry-run 없음 |

후보 검토에서는 snapshot·cutoff·평가 사용자·정답 정의·후보 수·방문 제외 규칙을 맞춘다.
검색된 정답 수뿐 아니라 기존 후보와의 정답 겹침, coverage, 실행 시간도 본다.
같은 실행 예산으로 비교할지 각 모델을 별도로 튜닝할지는 설정과 보고서에 명시한다.
최종 추천 비교로 넘어가면 후보가 바뀐 조건별로 LTR 학습 데이터와 score feature를
구성해 다시 학습한다. 기존 LightGCN 전용 feature를 X에 그대로 대응한다고 가정하지 않는다.

Cache는 snapshot·cutoff·모델 설정·seed·코드/feature 버전이 일치할 때만 재사용한다.
초기 탐색은 validation에서 진행하고, 조건을 정한 뒤 최종 test를 실행한다.
이 절차는 불필요한 전체 재학습과 test를 보며 조건을 고르는 문제를 줄이기 위한 설계다.
새 후보 모델을 추가하거나 모델 선택 방식을 바꾸는 실험은 이 작업에서 실행하지 않았다.

## LTR 실험: 여러 관측 식당의 순서 학습

### 먼저 무엇을 바꾸려는가

현재 추천 과정은 두 단계다. 후보 생성은 사용자가 갈 만한 식당 100개를 찾고,
LTR(Learning to Rank, 순위 학습)은 그 100개를 어떤 순서로 보여줄지 학습한다.
이번에 바꾸려는 것은 **LTR이 학습할 때 한 사용자의 정답을 묶는 방식**이다.
현재 평가에서는 이미 미래 기간에 방문한 여러 식당을 정답으로 사용한다.

학습 group은 **한 사용자·한 추천 시점의 후보 식당 목록과 각 후보의 정답 label을
묶은 것**이다. Ranker는 이 묶음 안에서 어떤 식당을 더 위에 놓을지 학습한다.
후보가 100개라면 그 group에는 식당별 feature·label이 들어 있는 행이 100개 있다.
여러 사용자의 식당을 서로 비교하는 것이 아니라, 같은 사용자의 후보끼리 비교한다.

이 문서에서 쓰는 P/W/WR은 비교 조건의 이름이다. 서로 다른 후보 모델 이름이 아니다.

| 표기 | 풀어 쓴 의미 | 바꾸는 부분 |
|---|---|---|
| P: 현재 방식 | **Prefix** — 방문할 때마다 이전 이력으로 다음 방문 하나를 학습 | 현재 baseline |
| W: 기간 단위 학습 | **Window** — 한 기간 안의 여러 방문을 같은 group의 정답으로 학습 | 정답을 묶는 방식 |
| WR: 기간 단위 + 평점 보존 | **Window + Rating** — W와 같은 기간 구성을 쓰되 원래 평점 차이를 유지 | W의 학습 label/gain |

아래 예시는 학습 데이터가 어떻게 달라지는지 보여주기 위한 가상 사례다.
실제 실행에서는 C5가 검색한 식당만 학습 후보에 들어간다.

### 같은 사용자의 방문으로 보는 세 방식

사용자 민수는 10월 1일 이전에 A·B 식당을 방문했다. 이후 10~12월에는 다음처럼
평가했다고 가정한다. C·D·E·F는 10월 1일 이전 다른 사용자 데이터에도 존재하는 식당이다.

| 식당 | 민수의 방문 | 민수의 평점 |
|---|---|---:|
| C | 10월 방문 | 3.5 |
| D | 11월 방문 | 4.5 |
| E | 12월 방문 | 5.0 |
| F | 이 기간에 방문한 기록 없음 | 알 수 없음 |

**P: 다음 방문 하나씩 학습한다.**

민수의 이력이 방문할 때마다 늘어난다. C 방문 직전에는 A·B를 보고 C를 정답으로,
D 방문 직전에는 A·B·C를 보고 D를 정답으로, E 방문 직전에는 A·B·C·D를 보고 E를
정답으로 둔다. 각 추천 시점마다 C5 후보를 새로 만든다.

| 학습 시점 | 입력 이력 | 그 group의 정답 | 다른 미방문 후보의 label |
|---|---|---|---|
| C 방문 직전 | A·B | C: label 1 (3.5점) | 0 |
| D 방문 직전 | A·B·C | D: label 2 (4.5점) | 0 |
| E 방문 직전 | A·B·C·D | E: label 2 (5점) | 0 |

C 방문 직전 후보에 D·E가 있더라도 그 group에서는 label 0이다. 이후에 실제로
방문하더라도 그 시점의 정답은 다음 방문 C 하나이기 때문이다. 다음 group에서는
이미 방문한 C가 후보에서 제외된다. 따라서 **D와 E를 같은 group의 관측 정답으로
놓고 4.5점보다 5점을 더 선호했는지 직접 비교하는 학습은 생기지 않는다.**

**W: 기간 안의 여러 방문을 한 번에 학습한다.**

10월 1일에 민수의 이력을 A·B로 고정하고, 그날의 정보로 C5 후보를 만든다.
10~12월 방문 C·D·E가 모두 그 후보에 포함됐다고 가정하면, 같은 group에 다음처럼
label을 붙인다. 11월 방문을 알게 됐다고 10월 1일의 입력 이력을 갱신하지 않는다.

| 후보 | 이후 관측 평점 | W 학습 label | 학습 gain |
|---|---:|---:|---:|
| C | 3.5 | 1 | 1 |
| D | 4.5 | 2 | 3 |
| E | 5.0 | 2 | 3 |
| F | 미관측 | 0 | 0 |

Label은 정답을 코드로 표시한 값이고, gain은 LambdaRank가 좋은 식당에 부여하는
가치다. W는 현재 baseline과 같은 기준, 즉 3점 미만=0, 3점 이상 4점 미만=1,
4점 이상=2와 gain 0/1/3을 쓴다. 위 group에서는 D·E를 C보다 높게 놓는 비교를
학습할 수 있다. 다만 **4.5점과 5점은 같은 label이므로 두 평점의 차이는 사라진다.**

여러 식당에 정답을 붙일 수 있는 구조라고 항상 여러 정답으로 학습되는 것은 아니다.
예를 들어 C5에서 E만 검색됐다면 이 group의 관측 positive는 여전히 하나다.
그래서 검색된 positive가 여러 개인 group 수와 실제 비교할 수 있는 평점 쌍 수를
함께 기록한다. 정답 식당이 후보 밖에 있으면 학습 목록에 끼워 넣지 않는다.

**WR: 같은 기간 학습에서 원래 평점 차이를 보존한다.**

입력 이력과 기간은 W와 같다. 원래 평점을 세 등급으로 묶지 않고 학습 gain으로 쓴다.
LightGBM의 정수 label 요구에 맞춰 평점 × 2를 label로 전달하고, 그 label에 대응하는
gain을 원래 평점으로 지정한다. Label 9는 9점 평가라는 뜻이 아니다.

| 후보 | 이후 관측 평점 | WR 학습 label (평점 × 2) | 학습 gain (원래 평점) |
|---|---:|---:|---:|
| C | 3.5 | 7 | 3.5 |
| D | 4.5 | 9 | 4.5 |
| E | 5.0 | 10 | 5.0 |
| F | 미관측 | 0 | 0 |

같은 group에서 E > D > C라는 선호 차이를 학습에 반영할 수 있다. 이는 순위를
학습하는 모델이며, 원래 평점을 gain으로 쓴다고 모델 출력이 예측 별점이 되는 것은
아니다. 개인의 평점 성향을 정규화한 방식도 아니다. 현재 구현은 반점 단위의 1~5점만
허용하고, 예를 들어 4.1점을 임의로 반올림하지 않는다.

WR에는 남는 가정이 있다. 관측 1점 식당도 gain 1이고, 미관측 식당은 gain 0이다.
따라서 **싫어했던 식당도 방문 기록이 없는 식당보다 위에 두도록 학습할 수 있다.**
미관측 0은 실제 0점 평가라는 뜻이 아니다. W에서는 3점 미만이 label 0이므로,
WR은 평점 해상도뿐 아니라 저평점 방문의 학습상 취급도 바꾼다. 학습에 사용할 수 있는
group 수도 달라질 수 있어 함께 보고한다. 추후 실제 관측 평점이 다른 식당 쌍만
비교하는 학습 방식과 미관측 식당의 취급을 분리해 확인할 수 있다.

### 미래 평점은 정답으로만 사용한다

W/WR의 기본 학습 기간은 3개월이다. 예를 들어 10~12월 group이라면 사용자 이력,
식당 평균·인기도, item-item 관계와 LightGCN을 10월 1일 이전 데이터로 만든다.
기간 안에 발생한 방문·평점은 label에만 사용한다. C의 10월 평점을 10월 1일의
사용자 평균이나 후보 식당 평균에 먼저 넣으면 정답 정보를 입력으로 보는 누수가 된다.

Train의 마지막 기간은 T1에서, 최종 학습의 마지막 기간은 T2에서 자른다.
최종 refit은 T2까지 기간을 다시 구성한다. T1에서 잘랐던 group과 같은 기간의
확장된 group을 모두 붙여 동일 방문을 두 번 학습하지 않는다. 현재 validation/test
기간은 약 4개월이므로 3개월 학습 기간과의 길이 차이를 실험 조건에 명시한다.

### 무엇을 비교하고, 무엇을 동일하게 유지하는가

P → W는 **단일 방문 group에서 기간 안 여러 방문 group으로 바꾼 효과**를 확인한다.
W → WR은 **같은 기간 구성에서 평점을 세 등급으로 묶지 않은 효과**를 확인한다.
W → WR에는 위에서 설명한 저평점 방문의 취급 변화도 포함된다.

Snapshot·전역 T1/T2·C5 후보 생성 규칙·feature 정의·seed·ranker 설정 탐색 범위를
동일하게 유지한다. P와 W는 학습 시점·이력이 다르므로 각 학습 query의 후보 목록이
같다고 가정하지 않는다. W/WR은 기간별 검색 조건이 같지만, label에 따른 학습 group
선별 결과는 달라질 수 있다. **Validation/test에서는 같은 시작 cutoff와 사용자 이력으로
같은 C5 후보를 만들어 평가한다.** 따라서 세 조건의 평가 후보 Recall은 같아야 한다.

최종 평가는 세 조건 모두 기존 relevance 0/1/2와 gain 0/1/3을 사용한다. WR에서
학습 gain을 바꿨다는 이유로 평가 지표까지 바꾸면 같은 척도의 비교가 되지 않기 때문이다.
후보 밖 정답도 포함한 전체 기간의 NDCG·Recall·Precision·MAP·MRR 등을 비교한다.

Early stopping은 검색된 positive가 있는 validation group의 후보 내부 NDCG를 쓴다.
이는 후보 밖 정답까지 포함하는 최종 성능 수치와 구분한다. Validation 전체 지표로
설정과 트리 수를 선택하고 T2까지 refit한 뒤 test를 평가한다. 동점은 모든 조건에서
C5 순위를 유지한다. 이미 확인한 test의 비교는 탐색 결과이며, 최종 일반화 확인에는
새로운 미래 holdout과 seed 반복이 필요하다.

### 구현 상태와 실행

P는 현재 baseline이다. W/WR의 query·label 옵션은 구현했고 작은 데이터 검증은
통과했지만, 전체 snapshot으로 P/W/WR을 비교한 성능 결과는 아직 없다.
2026-10-02에 실행한 것은 P 방식에서 평균 feature만 보정한 **shrinkage 비교**다.
그 결과를 W/WR의 성능으로 해석하지 않는다.

```bash
# P: 현재 방식 — 다음 방문 하나씩 학습
rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --ranker-training-mode prefix --ranker-label-mode relevance --no-mlflow --label prefix-control

# W: 기간 안 여러 방문을 함께 학습, 기존 세 등급 유지
rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --ranker-training-mode window --ranker-label-mode relevance --no-mlflow --label window-control

# WR: W의 기간 구성 + 원래 평점의 선형 gain
rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --ranker-training-mode window --ranker-label-mode rating --no-mlflow --label window-rating
```

이 명령들은 다음 비교용이다. 이번 문서 수정 과정에서 모델 실험을 추가 실행하지 않았다.

## 리뷰+평점 후보 설계

리뷰 정보는 우선 Stage 1 후보에 별도 feature로 넣고 평점 신호와 결합한다.
Sentence Transformers와 외부 사전학습 encoder 사용은 사용자 요청으로 보류한다.
Train 리뷰로만 학습한 글자 n-gram TF-IDF 또는 자체 학습 표현을 검토한다.
TF-IDF는 표현 겹침을 측정하며 문장 의미를 이해한다고 해석하지 않는다.

원래 평점을 학습하는 MF 관계 `μ + b_u + b_i + p_u·q_i`에 텍스트 항을 추가한다.
이 기본 관계는 V1 MF에도 있었고, 현재 V2 LightGCN은 방문 기반 사용자·식당 내적을
학습한다. V2 baseline에는 명시적인 평점 회귀 bias가 없다. V1 원본의 평균은 식당 평균의
평균, 수정본은 관측 평점 전체의 평균이므로 동일한 추정량으로 취급하지 않는다.
사용자별 상수 bias를 모든 후보에서 빼기만 하면 순위는 바뀌지 않는다.

| 비교 후보 | 확인할 효과 |
|---|---|
| 원래 평점을 학습한 MF | 방문 기반 C5 대비 평점 정보 |
| MF + regularized 사용자·식당 bias | 개인별 평가 성향 보정 |
| 텍스트 프로필 단독 | 텍스트 정보 자체의 참고 성능 |
| MF + 텍스트 / bias MF + 텍스트 | 평점·보정과 텍스트의 결합 |

좋아한/싫어한 과거 식당의 텍스트 프로필을 구분한다. 본인 리뷰를 쓰는 방식과 방문
식당의 다른 사용자 리뷰를 쓰는 방식을 비교한다. 문체를 취향으로 오인하는 영향은
확인할 가설이며 사실로 단정하지 않는다. 이력이 적거나 구분이 안 되면 공통 프로필로
fallback한다. 전체 평균 shrinkage도 개인의 후한 평가와 좋은 식당만 방문한 효과를
완전히 구분하지 못한다.

어휘·IDF·문서·프로필은 각 cutoff 전 리뷰만 사용한다. Target의 리뷰·평점·맛/가격/서비스는
label 외 입력에서 제외한다. 후보 100개·방문 제외·같은 학습 예산으로 비교하고,
점수 척도가 다른 평점과 유사도를 그대로 더하지 않는다. 결합 가중치는 validation에서
선택하거나 source별 순위를 RRF로 결합한다. 후보를 확정한 다음 LTR을 재학습한다.

판단 지표는 공통 후보 Recall@20/50/100, 전체 방문 Recall, 관측 평점 RMSE/MAE,
관측 평점 쌍의 순서 일치도다. 사용자 이력 수·평점 구간·식당 리뷰 수별 성능과
coverage·노출 쏠림·실행 비용도 확인한다. 관측되지 않은 식당의 실제 만족도를 측정한
것으로 해석하지 않는다. RMSE 개선이 후보 Recall 개선을 보장하지 않는다.

## 지난 실험과 판단

| 시점 | 실험 | 결과 / 결정 |
|---|---|---|
| 2026-09-28 | LightGCN + item-item 후보 | 같은 global split에서 Recall@100 20.16%, 당시 C3 16.66%보다 +3.50%p → C5 채택 |
| 2026-09-30 | DeepCoNN 리뷰 CNN/FM 후보 | 단독 8.69%, C5와 결합 18.44% < C5 20.16% → 채택 안 함 |
| 2026-09-30 | C5 + LambdaRank | 동점 ID 정렬 당시 NDCG@10 R0/R1 약 0.0276. 후보 밖 정답 삽입 run은 대체 |
| 2026-10-01 | Ranker 동점 규칙 | 점수가 같으면 C5 순위 유지 |
| 2026-10-02 | 평균 feature shrinkage λ=10 | 현재 코드 baseline NDCG@10 0.029360 → 0.028207 → 기본 λ=0 유지 |

[최신 shrinkage 원본](./artifacts/comparisons/shrinkage/20261002T062004863532Z-e7896add/report.md):
validation NDCG@10은 0.025151 → 0.027881였지만 test 개선이 이어지지 않았다.
Test NDCG 차이 −0.001153의 사용자별 paired bootstrap 2,000회 95% CI는
[−0.005006, +0.002692]다. λ·seed 각 1개이며 모든 shrinkage 방식이 나쁘다는 뜻은 아니다.
사용자·식당 평균 feature만 바꿨고 후보·학습 group·label은 공유했다. λ는 평가 전 고정했다.
Baseline은 1 tree, shrinkage는 96 trees를 같은 validation 규칙으로 선택했다.
이는 원래 평점 label의 사용자별 정규화나 MF bias 학습 실험이 아니다.

2026-09-27 이전 leave-last-two-out 결과는 현재 global split과 데이터/모수가 달라
직접 비교하지 않는다. 초기 후보 K·지역 ablation·LightGCN 시도와 원본 보고서는
`legacy/v2_experiments.zip`에 보관했다. DeepCoNN 결과는 텍스트 전체의 무효성을
뜻하지 않는다. 당시 CNN/FM의 제한과 현재 리뷰+평점 설계를 구분한다.

## 실행 방법

### Dashboard·MLflow: 환경 활성화 없이 실행

2026-10-04 현재 컴퓨터의 `~/.local/bin/`에 다음 명령을 등록했다.
이 폴더는 기존 `.zshrc`의 PATH에 이미 포함되어 있다.

| 명령 | 연결된 실행 파일 |
|---|---|
| `rating-recsys-dashboard` | 프로젝트 `.venv/bin/rating-recsys-dashboard` |
| `mlflow` | 프로젝트 `.venv/bin/mlflow` |
| `rating-recsys-prepare` | 프로젝트 `.venv/bin/rating-recsys-prepare` |
| `rating-recsys-experiment` | 프로젝트 `.venv/bin/rating-recsys-experiment` |
| `rating-recsys-compare` | 프로젝트 `.venv/bin/rating-recsys-compare` |

명령의 연결은 프로젝트 환경의 실행 파일을 가리키며, 그 실행 파일은 프로젝트의
Python을 지정하고 있다. 따라서 **환경을 활성화하지 않아도 설치된 프로젝트 환경으로
실행된다.** 가상환경 자체는 유지해야 하지만, 실행할 때마다 `conda activate`나
`conda deactivate`를 할 필요는 없다. `(base)` 상태에서도 사용할 수 있다.

기존 터미널에서는 `rehash`를 한 번 실행한다. 아래 명령은 각각 별도 터미널에서 실행한다.
Dashboard는 현재 디렉터리와 관계없이 프로젝트의 artifacts를 기본으로 읽는다.
MLflow도 같은 DB를 읽도록 여기서는 DB 절대 경로를 지정한다.

```bash
rating-recsys-dashboard --address 127.0.0.1
mlflow ui --backend-store-uri sqlite:////home/hananthony1/codes/ai-ds-projects/projects/rating_recsys/artifacts/mlflow.db --host 127.0.0.1 --port 5000
```

이 등록은 현재 컴퓨터의 설정이다. 다른 컴퓨터에서는 프로젝트 환경을 만든 뒤
그 컴퓨터의 프로젝트 경로로 한 번 등록한다. 현재 연결을 다시 만드는 명령은 아래와 같다.
기존 연결이 있으면 재실행할 필요가 없다.

```bash
mkdir -p "$HOME/.local/bin"
ln -s /home/hananthony1/codes/ai-ds-projects/projects/rating_recsys/.venv/bin/rating-recsys-dashboard "$HOME/.local/bin/rating-recsys-dashboard"
ln -s /home/hananthony1/codes/ai-ds-projects/projects/rating_recsys/.venv/bin/mlflow "$HOME/.local/bin/mlflow"
ln -s /home/hananthony1/codes/ai-ds-projects/projects/rating_recsys/.venv/bin/rating-recsys-prepare "$HOME/.local/bin/rating-recsys-prepare"
ln -s /home/hananthony1/codes/ai-ds-projects/projects/rating_recsys/.venv/bin/rating-recsys-experiment "$HOME/.local/bin/rating-recsys-experiment"
ln -s /home/hananthony1/codes/ai-ds-projects/projects/rating_recsys/.venv/bin/rating-recsys-compare "$HOME/.local/bin/rating-recsys-compare"
rehash
```

`command -v rating-recsys-dashboard mlflow rating-recsys-prepare rating-recsys-experiment rating-recsys-compare`로
어떤 명령이 선택되는지 확인할 수 있다.
다른 활성 환경에 같은 이름의 명령이 있으면 PATH 순서에 따라 그 명령이 먼저 선택될 수 있다.

### 개발·학습용 환경 활성화

프로젝트 루트 `projects/rating_recsys`에서 실행한다. 현재 프로젝트의 `.venv`에는
필요한 패키지가 설치되어 있어 재설치할 필요가 없다. 프로젝트의 `python`, `pip`와
나머지 학습 명령을 함께 쓰려면 기존 환경을 다음과 같이 활성화한다.

```bash
cd /home/hananthony1/codes/ai-ds-projects/projects/rating_recsys
conda activate /home/hananthony1/codes/ai-ds-projects/projects/rating_recsys/.venv
rehash
command -v python rating-recsys-dashboard mlflow
```

저장소 루트 `ai-ds-projects/.venv`와 프로젝트 내부 `rating_recsys/.venv`는 다른
환경이다. 2026-10-04 확인 결과 루트 환경에는 예전 적재용 명령만 등록되어 있고
Dashboard·MLflow 명령은 없다. 터미널의 `(.venv)` 표시만으로는 어느 환경인지
구분할 수 없다. 루트 Python venv를 활성화한 상태라면 먼저 `deactivate`하고 위의
Conda 환경을 활성화한다. 프로젝트 환경은 Conda prefix이므로
`source .venv/bin/activate` 대신 `conda activate`를 사용한다.

`command -v`의 세 경로가 모두 `rating_recsys/.venv/bin/` 아래인지 확인한다.
새 환경을 만들 때만 아래 설치를 사용한다.

```bash
conda create --prefix ./.venv python=3.10 pip libgomp -y
conda activate ./.venv
pip install -e '.[experiment,dev]'
cp .env.example .env
```

DB를 읽을 때 `.env`의 `DATABASE_URL`을 설정한다. 적재에는 고정 `USER_HASH_SALT`도
필요하다. Snapshot만 사용하는 실험에는 DB 연결이 필요 없다.

```bash
rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 python -m rating_recsys.experiments.shrinkage --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl --strength 10
rating-recsys-compare lightgcn --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl
rating-recsys-compare deepconn --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl
```

DeepCoNN만 `pip install -e '.[deepconn]' --extra-index-url https://download.pytorch.org/whl/cpu`가
필요하다. 리뷰 본문은 snapshot 옆 `.reviews.jsonl`로 읽으며 없으면 DB에서 만든다.
전체 옵션은 `rating-recsys-experiment --help`, `rating-recsys-compare <model> --help`로 본다.
기존 baseline 실행 약 24분, 이번 공유 shrinkage 비교 약 20.6분이었다.

Dashboard/MLflow는 `runs/`의 표준 실행용이다. Shrinkage 비교는 별도 report/metrics로 본다.

## 데이터·크롤링

모델 입력은 DB 또는 그 고정 snapshot이다. 수집 CSV는 적재 단계에서만 읽는다.
DB 검증 SQL은 `queries/`에 있으며 migration·적재 명령은 아래와 같다.

```bash
rating-recsys-migrate
rating-recsys-ingest --dry-run
rating-recsys-ingest --skip-migrations
python -m rating_recsys.ingestion.import_crawler
rating-recsys-dataset
```

전국 import는 `crawler/data/`의 최신 전국 `.csv.partial`을 고정 복사해 적재하고
`artifacts/snapshots/ingestion_sources/`에 CSV와 `.import.json`을 남긴다. 완료 `.csv`는
자동 선택하지 않는다. 사용자명은 고정 salt로 pseudonym화하고 content hash로 중복을
제외한다. 표시 이름이 같으면 같은 사용자로 매핑하는 현재 한계가 있다.

2026-09-27 기록: 원본 리뷰 96,922개(기존 23,207 / 전국 73,715), 최초 interaction
88,554개. 중간 전국 CSV 82,966행 중 필수값 누락 77건, 확실한 중복 9,174건이었다.
전국 crawl은 미완료이며 날짜 표시 변경에 따른 중복·본문 누락·날짜 precision 차이가
남는다. 고정 CSV와 import JSON은 유지하고 상세 품질 기록은 보관 ZIP에서 확인한다.
기존 import JSON의 `source_snapshot`은 적재 당시 경로다. 현재 CSV는
`artifacts/snapshots/ingestion_sources/`에서 같은 파일명으로 찾으며 파일 hash도 유지한다.

```bash
python -m pip install -r crawler/requirements.txt
python -m playwright install chromium
python crawler/diningcode_playwright.py --national-regions --max-restaurants 0
python crawler/diningcode_playwright.py --national-regions --max-restaurants 0 --resume
```

필요한 WSL 시스템 라이브러리는 `sudo .venv/bin/python -m playwright install-deps chromium`으로
설치한다. 기본은 창이 표시되며 `--headless`로 숨긴다. 중단 시 `.csv.partial`과
`.checkpoint.json`을 남긴다. `--profile-url`로 단일 식당, `--skip-incomplete`로 미완료
식당을 미루고 대기 식당을 처리할 수 있다. 이 옵션에서 `crawl_complete=true`도 모든
식당의 리뷰가 수집됐다는 뜻은 아니며 `skip_incomplete_urls`와 식당 상태를 함께 본다.
대표 음식 목록 수집은 사이트 전체 식당의 전수 수집이 아니다. 접근 제한 시 중단한다.

## 파일 관리

직접 관리하는 V2 문서는 README·BASELINE_MODEL·PLAN 세 개다. 자동 생성된 run 보고서는
각 결과와 함께 둔다. V1의 안내는 [내부 README](./legacy/v1_rating_prediction/README.md)
하나로 통합했다. 바깥 안내문·언어별 README·archive README의 내용은 이 문서로 옮겼고,
V1 notebook·데이터·모델·결과·개발 기록은 기존 구조를 유지한다.

```text
artifacts/
├── snapshots/             interaction·리뷰 본문·ingestion_sources 재현 입력
├── prepared/              snapshot·생성 조건별 고정 학습 행과 baseline 후보
├── runs/                  표준 baseline 실행 (현재 2026-09-30 run 보존)
├── comparisons/           비교 실행 (현재 2026-10-02 shrinkage 보존)
└── mlflow.db              로컬 지표 저장소
```

실행 결과는 폴더당 `report.md`, `manifest.json`, `metrics.json`, 모델·추천 파일이다.
새 모델 비교도 `comparisons/<model>/<run_id>/`로 생성한다. Git에는 결과 report와
후보 비교 learning_curve만 올라가며 snapshot·모델·리뷰 본문은 로컬에만 둔다.

오래된 V2 결과·분석·로그·정리 전 문서 140개는 `legacy/v2_experiments.zip`에 묶었다.
이 ZIP은 로컬 전용이다. 상세 이력이 필요하면 임시 폴더에 풀어 본다.

```bash
python -m zipfile -l legacy/v2_experiments.zip
python -m zipfile -e legacy/v2_experiments.zip /tmp/rating_recsys_history
```

보관된 MLflow 실행의 경로는 당시 artifact 위치다. 이전 run을 확인하려면 ZIP을 풀어
원본을 보고, 같은 데이터로 재현할 때는 당시 commit·평가 프로토콜을 맞춘다.

## 테스트 관리

```bash
python -m pytest -q tests
```

| 파일 | 역할 |
|---|---|
| `test_data.py` | 설정·적재 변환·전역 분할·snapshot |
| `test_retrieval.py` | 후보 제외·결정성·LightGCN·텍스트 모델 |
| `test_ranking.py` | 지표 수식·window label·shrinkage |
| `test_pipeline.py` | 학습/선택/평가·누수·CLI·저장 데이터 재사용/무효화/손상 복구 |
| `test_comparison.py` | 등록 모델의 공통 비교·학습 곡선·CLI·baseline 후보 재사용 |
| `test_observability.py` | MLflow·Streamlit 결과 조회 |

공통 fixture는 `support.py`다. 새 모델은 `MODEL_CASES`에 작은 설정을 추가하고
기존 retrieval/comparison 테스트를 사용한다. 기능별 테스트 파일을 계속 늘리지 않는다.
과거 V1 CSV 5개/31,085행을 통째로 읽는 검증은 V2 테스트에서 제거했다.
2026-10-04 저장 데이터 기능을 포함해 97개 테스트가 통과했다. 재사용 전후의 모델·추천
일치, 전처리 호출 생략, window/rating·shrinkage 재사용, 데이터·seed·전처리 변경 시 무효화,
손상 파일 재생성과 후보 비교 runner의 baseline 공유를 확인했다.
