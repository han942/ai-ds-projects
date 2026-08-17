# 🛒 Global Supermarket Analysis

> 51,290건의 글로벌 소매 거래 데이터를 EDA·손실 분석·다변량 분석으로 파고들어,
> **전체 손실의 68%가 소수의 고할인 거래에서 발생**한다는 구조적 수익 누수를 규명한 프로젝트입니다.

분석 결과를 정적 리포트(PDF)뿐 아니라 **인터랙티브 대시보드**로도 볼 수 있게 재구성했습니다.

[English README](README.md)

| | |
|---|---|
| **데이터** | Global Supermarket · 51,290 rows × 24 cols · 2019–2022 |
| **총 매출 / 총 이익** | $12.64M / $1.47M (이익률 **11.6%**) |
| **핵심 발견** | 할인율 40% 초과 거래 = 전체의 13.6%지만 **총 손실의 68.2%** 차지 |
| **스택** | Python (Pandas · NumPy · Matplotlib · Seaborn · SciPy) · MySQL · Vanilla JS 대시보드 |

---

## 📊 대시보드

핵심 결과를 한 화면에서 탐색할 수 있는 단일 HTML 대시보드입니다. 별도 서버나 설치 없이 브라우저에서 바로 열 수 있습니다.

- **파일:** [`dashboard.html`](dashboard.html) — 데이터가 내장된 self-contained 파일(더블클릭으로 실행)
- **구성:** KPI 헤더 · 매출/이익 추이 · 지역별 손익 · 카테고리별 손익 · 할인·손실 심층 분석 · 배송 분석 · 전략 제언
- **기능:** 호버 툴팁, 차트↔표 전환, 라이트/다크 테마 지원

```bash
# 로컬에서 열기
open dashboard.html        # macOS
xdg-open dashboard.html    # Linux
```

---

## 🗂️ 저장소 구조

```
Global_Supermarket_Analysis/
├── README.md                             # 영어 문서
├── README_KOR.md                         # 한글 문서
├── global_supermarket.ipynb              # 메인 분석 노트북 (EDA · 손실 · 다변량 · 할인)
├── gl_supermarkt_recsys.ipynb            # 데이터 기반 추천 시스템 실험
├── global_supermarket_to_sql.ipynb       # 정제 데이터를 SQL로 적재
├── dashboard.html                        # 인터랙티브 대시보드 (self-contained)
├── dashboard_build/
│   ├── build_dashboard_data.py           # CSV → 집계 JSON 생성 스크립트
│   └── dashboard_data.json               # 대시보드용 사전 집계 데이터
├── Global_Supermarket_Analysis.pdf       # 최종 리포트 (영문)
├── Global_Supermarket_Analysis_korean.pdf# 최종 리포트 (국문)
└── Global_Supermarket_Analysis_db_schema.png  # 정규화 스키마 다이어그램

../../SQL/global_supermarket/
├── global_supermarket.sql                # DDL/ETL: 플랫 CSV → 5개 정규화 테이블
└── global_supermarket_ERD.mwb             # MySQL Workbench ERD
```

---

## 🧪 데이터 & 파이프라인

**원본 데이터** (`datafile/[DArt-B 5기] Global_Supermarket.csv`)는 51,290행, 24열이며 결측치는 0%입니다.
고객·주문·상품·시장(지역)·배송 정보를 포함한 "Global Superstore" 형태의 데이터셋으로,
US·EU·APAC·LATAM·Africa 등 여러 시장에 걸쳐 있습니다.

**① 피처 엔지니어링** — 원본 지표를 넘어선 경제 지표를 직접 설계했습니다.

| 피처 | 정의 | 목적 |
|---|---|---|
| `pre_sales` | 할인 적용 전 원매출 복원 | 가격 정책의 실제 영향 측정 |
| `uni_cost` | 상품 단위 원가 | 마진 구조의 건전성 평가 |
| `eta` | 주문일 → 배송일 소요 기간 | 물류 리드타임 성과 평가 |

**② 정규화 (SQL)** — 플랫 CSV를 `row_id` 기준으로 5개 테이블(`customer` · `product` · `market` · `order` · `shipping`)로 분해했습니다. 스키마는 [`Global_Supermarket_Analysis_db_schema.png`](Global_Supermarket_Analysis_db_schema.png), DDL/ETL은 [`SQL/global_supermarket/global_supermarket.sql`](../../SQL/global_supermarket/global_supermarket.sql)을 참고하세요.

**③ 분석** — 고객·주문·상품·시장·배송에 대한 EDA 이후 상관 분석 → 손실 분석 → 이익 분석 → 다변량 분석 → 할인 분석 순으로 진행했습니다.

---

## 🔍 핵심 발견

### 1. 할인이 이익을 갉아먹는다

- 할인율과 이익의 피어슨 상관계수는 **-0.32**(검증값 -0.316)로 뚜렷한 음의 상관관계를 보입니다.
- **할인율 40% 초과 거래는 전체의 13.6%에 불과**하지만, **전사 총 손실의 68.2%**를 차지합니다.
  → 손실은 광범위하게 퍼져 있는 것이 아니라 소수의 고할인 거래에 집중되어 있습니다.

### 2. Tables가 유일한 적자 서브카테고리다

- 17개 서브카테고리 중 **Tables만 순손실**(약 -$64K)을 기록해 마진 압박의 핵심 요인으로 나타났습니다.
- 제조사별로 보면 **Bevis·Barricks·Lesro**가 손실 상위 3사입니다.

### 3. 지역별 손익 편차

- 순이익 규모는 **APAC > EU > US > LATAM** 순으로 크며, 매출이 큰 시장일수록 손실 규모도 함께 커지는 경향이 있습니다.
- 특히 APAC·US·LATAM에서는 테이블 판매의 손실 기여도가 시장 전체 손실 대비 과도하게 높습니다.

### 4. 우상향하는 거래량과 뚜렷한 계절성

- 거래 건수가 2019년 8,998건에서 2022년 17,531건으로 **약 95%** 성장했으며, 2022년에는 전년 대비 **27.0%** 증가했습니다.
- 매년 **하반기(9–12월)**에 뚜렷한 성수기 피크가 반복됩니다.

---

## 💡 전략적 제언

| # | 제언 | 근거 |
|---|---|---|
| 1 | **고거래량 카테고리에 할인 상한 40% 도입** (Phones, Bookcases 등) | 40% 초과 할인이 전체 손실의 68% 유발 |
| 2 | **EU 평균 할인율 20%p 하향** | EU의 큰 손실 규모를 줄이고 지역 수익성 교정 |
| 3 | **고손실 제조사의 테이블 상품군 축소·단종** (Bevis, Barricks, Lesro) | Tables가 유일한 적자 서브카테고리이며 해당 제조사의 손실이 큼 |
| 4 | **고손실 도시의 가격 정책 재설계** (Istanbul, Lagos 등) | 할인율을 40% 이하 구간으로 재구성 |

---

## 🔁 재현 방법

```bash
# 1. 대시보드용 집계 데이터 재생성 (원본 CSV 필요)
python dashboard_build/build_dashboard_data.py
#    → dashboard_build/dashboard_data.json 갱신
#    (총 매출/이익, 상관계수 -0.316, 고할인 손실 비중 68.2% 등을 콘솔에 출력)

# 2. 대시보드 열기
open dashboard.html

# 3. 전체 분석 재현
jupyter notebook global_supermarket.ipynb
```

> 참고: `dashboard.html`은 집계 결과가 내장된 정적 파일이라 CSV 없이도 바로 열 수 있습니다. 원본 데이터가 갱신되면 1번 스크립트를 다시 실행해 대시보드에 반영하세요.

---

## 🧰 기술 스택

- **언어:** Python, SQL, JavaScript
- **분석:** Pandas · NumPy · SciPy(통계) · Matplotlib · Seaborn(시각화)
- **데이터베이스:** MySQL(정규화 스키마 설계 · ETL)
- **대시보드:** 의존성 없는 Vanilla JS + SVG(외부 라이브러리 미사용)
- **기법:** 다변량 EDA · 피처 엔지니어링 · 상관 분석 · 리스크 프로파일링 · BI 리포팅
