# [SQL/Recsys] Constructing text-embedded hybrid recommendation models for korean dining resturants

## 👨‍🏫 개요
* __문제 정의__: 
    * 기존의 rating data만을 이용한 추천시스템은 정확한 유저의 의견을 반영하지 못함
    * 이를 극복하기 위해 User의 리뷰 텍스트 데이터를 추천시스템에 결합한 일종의 Hybrid model 구축
    * 기존보다 우수한 예측 성능, 실질적인 User Experience를 반영할 수 있는 추천 모델 개발
* __기간/인원__: 2025. 12. 01 - 진행중 / 개인프로젝트 (1명)
* __예상 결과물__: 
    * 향상된 성능의 Recommendation Model
    * 유지 관리 방안
* __주요 역할__:
    * Data 수집 및 전처리
    * DeepCONN 모델 구현 및 기존 모델 구조와 통합
<br>

## 데이터 수집
   * __데이터__: [다이닝코드]
      * 지역 설정 조건을 활용해서 한국 내 주요 5개 대도시에 대한 데이터 수집
      * 평점,리뷰 텍스트 (user_query) 뿐만 아니라 장소(item_area), user 팔로워 수(user_tot_follow_num)와 같은 side feature도 수집
      * 

   * __전처리__
     * 각 Feature 특성에 맞게 정제 (object, numerical)
     * user_i2n, item_n2i와 같이 user-item 매핑 전처리 도입
     * 중복되어 수집된 데이터 제거 (Selenium 패키지의 time latency로 인해 발생)
    
## 모델 개발
   * __DeepCONN__ : [Joint Deep Modeling of Users and Items Using Reviews for Recommendation](https://arxiv.org/pdf/1701.04783)
      * 듀얼 네트워크 구조: User/Item(식당)에 대해 작성된 리뷰를 각각 통합하여 병렬로 구성하여 학습함
      * CNN 기반의 특징 추출: 리뷰 텍스트 데이터의 맥락 및 특징을 효과적으로 추출할 수 있음
      * 
 
