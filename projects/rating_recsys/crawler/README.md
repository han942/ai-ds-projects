# DiningCode Playwright crawler

전국 수집은 DiningCode의 서울·경기·인천·부산·대구·광주·대전·울산·세종·강원·충북·충남·전북·전남·경북·경남·제주 대표 음식 페이지를 순회합니다. 각 페이지에서 현재 공개 목록에 표시되는 식당 상세 페이지를 모은 뒤 방문자 평가를 수집합니다. 이 목록은 지역별 대표 맛집 후보이며, 검색 결과의 1만여 식당 전체를 뜻하지는 않습니다.

## 저장 방식

기존 DiningCode 15개 컬럼과 restaurant_id, source_url, crawl_timestamp를 유지합니다. 방문자 평가 카드(div.latter-graph)마다 한 행을 만들고 작성자명, 작성자의 평균 평점·평가 수·팔로워 수, 해당 방문자 평점, 리뷰 본문, 맛·가격·응대, 주문 메뉴, 작성일을 각각 추출합니다. 플랫폼이 제공하는 별도 사용자 ID나 리뷰 ID는 저장하지 않습니다. 같은 표시 이름은 적재 단계에서 동일한 사용자로 매핑하며, 중복 행은 식당·작성자명·날짜·평점·리뷰 본문 조합으로 걸러냅니다. 리뷰 본문에 카드별 더보기 항목이 있으면 먼저 눌러 본문을 펼치고, review_text_complete 컬럼에는 더보기나 말줄임표가 남았는지 표시합니다. 본문 없이 별점만 남긴 평가는 user_query를 비운 채 한 행으로 보존합니다. 새 카드 배치가 화면에 나타날 때마다 CSV에 바로 append하고 flush합니다.

수집 중 파일은 .csv.partial, 진행 상태는 같은 이름의 .checkpoint.json에 기록합니다. 목록과 모든 식당의 방문자 평가 수 표시를 채우거나 평가 더보기 버튼을 눌러 끝까지 불러온 경우에만 .csv.partial을 최종 .csv로 바꿉니다. 중단되면 같은 명령에 --resume을 붙여 이어갈 수 있습니다.

## 실행

~~~bash
cd projects/rating_recsys
.venv/bin/python -m pip install -r crawler/requirements.txt
.venv/bin/python -m playwright install chromium
# Ubuntu/WSL에서 시스템 라이브러리 오류가 나면 한 번 실행
sudo .venv/bin/python -m playwright install-deps chromium
# 전국 17개 시·도 대표 음식 목록에서 발견되는 식당 전체
.venv/bin/python crawler/diningcode_playwright.py --national-regions --max-restaurants 0
~~~

기본값은 headless=False라 Chromium 창이 표시됩니다. 창 없이 실행하려면 --headless를 지정합니다. 전국 17개 지역 모드는 --national-regions를 사용합니다. 발견한 모든 식당을 선택하려면 --max-restaurants 0을 지정합니다. 목록 페이지 상한은 --max-list-pages 10, 식당 목록 스크롤 상한은 --max-list-scrolls 40, 식당별 평가 더보기 클릭 상한은 --max-evaluation-clicks 1000입니다. 버튼이 계속 활성 상태이고 평가 행이 추가되면 버튼이 없어질 때까지 반복합니다. 상한에 도달하면 미완료로 표시하고 체크포인트를 남깁니다.

중단된 수집을 이어가려면 같은 출력 디렉터리와 시작 URL로 실행합니다. 식당 하나만 확인하려면 --profile-url에 상세 페이지 URL을 전달할 수 있습니다. 이전 추출기에서 만든 이전 schema_version 체크포인트는 새 셀렉터로 다시 수집하도록 새 CSV와 체크포인트를 만들며, 이전 파일은 보존합니다.

~~~bash
.venv/bin/python crawler/diningcode_playwright.py --national-regions --max-restaurants 0 --resume
~~~

이미 시도했지만 불완전·차단·중단 상태인 식당의 재시도를 건너뛰고 새 대기 식당부터 진행하려면 `--skip-incomplete`를 함께 지정합니다. 기존 부분 CSV에 저장된 리뷰 행은 보존됩니다. 의도적으로 미룬 프로필 URL은 체크포인트의 `skip_incomplete_urls`에 기록됩니다.

~~~bash
.venv/bin/python crawler/diningcode_playwright.py --national-regions --max-restaurants 0 --resume --skip-incomplete
~~~

미완료 파일은 crawler/data 아래에서 .csv.partial 및 .checkpoint.json으로 확인할 수 있습니다. 첫 파일을 엑셀에서 바로 열려면 수집 완료 후 생성되는 .csv를 사용하세요.

## 접근 제한과 완료 판정

실행 전에 DiningCode의 robots.txt를 확인하고, 브라우저 요청도 같은 규칙으로 제한합니다. 로그인, CAPTCHA, 403, 429가 나타나면 멈추며 우회하지 않습니다. 평가 더보기는 화면에 보이는 컨트롤을 반복해서 누르고, 새 평가 카드가 더 나오지 않을 때까지 확인합니다. 리뷰 본문 안의 `...더보기`는 카드별로 펼친 뒤 저장하며, `review_text_complete`에는 본문이 여전히 잘렸는지 기록합니다. 현재 확인한 DiningCode 본문 컨트롤은 `a.more-btn-inline`이고 링크가 `javascript:void(0)`인 로컬 UI 동작입니다. 페이지에 평가 수가 표시되면 이 숫자와 수집한 행 수도 대조합니다. 더보기 컨트롤이 사라져도 수집 행이 표시된 평가 수보다 적으면 `.csv.partial`과 미완료 상태를 유지합니다.

식당 하나를 확인할 때는 `--profile-url`을 지정합니다. 각 평가 더보기 클릭 직후 체크포인트에 클릭 수를 저장하므로 실행 중에도 진행 상황을 확인할 수 있습니다. 이전 추출 규칙으로 만든 체크포인트는 schema 버전이 다르면 보존하고 새 결과 파일에서 다시 수집합니다.
