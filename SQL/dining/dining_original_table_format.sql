CREATE TABLE IF NOT EXISTS diningcode_busan (
    id                  INT,
    item_name           VARCHAR(100),
    item_area           VARCHAR(50),
    item_avg_rating     FLOAT,
    item_spec_area      VARCHAR(200),
    user_name           VARCHAR(100),
    user_tot_avg_rating FLOAT,
    user_tot_rating_num INT,
    user_tot_follow_num INT,
    user_rating         FLOAT,
    user_query          TEXT,
    taste               TINYINT  COMMENT '0부족 1보통 2좋음',
    price               TINYINT  COMMENT '0불만 1보통 2만족',
    service             TINYINT  COMMENT '0나쁨 1보통 2좋음',
    menu                TEXT,
    reviewed_at         DATE
) 
ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Data Loading

LOAD DATA LOCAL INFILE 'C:/Users/Public/diningcode_clean.csv'
INTO TABLE diningcode_busan
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\\n'
IGNORE 1 LINES
(
    id, item_name, item_area, item_avg_rating, item_spec_area,
    user_name, user_tot_avg_rating, user_tot_rating_num, user_tot_follow_num,
    @raw_rating, user_query,
    @raw_taste, @raw_price, @raw_service,
    menu, @raw_date
)
SET
    user_rating = NULLIF(REGEXP_REPLACE(@raw_rating, '[^0-9.]', ''), ''),
    taste       = CASE @raw_taste
                    WHEN '맛: 좋음' THEN 2 WHEN '맛: 보통' THEN 1 WHEN '맛: 부족' THEN 0
                    ELSE NULL END,
    price       = CASE @raw_price
                    WHEN '가격: 만족' THEN 2 WHEN '가격: 보통' THEN 1 WHEN '가격: 불만' THEN 0
                    ELSE NULL END,
    service     = CASE @raw_service
                    WHEN '응대: 좋음' THEN 2 WHEN '응대: 보통' THEN 1 WHEN '응대: 나쁨' THEN 0
                    ELSE NULL END,
    reviewed_at = STR_TO_DATE(
                    REGEXP_REPLACE(@raw_date, '[년월 ]', '-'),
                    '%Y-%m-%d일'
                  );