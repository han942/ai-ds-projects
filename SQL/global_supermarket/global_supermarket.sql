-- ============================================================
-- global_supermarket → 5개 테이블 분리 (row_id 기준)
-- 실행 순서: 1)테이블 생성 → 2)데이터 삽입 → 3)확인 쿼리
-- ============================================================

USE global_supermarket_db;  -- 사용 중인 DB명으로 변경하세요


-- ============================================================
-- STEP 1. 기존 테이블 초기화 (재실행 시 충돌 방지)
-- ============================================================
DROP TABLE IF EXISTS shipping;
DROP TABLE IF EXISTS market;
DROP TABLE IF EXISTS `order`;
DROP TABLE IF EXISTS product;
DROP TABLE IF EXISTS customer;


-- ============================================================
-- STEP 2. 테이블 생성 (row_id PK, FK 없음 → 단순 분리)
-- ============================================================

-- 2-1. customer
CREATE TABLE customer (
    row_id           INT          NOT NULL,
    customer_id      VARCHAR(20),
    customer_name    VARCHAR(100),
    customer_segment VARCHAR(50),
    PRIMARY KEY (row_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- 2-2. product
CREATE TABLE product (
    row_id       INT          NOT NULL,
    product_id   VARCHAR(30),
    product_name VARCHAR(300),
    category     VARCHAR(50),
    sub_category VARCHAR(50),
    PRIMARY KEY (row_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- 2-3. market
CREATE TABLE market (
    row_id         INT          NOT NULL,
    market_country VARCHAR(100),
    market_area    VARCHAR(10),
    market_city    VARCHAR(100),
    order_city     VARCHAR(100),
    order_region   VARCHAR(100),
    PRIMARY KEY (row_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- 2-4. order
CREATE TABLE `order` (
    row_id        INT          NOT NULL,
    order_id      VARCHAR(30),
    order_date    DATE,
    order_year    SMALLINT,
    order_weeknum TINYINT,
    quantity      INT,
    sales         INT,
    profit        DECIMAL(10, 4),
    discount      INT,
    PRIMARY KEY (row_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- 2-5. shipping
CREATE TABLE shipping (
    row_id        INT          NOT NULL,
    ship_date     DATE,
    ship_mode     VARCHAR(50),
    shipping_cost DECIMAL(10, 4),
    PRIMARY KEY (row_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- STEP 3. 데이터 삽입 (global_supermarket에서 row_id 기준 추출)
-- ============================================================

-- 3-1. customer
INSERT INTO customer (row_id, customer_id, customer_name, customer_segment)
SELECT
    row_id,
    customer_id,
    customer_name,
    customer_segment
FROM global_supermarket;


-- 3-2. product
INSERT INTO product (row_id, product_id, product_name, category, sub_category)
SELECT
    row_id,
    product_id,
    product_name,
    category,
    sub_category
FROM global_supermarket;


-- 3-3. market
INSERT INTO market (row_id, market_country, market_area, market_city, order_city, order_region)
SELECT
    row_id,
    market_country,
    market_area,
    market_city,
    order_city,
    order_region   -- 원본 오타(oreder_region) → 이미 정규화된 컬럼명
FROM global_supermarket;


-- 3-4. order
INSERT INTO `order` (
    row_id, order_id,
    order_date, order_year, order_weeknum,
    quantity, sales, profit, discount
)
SELECT
    row_id,
    order_id,
    order_date,
    order_year,
    order_weeknum,
    quantity,
    sales,
    profit,
    discount
FROM global_supermarket;


-- 3-5. shipping
INSERT INTO shipping (row_id, ship_date, ship_mode, shipping_cost)
SELECT
    row_id,
    ship_date,
    ship_mode,
    shipping_cost
FROM global_supermarket;
