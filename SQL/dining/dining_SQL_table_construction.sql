SET @regions = 'seoul,busan,gyeonggi,daegu'; 
SET @query = '';

-- Recursion Query
SELECT GROUP_CONCAT(
    CONCAT('SELECT row_num, user_tot_avg_rating, user_tot_rating_num, user_tot_follow_num, user_rating, user_query, ''', 
           region, ''' AS region FROM diningcode_', region)
    SEPARATOR ' UNION ALL '
) INTO @query
FROM (
    SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(@regions, ',', n.digit+1), ',', -1) AS region
    FROM (SELECT 0 AS digit UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) n 
) AS region_list;

-- Final Query for ITEMS
SET @final_query = CONCAT('CREATE TABLE items AS ', @query);

PREPARE stmt FROM @final_query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
-- -----------------------------------------------------------------
-- Create users table
SET @regions = 'seoul,busan,gyeonggi,daegu'; 
SET @query = '';
SELECT GROUP_CONCAT(
    CONCAT('SELECT row_num,item_name,item_avg_rating,item_spec_area,city, ''', 
           region, ''' AS region FROM diningcode_', region)
    SEPARATOR ' UNION ALL '
) INTO @query
FROM (
    SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(@regions, ',', n.digit+1), ',', -1) AS region
    FROM (SELECT 0 AS digit UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) n 
) AS region_list;


SET @final_query = CONCAT('CREATE TABLE users AS ', @query);

PREPARE stmt FROM @final_query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
-- ---------------------------------------------------------
-- Create reviews table
SET @regions = 'seoul,busan,gyeonggi,daegu'; 
SET @query = '';
SELECT GROUP_CONCAT(
    CONCAT('SELECT row_num,user_rating,user_query,taste,price,service,menu,item_name,item_area,item_spec_area, ''', 
           region, ''' AS region FROM diningcode_', region)
    SEPARATOR ' UNION ALL '
) INTO @query
FROM (
    SELECT SUBSTRING_INDEX(SUBSTRING_INDEX(@regions, ',', n.digit+1), ',', -1) AS region
    FROM (SELECT 0 AS digit UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) n 
) AS region_list;


SET @final_query = CONCAT('CREATE TABLE reviews AS ', @query);

PREPARE stmt FROM @final_query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
--	--------------------------------------------------------
-- Create regions table
CREATE TABLE regions (
    region_name VARCHAR(20) NOT NULL UNIQUE
);
INSERT INTO regions (region_name) VALUES 
('seoul'),('gyeonggi'),('busan'),('daegu');
