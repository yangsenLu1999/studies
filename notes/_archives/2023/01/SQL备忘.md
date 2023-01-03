SQL 备忘
===
<!--START_SECTION:badge-->

![last modify](https://img.shields.io/static/v1?label=last%20modify&message=2023-01-04%2000%3A08%3A14&color=yellowgreen&style=flat-square)

<!--END_SECTION:badge-->
<!--info
top: false
hidden: false
-->

<!-- TOC -->
<!-- TOC -->

> 关键词: SQL, Hive SQL, Spark SQL, Presto SQL

## 参考文档
- [Hive SQL - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual)
    - [Hive SQL build-in Functions - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF)
- [Presto Documentation](https://prestodb.io/docs/current/)
- [Spark SQL and DataFrames - Spark Documentation](https://spark.apache.org/docs/latest/sql-programming-guide.html)
    - [Functions - Spark Documentation](https://spark.apache.org/docs/latest/sql-ref-functions.html)
    - [Spark SQL, Built-in Functions](https://spark.apache.org/docs/latest/api/sql)


## 常用 SQL

### 建表
> [CreateTable - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTable)
```sql
-- Hive/Spark 适用
-- 示例 1: 分区表, 
CREATE TABLE db_name.table_name (
    `column_1`      string COMMENT '列注释1'
    , `column_2`    bigint COMMENT '列注释2'
    , `column_3`    double COMMENT '列注释3'
    , `column_4`    array < string > COMMENT '列注释4'
) COMMENT 'datastudio 表ddl模板' 
PARTITIONED BY (
  `pt` string COMMENT '分区注释1' 
  [, `hr` string COMMENT '分区注释2']
)
STORED AS ORC   -- 一种压缩格式
;

-- 示例 2: 外部 text 数据
CREATE EXTERNAL TABLE page_view (
    viewTime INT, userid BIGINT,
    page_url STRING, referrer_url STRING,
    ip STRING COMMENT 'IP Address of the User',
    country STRING COMMENT 'country of origination'
)
COMMENT 'This is the staging page view table'
ROW FORMAT DELIMITED    -- 指定行分隔符, 配合文本格式使用
    FIELDS TERMINATED BY '\t'   -- 列分割
    LINES TERMINATED BY '\n'    -- 行分割 (默认)
STORED AS TEXTFILE      -- 存储为文本格式
LOCATION '<hdfs_location>'
;

-- 示例 3: Create Table As Select
CREATE TABLE new_key_value_store
   ROW FORMAT SERDE "org.apache.hadoop.hive.serde2.columnar.ColumnarSerDe"
   STORED AS RCFile     -- 一种文件格式
   AS   -- 不指定列时, 直接使用查询结果的列名和类型 (无注释)
SELECT 
    (key % 1024) AS new_key
    , concat(key, value) AS key_value_pair
FROM key_value_store
SORT BY new_key, key_value_pair  -- 
;
```


### 临时表
```sql
-- Hive SQL
CREATE TEMPORARY TABLE IF NOT EXISTS tmp_table AS
SELECT ...
;

-- Spark SQL
CACHE TABLE tmp_table_name AS
SELECT ...
;

-- 物理临时表, 一些脚本中使用, 易于调试, 可在不同的提交中重复使用;
DROP TABLE IF EXISTS dbname.tmp_tabel_name;
CREATE TABLE dbname.tmp_tabel_name AS  
SELECT  ...
```

### 