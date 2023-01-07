SQL 备忘
===
<!--START_SECTION:badge-->

![last modify](https://img.shields.io/static/v1?label=last%20modify&message=2023-01-06%2000%3A09%3A20&color=yellowgreen&style=flat-square)

<!--END_SECTION:badge-->
<!--info
top: false
hidden: false
-->

> 关键词: Hive SQL, Spark SQL, Presto SQL

<!-- TOC -->
- [参考文档](#参考文档)
- [常用 SQL](#常用-sql)
    - [建表 (Hive/Spark)](#建表-hivespark)
    - [临时表](#临时表)
    - [`WITH t AS (...)` 子查询](#with-t-as--子查询)
    - [分页](#分页)
    - [`collect_list` 计数排序](#collect_list-计数排序)
<!-- TOC -->



## 参考文档
- [Hive SQL - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual)
    - [Hive SQL build-in Functions - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF)
- [Presto Documentation](https://prestodb.io/docs/current/)
- [Spark SQL and DataFrames - Spark Documentation](https://spark.apache.org/docs/latest/sql-programming-guide.html)
    - [Functions - Spark Documentation](https://spark.apache.org/docs/latest/sql-ref-functions.html)
    - [Spark SQL, Built-in Functions](https://spark.apache.org/docs/latest/api/sql)


## 常用 SQL

### 建表 (Hive/Spark)
> [CreateTable - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTable)
```sql
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
-- Hive
CREATE TEMPORARY TABLE IF NOT EXISTS tmp_table AS
SELECT ...
;

-- Spark
CACHE TABLE tmp_table_name AS
SELECT ...
;

-- 物理临时表, 一些脚本中使用, 易于调试, 可在不同的提交中重复使用;
DROP TABLE IF EXISTS dbname.tmp_tabel_name;
CREATE TABLE dbname.tmp_tabel_name AS  
SELECT  ...
```

### `WITH t AS (...)` 子查询
> [Common Table Expression (CTE) - Spark Documentation](https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-cte.html)
>> Hive 官方文档没查到相关的语法, 有些环境确实也不支持这个语法;
```sql
-- CTE with multiple column aliases
WITH t(x, y) AS (SELECT 1, 2)
SELECT * FROM t WHERE x = 1 AND y = 2;

-- CTE in subquery
SELECT max(c) FROM (
    WITH t(c) AS (SELECT 1)
    SELECT * FROM t
);

-- 多表
WITH t1 AS (
    SELECT ...
),
t2 AS (
    SELECT ...
),
t3 AS (
    SELECT * FROM t1  -- 可以使用前面子查询的结果
    WHERE ...
)
SELECT * FROM t2 JOIN t3 ON ...;

-- 嵌套 (好像有的 sql 不支持)
WITH t AS (
    WITH t2 AS (SELECT 1)
    SELECT * FROM t2
)
SELECT * FROM t;

-- 嵌套冲突时
SET spark.sql.legacy.ctePrecedencePolicy = CORRECTED;
-- 如果不开启该设置, 会抛 AnalysisException
WITH
    t AS (SELECT 1),
    t2 AS (
        WITH t AS (SELECT 2)
        SELECT * FROM t
    )
SELECT * FROM t2;
```

### 分页
> 使用场景: 限制每次下载/浏览的数据量时

```sql
-- 写法 1: 适用于小数据量, 且不含 ORDER BY 的情况, 如果存在 ORDER BY, 推荐写法 2
SELECT * 
FROM table_name A
WHERE ...
-- ORDER BY ...
LIMIT page_sz OFFSET (page_id - 1) * page_sz
-- LIMIT (page_id - 1) * page_sz, page_sz  -- 等价写法
;

-- 写法 2: 基于 row_number() 给每行添加一个可比较的自增 rn (从 1 开始)
SELECT * FROM (
    SELECT A.*, row_number() over (ORDER BY ...) as rn
    FROM table_name A
    WHERE ...
) A
WHERE rn > (page_id - 1) * page_sz AND rn <= (page_id * page_sz)
-- WHERE rn BETWEEN ((page_id - 1) * page_sz + 1) AND (page_id * page_sz);  -- 等价写法

-- 传统数据库因为存在主键, 还有其他写法, 这里略
```
> [传统数据库 SQL 窗口函数实现高效分页查询的案例分析_MsSql_脚本之家](https://www.jb51.net/article/212864.htm)


### `collect_list` 计数排序
> [hiveql - Sorting within collect_list() in hive - Stack Overflow](https://stackoverflow.com/questions/50766764/sorting-within-collect-list-in-hive/72458308#72458308)

```sql
SELECT key
    , sort_array(collect_list(STRUCT(-cnt, item_cnt_map))).col2 as item_cnt_list_sorted
    -- col2 为 STRUCT 中的默认列名, 负号表示倒序排列
FROM
(
    SELECT key, cnt
        , MAP(item, cnt) AS item_cnt_map
        -- , concat(item, ':', cnt) AS item_cnt_map
    FROM
    (
        SELECT key, item, count(1) AS cnt
        -- FROM db.some_table A
        FROM (
            SELECT 'A' AS key, 'red' AS item UNION ALL
            SELECT 'A' AS key, 'red' AS item UNION ALL
            SELECT 'A' AS key, 'blue' AS item UNION ALL
            SELECT 'A' AS key, 'blue' AS item UNION ALL
            SELECT 'A' AS key, 'yellow' AS item UNION ALL
            SELECT 'A' AS key, 'yellow' AS item UNION ALL
            SELECT 'A' AS key, 'yellow' AS item UNION ALL
            SELECT 'B' AS key, 'yellow' AS item UNION ALL
            SELECT 'B' AS key, 'yellow' AS item UNION ALL
            SELECT 'B' AS key, 'green' AS item
        ) A
        GROUP BY key, item
    ) A
) A
GROUP BY key
;
-- | A | [{"yellow":3},{"blue":2},{"red":2}] |
-- | B | [{"yellow":2},{"green":1}] |
```
- `sort_array(collect_list(STRUCT(cnt, item_cnt_map))).col2` 相当于 python 中的对一个**元组列表**进行排序, 排序的 key 依次从元组中取;
