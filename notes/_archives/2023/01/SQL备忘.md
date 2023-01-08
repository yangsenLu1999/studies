SQL 备忘
===
<!--START_SECTION:badge-->

![last modify](https://img.shields.io/static/v1?label=last%20modify&message=2023-01-08%2016%3A00%3A31&color=yellowgreen&style=flat-square)

<!--END_SECTION:badge-->
<!--info
top: false
hidden: false
-->

> 关键词: Hive SQL, Spark SQL, Presto SQL

<!-- TOC -->
- [参考文档](#参考文档)
- [数据类型](#数据类型)
    - [基本类型 TODO](#基本类型-todo)
    - [容器类型](#容器类型)
- [常用 DDL](#常用-ddl)
    - [建表 (`CREATE`)](#建表-create)
        - [临时表](#临时表)
    - [修改 (`ALTER`)](#修改-alter)
        - [修改列](#修改列)
        - [增加列](#增加列)
- [常用 DQL](#常用-dql)
    - [聚合操作 (`GROUP BY`)](#聚合操作-group-by)
        - [排序 `sort_array(collect_list(...))`](#排序-sort_arraycollect_list)
    - [侧视图 (`LATERAL VIEW`)](#侧视图-lateral-view)
        - [侧视图 for presto](#侧视图-for-presto)
    - [窗口函数](#窗口函数)
        - [排序 (`ROW_NUMBER/RANK/DENSE_RANK`)](#排序-row_numberrankdense_rank)
        - [切片 (`NTILE`)](#切片-ntile)
    - [子查询 (`WITH t AS (...)`)](#子查询-with-t-as-)
    - [数组操作](#数组操作)
    - [分页](#分页)
- [易错记录](#易错记录)
    - [修改 `GROUP BY` 的对象](#修改-group-by-的对象)
    - [日期加减](#日期加减)
<!-- TOC -->



## 参考文档
- [Hive SQL - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual)
    - [Hive SQL build-in Functions - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF)
- [Presto Documentation](https://prestodb.io/docs/current/)
- [Spark SQL and DataFrames - Spark Documentation](https://spark.apache.org/docs/latest/sql-programming-guide.html)
    - [Functions - Spark Documentation](https://spark.apache.org/docs/latest/sql-ref-functions.html)
    - [Spark SQL, Built-in Functions](https://spark.apache.org/docs/latest/api/sql)

## 数据类型

### 基本类型 TODO

### 容器类型
- 主要有 4 中容器类型: 
    - `ARRAY<data_type>`
    - `MAP<primitive_type, data_type>`
    - `STRUCT<col_name : data_type [COMMENT col_comment], ...>`
    - `UNIONTYPE<data_type, data_type, ...>` (支持不完整, 一般不用)

```sql
-- ARRAY
SELECT ARRAY(1,2,3)  -- [1,2,3]
SELECT ARRAY('a','b','c')[0]  -- "a"

-- MAP
SELECT MAP('a', 1), MAP('b', '2', 'c', 3)
-- {"a":1}  {"b":"2","c":"3"}  -- 注意, 整数 3 转成了字符串 3
SELECT MAP('a', 1)['a']  -- 1

-- STRUCT
SELECT STRUCT('a', 1, ARRAY(1,2,3))
-- {"col1":"a","col2":1,"col3":[1,2,3]}
SELECT STRUCT('a', 1, ARRAY(1,2,3)).col2  -- 1

-- ARRAY + STRUCT
SELECT ARRAY(STRUCT('a', 1), STRUCT('b', 2), STRUCT('c', 3)).col1
-- ["a","b","c"]
```

## 常用 DDL
> 数据定义语言 (Data Definition Language, DDL)

### 建表 (`CREATE`)
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

#### 临时表
- 注意: Hive 和 Spark 中的临时表语法不同; 
```sql
-- Hive
CREATE TEMPORARY TABLE IF NOT EXISTS tmp_table AS
SELECT ...
;

-- Spark
CACHE TABLE tmp_table AS
SELECT ...
;

-- Hive/Spark 都适用 (如果公司支持自动删除临时表, 推荐这种写法)
-- 物理临时表, 一些脚本中使用, 易于调试, 可重复使用;
DROP TABLE IF EXISTS db.tmp_tabel;
CREATE TABLE db.tmp_tabel AS
SELECT  ...
```

### 修改 (`ALTER`)
> [Alter Table/Partition/Column - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-AlterTable/Partition/Column)

#### 修改列
> [Change Column Name/Type/Position/Comment - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-ChangeColumnName/Type/Position/Comment)

- 一条 `ALTER` 语句一次只能修改一列;
```sql
-- 语法
ALTER TABLE table_name [PARTITION partition_spec] CHANGE [COLUMN] col_old_name col_new_name column_type
  [COMMENT col_comment] [FIRST|AFTER column_name] [CASCADE|RESTRICT];

-- 基础示例
ALTER TABLE db.table CHANGE a x BIGINT COMMENT 'column x';  -- 修改列名
ALTER TABLE db.table CHANGE b b STRING COMMENT 'column b';  -- 修改类型 (名字不变)
```

#### 增加列
> [Add/Replace Columns - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-Add/ReplaceColumns)

- 一条 `ALTER` 语句一次只能修改一列, 但是能增加多列;
```sql
-- 语法
ALTER TABLE table_name 
  [PARTITION partition_spec]                 -- (Note: Hive 0.14.0 and later)
  ADD|REPLACE COLUMNS (col_name data_type [COMMENT col_comment], ...)
  [CASCADE|RESTRICT]                         -- (Note: Hive 1.1.0 and later)

-- 基础示例
ALTER TABLE db.table
ADD COLUMNS (
    a BIGINT    COMMENT 'column a'
    , b STRING  COMMENT 'column b'
)
```



## 常用 DQL
> 数据查询语言 (Data Query Language, DQL)

### 聚合操作 (`GROUP BY`)

#### 排序 `sort_array(collect_list(...))`
> [hiveql - Sorting within collect_list() in hive - Stack Overflow](https://stackoverflow.com/questions/50766764/sorting-within-collect-list-in-hive/72458308#72458308)

```sql
SELECT key
    , sort_array(collect_list(STRUCT(-cnt, item_cnt_pair))).col2 as item_cnt_list_sorted
    -- col2 为 STRUCT 中的默认列名, 负号表示倒序排列
FROM
(
    SELECT key, cnt
        , concat(item, ':', cnt) AS item_cnt_pair  -- 可以通过 SPLIT 取值
        -- , STRUCT(item, cnt) AS item_cnt_pair  -- 根据 STRUCT 的默认列名取值
        -- , MAP(item, cnt) AS item_cnt_pair  -- 可以利用 map_keys/map_values 取值
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
-- A, ["yellow:3","blue:2","red:2"]
-- B, ["yellow:2","green:1"]
```
- `sort_array(collect_list(STRUCT(cnt, item_cnt_pair))).col2` 相当于 python 中的对一个**元组列表**进行排序, 排序的 key 依次从元组中取;
    - 一般情况下, 先排序(`ORDER/SORT BY`), 再 `collect_list` 也可以, 但是速度比较慢;


### 侧视图 (`LATERAL VIEW`)
```sql
-- 语法
lateralView: LATERAL VIEW udtf(expression) tableAlias AS columnAlias (',' columnAlias)*
fromClause: FROM baseTable (lateralView)*

-- 示例 1: 单
SELECT pageid, adid
FROM pageAds 
LATERAL VIEW explode(adid_list) adTable AS adid
;

-- 示例 2: 多
SELECT myCol1, myCol2 FROM baseTable
LATERAL VIEW explode(col1) myTable1 AS myCol1
LATERAL VIEW explode(col2) myTable2 AS myCol2
;

-- 示例 3: LATERAL VIEW OUTER
SELECT * FROM src LATERAL VIEW explode(array()) C AS a limit 10;  -- 结果为空
SELECT * FROM src LATERAL VIEW OUTER explode(array()) C AS a limit 10;  -- 有结果
```

#### 侧视图 for presto
> 
```sql
```

### 窗口函数
> [HIVE SQL奇技淫巧 - 知乎](https://zhuanlan.zhihu.com/p/80887746)

#### 排序 (`ROW_NUMBER/RANK/DENSE_RANK`)
```sql
SELECT 
    cookieid, pt, pv,
    ROW_NUMBER() OVER(PARTITION BY cookieid ORDER BY pv desc) AS rn1,   -- 形如 1,2,3,4,5 (最常用)
    RANK() OVER(PARTITION BY cookieid ORDER BY pv desc) AS rn2,         -- 形如 1,1,3,3,5
    DENSE_RANK() OVER(PARTITION BY cookieid ORDER BY pv desc) AS rn3    -- 形如 1,1,2,2,3
FROM (
    -- 测试数据
    SELECT cookieid, T.col2[idx] AS pt, T.col3[idx] AS pv
    FROM (
        SELECT ARRAY(
            STRUCT('cookie1', '2015-04-10', 1)
            , STRUCT('cookie1', '2015-04-11', 5)
            , STRUCT('cookie1', '2015-04-12', 7)
            , STRUCT('cookie1', '2015-04-13', 3)
            , STRUCT('cookie1', '2015-04-14', 2)
            , STRUCT('cookie1', '2015-04-15', 4)
            , STRUCT('cookie1', '2015-04-16', 4)
            -- , STRUCT('cookie2', '2015-04-10', 2)
            -- , STRUCT('cookie2', '2015-04-11', 3)
            -- , STRUCT('cookie2', '2015-04-12', 5)
            -- , STRUCT('cookie2', '2015-04-13', 6)
            -- , STRUCT('cookie2', '2015-04-14', 3)
            -- , STRUCT('cookie2', '2015-04-15', 9)
            -- , STRUCT('cookie2', '2015-04-16', 7)
        ) T
    ) A
    LATERAL VIEW posexplode(T.col1) B AS idx, cookieid
) A
;
-- cookieid  pt          pv  rn1 rn2 rn3
-- cookie1   2015-04-12  7   1   1   1
-- cookie1   2015-04-11  5   2   2   2
-- cookie1   2015-04-15  4   3   3   3
-- cookie1   2015-04-16  4   4   3   3
-- cookie1   2015-04-13  3   5   5   4
-- cookie1   2015-04-14  2   6   6   5
-- cookie1   2015-04-10  1   7   7   6
```

#### 切片 (`NTILE`)



### 子查询 (`WITH t AS (...)`)
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

### 数组操作
- Hive 提供的内置函数较少, 一般使用外部 UDF; Spark 则提供了丰富的数组操作函数, 一般命名为 `array_*`;
- 常用的 Hive UDF 库
    - [brickhouse - Hive](https://github.com/klout/brickhouse);

```sql
-- 以下 函数名 默认兼容 Hive 和 Spark
-- brickhouse
ADD JAR hdfs://path/to/brickhouse.jar;
-- 交集(去重)
CREATE TEMPORARY FUNCTION array_intersect AS "brickhouse.udf.collect.ArrayIntersectUDF";
-- 并集(去重)
CREATE TEMPORARY FUNCTION array_union AS "brickhouse.udf.collect.ArrayUnionUDF";
-- 差集(存在的都移除, 不存在的都保留)
CREATE TEMPORARY FUNCTION array_except AS "brickhouse.udf.collect.SetDifferenceUDF";

-- 以下为 Hive 中测试结果, Spark 未测试
-- 交集
SELECT array_intersect(array(1,2,3,3), array(3,3,4,5));  -- [3]
SELECT array_intersect(array(1,2,2,3,3), array(2,3,3,4,5));  -- [2,3]
SELECT array_intersect(array(1,2,2,3,NULL), array(2,3,3,4,5));  -- [2,3]
SELECT array_intersect(array(1,2,2,3,NULL), array(2,3,3,4,NULL));  -- [null,2,3]
-- 并集
SELECT array_union(array(1,2,3,3), array(3,3,4,5));  -- [1,2,3,4,5]
SELECT array_union(array(1,2,3,NULL), array(3,3,4,5));  -- [null,1,2,3,4,5]
-- 差集
SELECT array_except(array(1,2,2,3), array(1,3,5)); -- [2,2], 不存在的都保留
SELECT array_except(array(1,2,2,3,3), array(1,3,5)); -- [2,2], 存在的都移除
SELECT array_except(array(1,2,2), array(1,2,3)); -- []
-- 判断 a 是否 b 的子集
SELECT size(array_except(array(1,2), array(1,2,3))) = 0;  -- true
SELECT size(array_except(array(1,2,2), array(1,2,3))) = 0;  -- true
SELECT size(array_except(array(1,2,2,NULL), array(1,2,3))) = 0;  -- false
SELECT size(array_except(array(1,2,2), array(1,2,3,NULL))) = 0;  -- true


```

<!-- 
#### 判断子集
- 一个通用的方法是编写 UDF
- 下面是一些 trick 方法

```sql
-- 判断 a 是否 b 的子集
-- 思路是利用 a 构造正则表达式, 判断 b 中是否存在 a 的所有元素
SELECT concat('#',concat_ws('#',b),'#') AS tmp_a
    , concat('#(',concat_ws('|',a),')#') AS tmp_b
    , concat('#',concat_ws('#',b),'#') RLIKE concat('#(',concat_ws('|',a),')#')  AS is_subset
FROM (
    SELECT array('a','b','c') AS a, array('b','c') AS b
    UNION ALL
    SELECT array('a','b','c') AS a, array('b','d') AS b
) A
-- #a#b#c#, #(b|c)#, true
-- #a#b#c#, #(b|d)#, false

```
-->

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


## 易错记录

### 修改 `GROUP BY` 的对象
```sql
SELECT lower(query) AS query, count(1) AS cnt
FROM ...
GROUP BY lower(query)   -- OK
-- GROUP BY query       -- err
```

### 日期加减
```sql
pt >  DATE_SUB('${YYYY-MM-DD}', 7)  -- 7 天
pt >= DATE_SUB('${YYYY-MM-DD}', 7)  -- 8 天
```