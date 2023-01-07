Hive SQL 备忘
===
<!--START_SECTION:badge-->

![last modify](https://img.shields.io/static/v1?label=last%20modify&message=2023-01-06%2000%3A29%3A03&color=yellowgreen&style=flat-square)

<!--END_SECTION:badge-->
<!--info
top: false
hidden: false
-->

<!-- TOC -->
- [参考文档](#参考文档)
- [常用语法](#常用语法)
    - [建表 CREATE](#建表-create)
    - [临时表](#临时表)
    - [侧视图 (LATERAL VIEW)](#侧视图-lateral-view)
    - [`CASE WHEN`](#case-when)
- [常用技巧](#常用技巧)
    - [分页](#分页)
    - [物理临时表](#物理临时表)
    - [对称元素去重](#对称元素去重)
    - [侧视图 - 多列转行](#侧视图---多列转行)
    - [侧视图 - 多 array 并行转行](#侧视图---多-array-并行转行)
- [常用函数](#常用函数)
    - [字符串](#字符串)
    - [数学](#数学)
- [常用设置](#常用设置)
- [TRANSFORM 脚本](#transform-脚本)
    - [MAP/REDUCE 语法](#mapreduce-语法)
- [其他](#其他)
    - [VTL 变量](#vtl-变量)
<!-- TOC -->


## 参考文档
- [LanguageManual - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual)
- [Hive SQL 教程 - 盖若](https://www.gairuo.com/p/hive-sql-tutorial)

## 常用语法

### 建表 CREATE
> [CreateTable - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTable)
```sql
DROP TABLE IF EXISTS `table_name`;
CREATE TABLE `table_name` 
STORED AS ORC 
AS
SELECT ...
;
```

### 临时表
```sql
CREATE TEMPORARY TABLE IF NOT EXISTS tmp_table AS
...
;
```

### 侧视图 (LATERAL VIEW)
```sql

```

### `CASE WHEN`
```sql
-- 写法 1: When a = b, returns c; when a = d, returns e; else returns f.
CASE a 
    WHEN b THEN c 
    [WHEN d THEN e]* 
    [ELSE f] 
END

-- 写法 2: When a = true, returns b; when c = true, returns d; else returns e.
CASE 
    WHEN a THEN b 
    [WHEN c THEN d]* 
    [ELSE e] 
END
```


## 常用技巧

### 分页
```sql

```

### 物理临时表
逻辑复杂时可以使用物理临时表, 用于排错;
```sql
DROP TABLE IF EXISTS dbname.tmp_tabel_name;
CREATE TABLE dbname.tmp_tabel_name AS  
SELECT  ...
```

### 对称元素去重
```sql
SELECT sort_array(array(A.x, A.y))[0] AS x
    , sort_array(array(A.x, A.y))[1] AS y
FROM some_table A
GROUP BY 1, 2
```
> hive 中数组下标从 0 开始; 一些 sql 语言是从 1 开始的, 如 presto;

### 侧视图 - 多列转行
```sql
SELECT A.x, B.y
FROM some_table A
LATERAL VIEW EXPLODE(array(A.a, A.b, A.c)) B AS y
```

### 侧视图 - 多 array 并行转行
- 类似 python 中 `for x, y, z in zip(a, b, c): ...` 的操作;
```sql
SELECT B.x, A.b[B.i] AS y, A.c[B.i] AS z
FROM some_table A
LATERAL VIEW POSEXPLODE(A.a) B AS i, x
```
> [Hive Explode / Lateral View multiple arrays - Stack Overflow](https://stackoverflow.com/questions/20667473/hive-explode-lateral-view-multiple-arrays)


## 常用函数
> [LanguageManual UDF - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF)

### 字符串
```sql
-- 编辑距离: levenshtein(string A, string B) -> int
SELECT levenshtein('kitten', 'sitting');  -- 3
```

### 数学
```sql
-- 最小值: least(T v1, T v2, ...)
SELECT least(3, 1, -1);  -- -1

-- 最大值: greatest(T v1, T v2, ...)
SELECT least(3, 1, -1);  -- 3
```


## 常用设置
```sql
-- 使支持列位置别名
SET hive.groupby.orderby.position.alias=true;  -- Deprecated In: Hive 2.2.0
SET hive.groupby.position.alias=true;  -- Added In: Hive 2.2.0
SET hive.orderby.position.alias=true;  -- Added In: Hive 2.2.0
```

## TRANSFORM 脚本

### MAP/REDUCE 语法
> [Transform/Map-Reduce Syntax - Apache Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Transform)
- `MAP ...` 和 `REDUCE ...` 实际上就是 `SELECT TRANSFORM ( ... )` 的语法转换, 即以下两种写法是等价的;
    ```sql
    FROM (
        FROM pv_users A
        MAP A.userid, A.date
        USING 'python map_script.py'
        AS dt, uid
        CLUSTER BY dt
    ) M
    INSERT OVERWRITE TABLE pv_users_reduced
        REDUCE M.dt, M.uid
        USING 'python reduce_script.py'
        AS dt, count
    ;

    -- 等价于
    FROM (
        FROM pv_users A
        SELECT TRANSFORM(A.userid, A.date)
        USING 'python map_script.py'
        AS dt, uid
        CLUSTER BY dt
    ) M
    INSERT OVERWRITE TABLE pv_users_reduced
        SELECT TRANSFORM(M.dt, M.uid)
        USING 'python reduce_script.py'
        AS dt, count
    ;
    ```

- 示例: [Hive-Transform-Python: 快捷的 Map/Reduce - 简书](https://www.jianshu.com/p/8a7b3cf4cac5/)


## 其他

### VTL 变量
> [Velocity模板语言(VTL): 介绍 | Velocity 中文文档](https://wizardforcel.gitbooks.io/velocity-doc/content/5.html)

```sql
-- 注意不是 hive 变量
#set( $day_delta = 60 );
#set( $tmp_table = 'tmp_table' );

SELECT *
FROM $tmp_table A
WHERE A.dt > DATA_SUB('2022-12-12', $day_delta)
```