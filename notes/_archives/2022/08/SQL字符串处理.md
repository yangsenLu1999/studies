SQL 字符串处理
===

- [参考文档](#参考文档)
- [常规](#常规)
    - [小写](#小写)
    - [移除首尾空白符](#移除首尾空白符)
    - [拼接](#拼接)
- [正则](#正则)
    - [正则替换](#正则替换)
    - [正则切分](#正则切分)

## 参考文档
- [Presto Documentation](https://prestodb.io/docs/current/)

## 常规

### 小写
`lower(string)`
- 支持的引擎：
    - [Presto](https://prestodb.io/docs/current/functions/string.html#lower)
- 对应的 Python 操作：`str.lower()`

### 移除首尾空白符
`trim(string)`
- 支持的引擎：
    - [Presto](https://prestodb.io/docs/current/functions/string.html#trim)
- 对应的 Python 操作：`str.strip()`

### 拼接
`array_join(array, sep, null_replacement)`
- 支持的引擎：
    - [Presto](https://prestodb.io/docs/current/functions/array.html#array_join)
        ```sql
        SELECT array_join(ARRAY[1, NULL, 2], ','); -- 1,2
        SELECT array_join(ARRAY[1, NULL, 2], '_', 'N'); -- 1_N_2
        SELECT array_join(
                    regexp_split('a(b), c', '[^a-z]'), 
                    ' ');  -- 'a b c'
        ```
- 对应的 Python 操作：`sep.join(array)`

## 正则

### 正则替换
`regexp_replace(string, pattern, replacement)`
- 支持的引擎：
    - [Presto](https://prestodb.io/docs/current/functions/regexp.html#regexp_replace)
        ```sql
        SELECT regexp_replace('a   b  c', '\s+', ' ');  -- 'a b c'
        ```

### 正则切分
`regexp_split(string, pattern)`
- 支持的引擎：
    - [Presto](https://prestodb.io/docs/current/functions/regexp.html#regexp_split)
        ```sql
        SELECT regexp_split('a(b), c', '[^a-z]');  -- '[a, b, , , c]'
        SELECT array_join(
                    regexp_split('a(b), c', '[^a-z]'), 
                    ' ');  -- 'a b c'
        ``` 