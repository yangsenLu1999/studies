Query 扩展 (电商领域)
===
<!--START_SECTION:badge-->

![last modify](https://img.shields.io/static/v1?label=last%20modify&message=2022-12-12%2023%3A55%3A28&color=yellowgreen&style=flat-square)

<!--END_SECTION:badge-->
<!--info
top: false
hidden: false
-->

<!-- TOC -->
- [关键词](#关键词)
- [概述](#概述)
- [同义词挖掘](#同义词挖掘)
- [参考阅读](#参考阅读)
<!-- TOC -->

## 关键词
> query expansion (查询扩展), query rewriting (查询重写/改写)


## 概述

**查询扩展**的目的:
- 弥合 query 和 doc 之间的词汇差距 (vocabulary gap), 比如拼写错误 (miss-spelling), 描述同义实体的不同方式 (different ways of describing the same entity);
- 在电子商务 (e-commerce) 领域, 用户 query 通常更短且口语化, 而产品标题通常更冗长且包含正式术语;
    - 示例: query = "noise absorbing blankets" (吸音地毯), rewriting = "acoustic blankets" (声学地毯), "soundproof blankets" (隔音毯), "soundproof blanket"


Query 扩展的一般过程:
1. 离线阶段, token 维度生成同义词;
2. 在线阶段, 对 query 进行改写; 


## 同义词挖掘
- [同义词挖掘 (电商)](同义词挖掘.md)


## 参考阅读
- [[2103.00800] Query Rewriting via Cycle-Consistent Translation for E-Commerce Search](https://arxiv.org/abs/2103.00800)