## <title - autoUpdate>
<!--START_SECTION:badge-->
<!--END_SECTION:badge-->
<!--info
tags: [二分查找, 热门]
source: LeetCode
level: 简单
number: '0069'
name: x 的平方根
companies: []
-->

> [69. x 的平方根 - 力扣（LeetCode）](https://leetcode.cn/problems/sqrtx/)

<summary><b>问题简述</b></summary>

```txt
给你一个非负整数 x ，计算并返回 x 的 算术平方根 (整数部分)。
```

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```

</details>
-->

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路: 二分查找</b></summary>

<details><summary><b>Python</b></summary>

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x in (0, 1): return x

        l, r = 0, x
        while l < r:
            m = (l + r) // 2
            if m ** 2 <= x < (m + 1) ** 2:
                break
            
            if m ** 2 < x:
                l = m
            else:
                r = m
        
        return m
```

</details>


<summary><b>进阶: 浮点数版本</b></summary>

- 定义 `mySqrt(x: float, e: int)`, 其中 `e` 为小数精度;

<details><summary><b>Python</b></summary>

```python
class Solution:
    def mySqrt(self, x: float, e: int) -> int:
        if x in (0, 1): return x
        
        l, r = 0, x
        while l < r:
            m = (l + r) / 2
            if abs(m ** 2 - x) <= 0.1 ** e:
                break
            
            if m ** 2 < x:
                l = m
            else:
                r = m
        
        return m
```

</details>

<!-- 
<summary><b>相关问题</b></summary>

-->
