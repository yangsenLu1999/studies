## <title - autoUpdate>
<!--START_SECTION:badge-->
<!--END_SECTION:badge-->
<!--info
tags: [堆, 热门]
source: LeetCode
level: 困难
number: '0239'
name: 滑动窗口最大值
companies: [Soul]
-->

> [239. 滑动窗口最大值 - 力扣（LeetCode）](https://leetcode.cn/problems/sliding-window-maximum/)

<summary><b>问题简述</b></summary>

```txt
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。
你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
返回 滑动窗口中的最大值。
```

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```

</details>
-->

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路 1: 堆/优先队列</b></summary>

- 维护一个最大堆保存窗口内的值;
- 难点是如何保证堆内 (主要是堆顶) 的值正好在窗口内;
    - 方法是同时保存值的索引, 利用索引判断当前堆顶值是否在窗口内, 详见代码;

<details><summary><b>Python</b></summary>

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:

        import heapq

        h = []
        for i in range(k):
            heapq.heappush(h, (-nums[i], i))
        
        ret = [-h[0][0]]
        for i in range(k, len(nums)):
            while h and h[0][1] <= i - k:
                heapq.heappop(h)
            heapq.heappush(h, (-nums[i], i))
            ret.append(-h[0][0])
        
        return ret
```

</details>


<summary><b>思路 2: 单调队列</b></summary>

> [滑动窗口最大值 (方法二) - 力扣官方题解](https://leetcode.cn/problems/sliding-window-maximum/solution/hua-dong-chuang-kou-zui-da-zhi-by-leetco-ki6m/)

<!--
<summary><b>相关问题</b></summary>

-->
