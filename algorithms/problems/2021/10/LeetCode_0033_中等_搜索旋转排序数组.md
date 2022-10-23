## 搜索旋转排序数组
<!--START_SECTION:badge-->

![last modify](https://img.shields.io/static/v1?label=last%20modify&message=2022-10-14%2014%3A59%3A33&color=yellowgreen&style=flat-square)
[![](https://img.shields.io/static/v1?label=&message=%E4%B8%AD%E7%AD%89&color=yellow&style=flat-square)](../../../README.md#中等)
[![](https://img.shields.io/static/v1?label=&message=LeetCode&color=green&style=flat-square)](../../../README.md#leetcode)
[![](https://img.shields.io/static/v1?label=&message=%E4%BA%8C%E5%88%86%E6%9F%A5%E6%89%BE&color=blue&style=flat-square)](../../../README.md#二分查找)
[![](https://img.shields.io/static/v1?label=&message=LeetCode%20Hot%20100&color=blue&style=flat-square)](../../../README.md#leetcode-hot-100)

<!--END_SECTION:badge-->
<!--info
tags: [二分查找, lc100]
source: LeetCode
level: 中等
number: '0033'
name: 搜索旋转排序数组
companies: []
-->

> [33. 搜索旋转排序数组 - 力扣（LeetCode）](https://leetcode.cn/problems/search-in-rotated-sorted-array)

<summary><b>问题简述</b></summary>

```txt
在一个旋转过的有序数组中搜索某值，若存在返回下标，否则返回 -1。
```


<details><summary><b>详细描述</b></summary>

```txt
整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

示例 1：
    输入：nums = [4,5,6,7,0,1,2], target = 0
    输出：4
示例 2：
    输入：nums = [4,5,6,7,0,1,2], target = 3
    输出：-1
示例 3：
    输入：nums = [1], target = 0
    输出：-1
 

提示：
    1 <= nums.length <= 5000
    -10^4 <= nums[i] <= 10^4
    nums 中的每个值都 独一无二
    题目数据保证 nums 在预先未知的某个下标上进行了旋转
    -10^4 <= target <= 10^4
 
进阶：你可以设计一个时间复杂度为 O(log n) 的解决方案吗？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/search-in-rotated-sorted-array
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>


<summary><b>思路</b></summary>

- “二分”的本质是两段性，而不是单调性；即只要二分后，左边满足某个性质，右边不满足某个性质，即可使用二分；
    > [LogicStack-LeetCode/33.搜索旋转排序数组（中等）](https://github.com/SharingSource/LogicStack-LeetCode/blob/main/LeetCode/31-40/33.%20搜索旋转排序数组（中等）.md#二分解法)
- 本题中，将数组从中间分开后，其中一个部分一定是有序的，有序部分可以通过比较 `a[m]` 和 `a[0]` 得到；
- 此时如果 target 在有序部分，那么可以排除无序的一半，否则可以排除有序的一半；
- 注意本题有很多编码细节，详见代码注释；

<details><summary><b>Python 写法 1：闭区间</b></summary>

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:

        l, r = 0, len(nums) - 1  # [l, r] 闭区间
        while l <= r:
            # 注意如果这里使用 l < r，推出循环时 l == r，返回时要判断 nums[l] 是否等于 target
            # 而使用 l <= r，那么当 l == r 时，会继续执行依次判断流程，此时 m == l == r
            m = l + (r - l) // 2

            if nums[m] == target: return m
            
            # 以下 nums[m] != target
            if nums[0] <= nums[m]:  # [l, m] 是有序的；注意，这里必须使用 <=，考虑 m == 0 的情况
                # 判断 target 是否在有序部分
                if nums[l] <= target < nums[m]:  # 因为不能确定 nums[l] 是否等于 target，所以要用 <=
                    r = m - 1
                else:
                    l = m + 1
            else:  # (m, r] 是有序的
                # 同理，判断 target 是否在有序部分
                if nums[m] < target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1

        return -1
        # 如果使用 while l < r，就要如下返回
        # return -1 if nums[l] != target else l
```

</details>


<details><summary><b>Python 写法 2：左闭右开区间（推荐）</b></summary>

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:

        l, r = 0, len(nums)  # [l, r) 左闭右开区间
        while l < r:  # 这里不需要 l <= r，因为是半开区间，退出循环时 l == r 就表示区间内无元素
            m = l + (r - l) // 2

            if nums[m] == target: return m
            
            # 以下 nums[m] != target
            if nums[0] < nums[m]:  # [l, m) 是有序的，这里用 < 或 <= 不影响
                # 判断 target 是否在有序部分
                if nums[l] <= target < nums[m]:  # 因为不能确定 nums[l] 是否等于 target，所以要用 <=
                    r = m  # 右边界
                else:
                    l = m + 1
            else:  # (m, r] 是有序的
                # 同理，判断 target 是否在有序部分
                if nums[m] < target <= nums[r - 1]:
                    l = m + 1
                else:
                    r = m  # 右边界

        return -1
```

</details>
