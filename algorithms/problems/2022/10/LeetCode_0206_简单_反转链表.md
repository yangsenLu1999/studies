## 反转链表
<!--START_SECTION:badge-->

![last modify](https://img.shields.io/static/v1?label=last%20modify&message=2022-10-16%2017%3A41%3A53&color=yellowgreen&style=flat-square)
[![](https://img.shields.io/static/v1?label=&message=%E7%AE%80%E5%8D%95&color=yellow&style=flat-square)](../../../README.md#简单)
[![](https://img.shields.io/static/v1?label=&message=LeetCode&color=green&style=flat-square)](../../../README.md#leetcode)
[![](https://img.shields.io/static/v1?label=&message=%E9%93%BE%E8%A1%A8&color=blue&style=flat-square)](../../../README.md#链表)
[![](https://img.shields.io/static/v1?label=&message=%E7%83%AD%E9%97%A8%26%E7%BB%8F%E5%85%B8%26%E6%98%93%E9%94%99&color=blue&style=flat-square)](../../../README.md#热门经典易错)

<!--END_SECTION:badge-->
<!--info
tags: [链表, 经典]
source: LeetCode
level: 简单
number: '0206'
name: 反转链表
companies: []
-->

> [206. 反转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-linked-list/)

<summary><b>问题简述</b></summary>

```txt
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
```

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```

</details>
-->

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路</b></summary>

- 定义 `pre`, `cur`, `nxt` 三个指针, 详见代码;

<details><summary><b>Python</b></summary>

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        pre, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt

        return pre
```

</details>


<!-- 
<summary><b>相关问题</b></summary>

-->
