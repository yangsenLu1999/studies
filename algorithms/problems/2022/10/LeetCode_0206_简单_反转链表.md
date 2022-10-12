## LeetCode_0206_反转链表（简单, 2022-10）
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
