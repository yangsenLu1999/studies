## LeetCode_0025_K个一组翻转链表（困难, 2022-10）
<!--info
tags: [链表, 热门]
source: LeetCode
level: 困难
number: '0025'
name: K个一组翻转链表
companies: []
-->

> [25. K 个一组翻转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

<summary><b>问题简述</b></summary>

```txt
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
```

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```

</details>
-->

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路</b></summary>

- 找到关键的 4 个节点, 待反转子链表的头节点 `h`, `h` 的前一个节点, 尾节点 `t`, `t` 的下一个节点; 

<details><summary><b>Python</b></summary>

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:

    def reverse(self, head):
        pre, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre, head

    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:

        dummy = cur = ListNode(next=head)  # 设置伪头节点
        while cur:
            pre = cur  # 子链表头节点的前一个节点
            for _ in range(k):
                if not cur.next:
                    return dummy.next
                cur = cur.next
            sub_head = cur.next  # 子链表尾节点的下一个节点
            cur.next = None  # 断开子链表
            pre.next, tail = self.reverse(pre.next)  # 反转子链表, 返回反转后的头尾节点
            tail.next = sub_head
            cur = tail

        return dummy.next
```

</details>


<!-- 
<summary><b>相关问题</b></summary>

-->
