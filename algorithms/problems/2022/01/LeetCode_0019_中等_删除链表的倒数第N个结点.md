## LeetCode_0019_删除链表的倒数第N个结点（中等, 2022-01）
<!--info
tags: [链表, 快慢指针, lc100]
source: LeetCode
level: 中等
number: 0019
name: 删除链表的倒数第N个结点
companies: []
-->

<summary><b>问题简述</b></summary>

```txt
给定链表，删除链表的倒数第 n 个结点，返回删除后链表的头结点。
```
> [19. 删除链表的倒数第 N 个结点 - 力扣（LeetCode）](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

<details><summary><b>详细描述</b></summary>

```txt
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

示例 1：
    输入：head = [1,2,3,4,5], n = 2
    输出：[1,2,3,5]
示例 2：
    输入：head = [1], n = 1
    输出：[]
示例 3：
    输入：head = [1,2], n = 1
    输出：[1]

提示：
    链表中结点的数目为 sz
    1 <= sz <= 30
    0 <= Node.val <= 100
    1 <= n <= sz

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路</b></summary>

<details><summary><b>Python</b></summary>

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:

        dummy = ListNode(next=head)

        fast, slow = dummy, dummy
        # 快指针先走 n+1 步（包括新加入的伪头节点）
        for _ in range(n + 1):
            fast = fast.next
        
        while fast:
            fast = fast.next
            slow = slow.next
        
        # 删除节点
        slow.next = slow.next.next
        return dummy.next
```

</details>
