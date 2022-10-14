## 合并两个排序的链表
<!--START_SECTION:badge-->

![2022-10-14 14:59:33](https://img.shields.io/static/v1?label=last%20modify&message=2022-10-14%2014%3A59%3A33&color=yellowgreen&style=flat-square)
![剑指Offer](https://img.shields.io/static/v1?label=source&message=%E5%89%91%E6%8C%87Offer&color=green&style=flat-square)
![简单](https://img.shields.io/static/v1?label=level&message=%E7%AE%80%E5%8D%95&color=yellow&style=flat-square)
![链表, 递归, 动态规划](https://img.shields.io/static/v1?label=tags&message=%E9%93%BE%E8%A1%A8%2C%20%E9%80%92%E5%BD%92%2C%20%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92&color=orange&style=flat-square)

<!--END_SECTION:badge-->
<!--info
tags: [链表, 递归, 迭代]
source: 剑指Offer
level: 简单
number: '2500'
name: 合并两个排序的链表
companies: []
-->

<summary><b>问题简述</b></summary>

```txt
合并两个有序链表，且合并后依然有序；
```

<details><summary><b>详细描述</b></summary>

```txt
输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

示例1：
    输入：1->2->4, 1->3->4
    输出：1->1->2->3->4->4

限制：
    0 <= 链表长度 <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路1：递归</b></summary>

- 递归公式：`merge(l1, l2) = li + merge(li.next, lj)`，  
  其中当 `l1<l2` 时 `i,j = 1,2`，否则 `i,j=2,1`

<details><summary><b>Python</b></summary>

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        def dfs(p1, p2):
            if not p1: return p2
            if not p2: return p1

            if p1.val < p2.val:
                p1.next = dfs(p1.next, p2)
                return p1
            else:
                p2.next = dfs(p1, p2.next)
                return p2

        return dfs(l1, l2)
```

</details>

<summary><b>思路2：迭代</b></summary>

> [合并两个排序的链表（伪头节点，清晰图解）](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/solution/mian-shi-ti-25-he-bing-liang-ge-pai-xu-de-lian-b-2/)

<details><summary><b>Python：伪头结点（推荐）</b></summary>

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        ret = cur = ListNode(0)

        while l1 and l2:
            if l1.val < l2.val:
                cur.next, l1 = l1, l1.next
            else:
                cur.next, l2 = l2, l2.next
            
            cur = cur.next  # 这一步容易忽略
        
        cur.next = l1 if l1 else l2
        return ret.next
```

</details>

<details><summary><b>Python：不使用伪头结点</b></summary>

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1: return l2
        if not l2: return l1

        cur = ret = l1 if l1.val < l2.val else l2  # 
        
        while l1 and l2:
            if l1.val < l2.val:  # 这两处的判断条件要一致，否则会出错
                cur.next, l1 = l1, l1.next
            else:
                cur.next, l2 = l2, l2.next
            cur = cur.next
        
        cur.next = l1 if l1 else l2
        return ret
```

</details>
