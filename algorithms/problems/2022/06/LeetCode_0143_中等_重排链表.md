## 重排链表
<!--START_SECTION:badge-->

![last modify](https://img.shields.io/static/v1?label=last%20modify&message=2022-10-16%2017%3A41%3A53&color=yellowgreen&style=flat-square)
[![](https://img.shields.io/static/v1?label=&message=%E4%B8%AD%E7%AD%89&color=yellow&style=flat-square)](../../../README.md#中等)
[![](https://img.shields.io/static/v1?label=&message=LeetCode&color=green&style=flat-square)](../../../README.md#leetcode)
[![](https://img.shields.io/static/v1?label=&message=%E9%93%BE%E8%A1%A8&color=blue&style=flat-square)](../../../README.md#链表)
[![](https://img.shields.io/static/v1?label=&message=%E6%A8%A1%E6%8B%9F&color=blue&style=flat-square)](../../../README.md#模拟)

<!--END_SECTION:badge-->
<!--info
tags: [链表, 模拟]
source: LeetCode
level: 中等
number: '0143'
name: 重排链表
companies: [字节, 度小满]
-->

<summary><b>问题简述</b></summary>

```txt
给定一个单链表 L 的头节点 head ，单链表 L 表示为：
    L0 → L1 → … → Ln - 1 → Ln
请将其重新排列后变为：
    L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → …
不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
```

<details><summary><b>详细描述</b></summary>

```txt
给定一个单链表 L 的头节点 head ，单链表 L 表示为：
    L0 → L1 → … → Ln - 1 → Ln
请将其重新排列后变为：
    L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

示例 1：
    输入：head = [1,2,3,4]
    输出：[1,4,2,3]
示例 2：
    输入：head = [1,2,3,4,5]
    输出：[1,5,2,4,3]

提示：
    链表的长度范围为 [1, 5 * 104]
    1 <= node.val <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/reorder-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

</details>

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路：模拟</b></summary>

1. 先找到中间节点 mid；
2. 将链表 mid 反转；
3. 然后合并 head 和 mid；

<details><summary><b>Python：写法1</b></summary>

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
    
        def  get_mid(p):
            lp, fp = p, p

            while fp and fp.next:
                lp = lp.next
                fp = fp.next.next
            
            return lp
        
        def reverse(p):
            cur, pre = p, None
            while cur:
                nxt = cur.next
                cur.next = pre
                pre = cur
                cur = nxt
            
            return pre
        
        mid = get_mid(head)  # 注意此时还没有断开两个链表
        mid = reverse(mid)

        # merge
        l, r = head, mid
        while True:
            l_nxt, r_nxt = l.next, r.next
            if not r_nxt:  # 这是一种写法，另一种写法是在获得 mid 后将 mid 与原链表断开（后移一个节点，结果也是正确的，见写法2）
                break
            l.next, r.next = r, l_nxt
            l, r = l_nxt, r_nxt
```

</details>


<details><summary><b>Python：写法2</b></summary>

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
    
        def  get_mid(p):
            lp, fp = p, p

            while fp and fp.next:
                lp = lp.next
                fp = fp.next.next
            
            return lp
        
        def reverse(p):
            cur, pre = p, None
            while cur:
                nxt = cur.next
                cur.next = pre
                pre = cur
                cur = nxt
            
            return pre
        
        mid = get_mid(head)
        mid.next, mid = None, mid.next  # 写法2）提前断开 mid
        # mid, mid.next = mid.next, None  # err
        mid = reverse(mid)

        # merge
        l, r = head, mid
        while r:
            l_nxt, r_nxt = l.next, r.next
            # if not r_nxt:
            #     break
            l.next, r.next = r, l_nxt
            l, r = l_nxt, r_nxt
```

</details>

