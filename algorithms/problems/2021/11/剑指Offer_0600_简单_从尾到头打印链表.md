## 从尾到头打印链表
<!--START_SECTION:badge-->

![last modify](https://img.shields.io/static/v1?label=last%20modify&message=2022-10-14%2000%3A39%3A24&color=yellowgreen&style=flat-square)
![source](https://img.shields.io/static/v1?label=source&message=%E5%89%91%E6%8C%87Offer&color=green&style=flat-square)
![level](https://img.shields.io/static/v1?label=level&message=%E7%AE%80%E5%8D%95&color=yellow&style=flat-square)
![tags](https://img.shields.io/static/v1?label=tags&message=%E9%93%BE%E8%A1%A8%2C%20%E6%A0%88%2C%20DFS%2C%20%E9%80%92%E5%BD%92&color=orange&style=flat-square)

<!--END_SECTION:badge-->
<!--info
tags: [链表, 栈, DFS, 递归]
source: 剑指Offer
level: 简单
number: '0600'
name: 从尾到头打印链表
companies: []
-->

<summary><b>问题简述</b></summary>

```txt
从尾到头打印链表（用数组返回）
```

<details><summary><b>详细描述</b></summary>

```txt
输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

示例 1：
    输入：head = [1,3,2]
    输出：[2,3,1]

限制：
    0 <= 链表长度 <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

</details>


<summary><b>思路</b></summary>

- 法1）利用栈，顺序入栈，然后依次出栈即可
- 法2）利用深度优先遍历思想（二叉树的先序遍历）


<details><summary><b>Python：栈</b></summary>

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        
        # ret = []
        # for _ in range(len(stack)):  # 相当于逆序遍历
        #     ret.append(stack.pop())
        # return ret
        return stack[::-1]  # 与以上代码等价
```

</details>

<details><summary><b>Python：DFS、递归</b></summary>

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        if head is None:
            return []

        ret = self.reversePrint(head.next)
        ret.append(head.val)

        return ret
```

</details>

