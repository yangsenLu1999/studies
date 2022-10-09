## LeetCode_0020_有效的括号（简单, 2022-03）
<!--info
tags: [栈, lc100]
source: LeetCode
level: 简单
number: '0020'
name: 有效的括号
companies: []
-->

<summary><b>问题简述</b></summary>

```txt
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
有效字符串需满足：
    左括号必须用相同类型的右括号闭合。
    左括号必须以正确的顺序闭合。
```
> [20. 有效的括号 - 力扣（LeetCode）](https://leetcode-cn.com/problems/valid-parentheses/)

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```

</details>
-->


<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路</b></summary>

- 利用栈，遇到左括号就压栈，遇到右括号就出栈；
- 无效的情况：栈顶与当前遇到的右括号不匹配，或栈为空；
- 当遍历完所有字符，且栈为空时，即有效；

<details><summary><b>Python</b></summary>

```python
class Solution:
    def isValid(self, s: str) -> bool:

        stack = []  # 模拟栈
        table = {')':'(', ']': '[', '}': '{'}

        for c in s:
            if c in '([{':
                stack.append(c)
            elif stack and table[c] == stack[-1]:
                stack.pop()
            else:  # 栈为空，且遇到左括号，一定无效
                return False
        
        return len(stack) == 0
```

</details>

