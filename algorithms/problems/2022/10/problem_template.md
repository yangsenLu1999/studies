<!--info
tags: [hash, lc100]
source: LeetCode
level: 中等
number: '0049'
name: 字母异位词分组
companies: []
-->

> [49. 字母异位词分组 - 力扣（LeetCode）](https://leetcode.cn/problems/group-anagrams/)

<summary><b>问题简述</b></summary>

```txt
给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
字母异位词 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。
```

<!-- 
<details><summary><b>详细描述</b></summary>

```txt
```

</details>
-->

<!-- <div align="center"><img src="../../../_assets/xxx.png" height="300" /></div> -->

<summary><b>思路</b></summary>

- 设计一个方法将字母异位词转化为相同的 key, 然后使用字典存储;

<details><summary><b>Python: 写法 1</b></summary>

- 将 s 排序后保存为 key;

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        from collections import defaultdict
        
        def to_key(s):
            return tuple(sorted(s))  # ''.join(sorted(s))

        ret = defaultdict(list)
        for s in strs:
            ret[to_key(s)].append(s)
        
        return list(ret.values())
```

</details>

<details><summary><b>Python: 写法 2</b></summary>

- 记录每个字符出现的次数; 

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        from collections import defaultdict
        
        def to_key(s):
            cnt = [0] * 26
            for c in s:
                cnt[ord(c) - ord('a')] += 1
            return tuple(cnt)

        ret = defaultdict(list)
        for s in strs:
            ret[to_key(s)].append(s)
        
        return list(ret.values())
```

</details>

<!-- 
<summary><b>相关问题</b></summary>

-->
