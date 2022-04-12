## ❗️master 分支暂停更新，内容将陆续迁移到 gitbook 分支。

studies
---

![python-version](https://img.shields.io/badge/python-3.8+-green)
![pytorch-version](https://img.shields.io/badge/pytorch-1.8+-green)
![tf-version](https://img.shields.io/badge/tensorflow-2.3+-green)
![repo-size](https://img.shields.io/github/repo-size/imhuay/studies)
<!-- ![total-lines](https://img.shields.io/tokei/lines/github/imhuay/studies) -->
<!-- ![code-size](https://img.shields.io/github/languages/code-size/imhuay/studies) -->

<!-- ![followers](https://img.shields.io/github/followers/imhuay?style=social) -->
<!-- ![user-stars](https://img.shields.io/github/stars/imhuay?style=social) -->


<details><summary><b> TODO </b></summary>

- [ ] 尝试 GitHub 提供的 projects 栏：参考 [Projects · zhaoyan346a/Blog](https://github.com/zhaoyan346a/Blog/projects)
- [ ] 重构 README 生成的 Algorithms 和 Codes 两个类，并迁移至 tools 目录。
- [ ] 优化主页 README 下的 Algorithms 链接，调整为层级目录的形式（类似 Notes）

<!-- - [ ] 【`2021.11.11`】pytorch_trainer: 为 EvaluateCallback 添加各种预定义评估指标，如 acc、f1 等，目前只有 loss； -->
<!-- - [ ] 【`2021.11.11`】论文：What does BERT learn about the structure of language? —— Bert 各层的含义； -->
<!-- - [ ] 【`2021.11.10`】bert-tokenizer 自动识别 `[MASK]` 等特殊标识； -->
<!-- - [ ] 【`2021.11.07`】面试笔记：通识问题/项目问题 -->
<!-- - [ ] 【`2021.10.22`】max_batch_size 估算 -->

</details>

<details><summary><b> Done </b></summary>

- [x] 【`2022.01.18`】优化 algorithm 笔记模板的 tag 部分，使用 json 代替目前的正则抽取。
- [x] 【`2022.01.17`】自动生成目录结构（books、papers 等）
- [x] 【`2021.11.12`】优化 auto-readme，使用上一次的 commit info，而不是默认 'Auto-README'
    - 参考：`git commit -m "$(git log -"$(git rev-list origin/master..master --count)" --pretty=%B | cat)"`
    - 说明：使用 origin/master 到 master 之间所有的 commit 信息作为这次的 message；
- [x] 【`2021.11.11`】bert 支持加载指定层 -> `_test_load_appointed_layers()`
- [x] 【`2021.11.08`】把 __test.py 文件自动加入文档测试（放弃）
    - 有些测试比较耗时，不需要全部加入自动测试；
    - __test.py 针对的是存在相对引用的模块，如果这些模块有改动，会即时测试，所以也不需要自动测试
- [x] 【`2021.11.03`】[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) 代码阅读

</details>

<!-- 

### 其他仓库
- [Algorithm_Interview_Notes-Chinese](https://github.com/imhuay/Algorithm_Interview_Notes-Chinese_backups): 在校期间的学习/面试笔记；
- [bert_by_keras](https://github.com/imhuay/bert_by_keras): 使用 keras 重构的 Bert；
- [algorithm](https://github.com/imhuay/algorithm): 刷题笔记，实际上就是本仓库 algorithm 目录下的内容；

 -->

---

Repo Index
---
- [Algorithms](#algorithms)
- [Notes](#notes)
- [Papers](#papers)
- [Books](#books)
- [Codes](#codes)

---

Algorithms
---
<details><summary><b> 合集-LeetCode [61] <a href="algorithms/topics/合集-LeetCode.md">¶</a></b></summary>

- [`LeetCode 0001 两数之和 (简单, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0001-两数之和-简单-2021-10)
- [`LeetCode 0002 两数相加 (中等, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0002-两数相加-中等-2021-10)
- [`LeetCode 0003 无重复字符的最长子串 (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0003-无重复字符的最长子串-中等-2022-02)
- [`LeetCode 0004 寻找两个正序数组的中位数 (困难, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0004-寻找两个正序数组的中位数-困难-2022-02)
- [`LeetCode 0005 最长回文子串 (中等, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0005-最长回文子串-中等-2021-10)
- [`LeetCode 0010 正则表达式匹配 (困难, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0010-正则表达式匹配-困难-2022-01)
- [`LeetCode 0011 盛最多水的容器 (中等, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0011-盛最多水的容器-中等-2021-10)
- [`LeetCode 0015 三数之和 (中等, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0015-三数之和-中等-2021-10)
- [`LeetCode 0016 最接近的三数之和 (中等, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0016-最接近的三数之和-中等-2021-10)
- [`LeetCode 0019 删除链表的倒数第N个结点 (中等, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0019-删除链表的倒数第n个结点-中等-2022-01)
- [`LeetCode 0020 有效的括号 (简单, 2022-03)`](algorithms/topics/合集-LeetCode.md#leetcode-0020-有效的括号-简单-2022-03)
- [`LeetCode 0021 合并两个有序链表 (简单, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0021-合并两个有序链表-简单-2021-10)
- [`LeetCode 0025 K个一组翻转链表 (困难, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0025-k个一组翻转链表-困难-2022-02)
- [`LeetCode 0029 两数相除 (中等, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0029-两数相除-中等-2021-10)
- [`LeetCode 0033 搜索旋转排序数组 (中等, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0033-搜索旋转排序数组-中等-2021-10)
- [`LeetCode 0042 接雨水 (困难, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0042-接雨水-困难-2021-10)
- [`LeetCode 0053 最大子数组和 (简单, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0053-最大子数组和-简单-2022-01)
- [`LeetCode 0064 最小路径和 (中等, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0064-最小路径和-中等-2022-01)
- [`LeetCode 0070 爬楼梯 (简单, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0070-爬楼梯-简单-2022-01)
- [`LeetCode 0072 编辑距离 (困难, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0072-编辑距离-困难-2022-01)
- [`LeetCode 0086 分隔链表 (中等, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0086-分隔链表-中等-2021-10)
- [`LeetCode 0091 解码方法 (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0091-解码方法-中等-2022-02)
- [`LeetCode 0096 不同的二叉搜索树 (中等, 2022-03)`](algorithms/topics/合集-LeetCode.md#leetcode-0096-不同的二叉搜索树-中等-2022-03)
- [`LeetCode 0098 验证二叉搜索树 (中等, 2022-03)`](algorithms/topics/合集-LeetCode.md#leetcode-0098-验证二叉搜索树-中等-2022-03)
- [`LeetCode 0104 二叉树的最大深度 (简单, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0104-二叉树的最大深度-简单-2021-10)
- [`LeetCode 0110 平衡二叉树 (简单, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0110-平衡二叉树-简单-2022-02)
- [`LeetCode 0111 二叉树的最小深度 (简单, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0111-二叉树的最小深度-简单-2021-10)
- [`LeetCode 0112 路径总和 (简单, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0112-路径总和-简单-2022-02)
- [`LeetCode 0113 路径总和II (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0113-路径总和ii-中等-2022-02)
- [`LeetCode 0120 三角形最小路径和 (中等, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0120-三角形最小路径和-中等-2022-01)
- [`LeetCode 0121 买卖股票的最佳时机 (简单, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0121-买卖股票的最佳时机-简单-2022-01)
- [`LeetCode 0122 买卖股票的最佳时机II (中等, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0122-买卖股票的最佳时机ii-中等-2022-01)
- [`LeetCode 0123 买卖股票的最佳时机III (困难, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0123-买卖股票的最佳时机iii-困难-2022-01)
- [`LeetCode 0124 二叉树中的最大路径和 (困难, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0124-二叉树中的最大路径和-困难-2022-02)
- [`LeetCode 0129 求根节点到叶节点数字之和 (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0129-求根节点到叶节点数字之和-中等-2022-02)
- [`LeetCode 0143 重排链表 (中等, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0143-重排链表-中等-2022-01)
- [`LeetCode 0152 乘积最大子数组 (中等, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0152-乘积最大子数组-中等-2022-01)
- [`LeetCode 0167 两数之和2(输入有序数组) (简单, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0167-两数之和2输入有序数组-简单-2021-10)
- [`LeetCode 0187 重复的DNA序列 (中等, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0187-重复的dna序列-中等-2021-10)
- [`LeetCode 0198 打家劫舍 (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0198-打家劫舍-中等-2022-02)
- [`LeetCode 0213 打家劫舍II (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0213-打家劫舍ii-中等-2022-02)
- [`LeetCode 0240 搜索二维矩阵2 (中等, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0240-搜索二维矩阵2-中等-2021-10)
- [`LeetCode 0257 二叉树的所有路径 (简单, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0257-二叉树的所有路径-简单-2022-02)
- [`LeetCode 0279 完全平方数 (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0279-完全平方数-中等-2022-02)
- [`LeetCode 0300 最长递增子序列 (中等, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0300-最长递增子序列-中等-2022-01)
- [`LeetCode 0322 零钱兑换 (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0322-零钱兑换-中等-2022-02)
- [`LeetCode 0337 打家劫舍III (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0337-打家劫舍iii-中等-2022-02)
- [`LeetCode 0343 整数拆分 (中等, 2021-12)`](algorithms/topics/合集-LeetCode.md#leetcode-0343-整数拆分-中等-2021-12)
- [`LeetCode 0352 将数据流变为多个不相交区间 (困难, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0352-将数据流变为多个不相交区间-困难-2021-10)
- [`LeetCode 0434 字符串中的单词数 (简单, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0434-字符串中的单词数-简单-2021-10)
- [`LeetCode 0437 路径总和III (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0437-路径总和iii-中等-2022-02)
- [`LeetCode 0441 排列硬币 (简单, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0441-排列硬币-简单-2021-10)
- [`LeetCode 0474 一和零 (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0474-一和零-中等-2022-02)
- [`LeetCode 0496 下一个更大元素 (简单, 2021-11)`](algorithms/topics/合集-LeetCode.md#leetcode-0496-下一个更大元素-简单-2021-11)
- [`LeetCode 0518 零钱兑换II (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0518-零钱兑换ii-中等-2022-02)
- [`LeetCode 0611 有效三角形的个数 (中等, 2021-10)`](algorithms/topics/合集-LeetCode.md#leetcode-0611-有效三角形的个数-中等-2021-10)
- [`LeetCode 0859 亲密字符串 (简单, 2021-11)`](algorithms/topics/合集-LeetCode.md#leetcode-0859-亲密字符串-简单-2021-11)
- [`LeetCode 0876 链表的中间结点 (简单, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0876-链表的中间结点-简单-2022-01)
- [`LeetCode 0915 分割数组 (中等, 2022-01)`](algorithms/topics/合集-LeetCode.md#leetcode-0915-分割数组-中等-2022-01)
- [`LeetCode 0958 二叉树的完全性检验 (中等, 2022-03)`](algorithms/topics/合集-LeetCode.md#leetcode-0958-二叉树的完全性检验-中等-2022-03)
- [`LeetCode 0988 从叶结点开始的最小字符串 (中等, 2022-02)`](algorithms/topics/合集-LeetCode.md#leetcode-0988-从叶结点开始的最小字符串-中等-2022-02)

</details>

<details><summary><b> 合集-剑指Offer [75] <a href="algorithms/topics/合集-剑指Offer.md">¶</a></b></summary>

- [`剑指Offer 0300 数组中重复的数字 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-0300-数组中重复的数字-简单-2021-11)
- [`剑指Offer 0400 二维数组中的查找 (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-0400-二维数组中的查找-中等-2021-11)
- [`剑指Offer 0500 替换空格 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-0500-替换空格-简单-2021-11)
- [`剑指Offer 0600 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-0600-从尾到头打印链表-简单-2021-11)
- [`剑指Offer 0700 重建二叉树 (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-0700-重建二叉树-中等-2021-11)
- [`剑指Offer 0900 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-0900-用两个栈实现队列-简单-2021-11)
- [`剑指Offer 1001 斐波那契数列 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1001-斐波那契数列-简单-2021-11)
- [`剑指Offer 1002 跳台阶 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1002-跳台阶-简单-2021-11)
- [`剑指Offer 1100 旋转数组的最小数字 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1100-旋转数组的最小数字-简单-2021-11)
- [`剑指Offer 1200 矩阵中的路径 (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1200-矩阵中的路径-中等-2021-11)
- [`剑指Offer 1300 机器人的运动范围 (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1300-机器人的运动范围-中等-2021-11)
- [`剑指Offer 1401 剪绳子（整数拆分） (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1401-剪绳子整数拆分-中等-2021-11)
- [`剑指Offer 1402 剪绳子 (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1402-剪绳子-中等-2021-11)
- [`剑指Offer 1500 二进制中1的个数 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1500-二进制中1的个数-简单-2021-11)
- [`剑指Offer 1600 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1600-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer 1700 打印从1到最大的n位数（N叉树的遍历） (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1700-打印从1到最大的n位数n叉树的遍历-中等-2021-11)
- [`剑指Offer 1800 删除链表的节点 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1800-删除链表的节点-简单-2021-11)
- [`剑指Offer 1900 正则表达式匹配 (困难, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-1900-正则表达式匹配-困难-2021-11)
- [`剑指Offer 2000 表示数值的字符串 (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-2000-表示数值的字符串-中等-2021-11)
- [`剑指Offer 2100 调整数组顺序使奇数位于偶数前面 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-2100-调整数组顺序使奇数位于偶数前面-简单-2021-11)
- [`剑指Offer 2200 链表中倒数第k个节点 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-2200-链表中倒数第k个节点-简单-2021-11)
- [`剑指Offer 2400 反转链表 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-2400-反转链表-简单-2021-11)
- [`剑指Offer 2500 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-2500-合并两个排序的链表-简单-2021-11)
- [`剑指Offer 2600 树的子结构 (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-2600-树的子结构-中等-2021-11)
- [`剑指Offer 2700 二叉树的镜像 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-2700-二叉树的镜像-简单-2021-11)
- [`剑指Offer 2800 对称的二叉树 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-2800-对称的二叉树-简单-2021-11)
- [`剑指Offer 2900 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-2900-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer 3000 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3000-包含min函数的栈-简单-2021-11)
- [`剑指Offer 3100 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3100-栈的压入弹出序列-中等-2021-11)
- [`剑指Offer 3201 层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3201-层序遍历二叉树-简单-2021-11)
- [`剑指Offer 3202 层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3202-层序遍历二叉树-简单-2021-11)
- [`剑指Offer 3203 层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3203-层序遍历二叉树之字形遍历-简单-2021-11)
- [`剑指Offer 3300 二叉搜索树的后序遍历序列 (中等, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3300-二叉搜索树的后序遍历序列-中等-2021-12)
- [`剑指Offer 3400 二叉树中和为某一值的路径 (中等, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3400-二叉树中和为某一值的路径-中等-2021-12)
- [`剑指Offer 3500 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3500-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer 3600 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3600-二叉搜索树与双向链表-中等-2021-12)
- [`剑指Offer 3700 序列化二叉树 (困难, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3700-序列化二叉树-困难-2021-12)
- [`剑指Offer 3800 字符串的排列（全排列） (中等, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3800-字符串的排列全排列-中等-2021-12)
- [`剑指Offer 3900 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-3900-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer 4000 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-4000-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer 4100 数据流中的中位数 (困难, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-4100-数据流中的中位数-困难-2021-12)
- [`剑指Offer 4200 连续子数组的最大和 (简单, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-4200-连续子数组的最大和-简单-2021-12)
- [`剑指Offer 4300 1～n整数中1出现的次数 (困难, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-4300-1n整数中1出现的次数-困难-2021-12)
- [`剑指Offer 4400 数字序列中某一位的数字 (中等, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-4400-数字序列中某一位的数字-中等-2021-12)
- [`剑指Offer 4500 把数组排成最小的数 (中等, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-4500-把数组排成最小的数-中等-2021-12)
- [`剑指Offer 4600 斐波那契数列-3（把数字翻译成字符串） (中等, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-4600-斐波那契数列-3把数字翻译成字符串-中等-2021-12)
- [`剑指Offer 4700 礼物的最大价值 (中等, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-4700-礼物的最大价值-中等-2021-12)
- [`剑指Offer 4800 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-4800-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer 4900 丑数 (中等, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-4900-丑数-中等-2021-12)
- [`剑指Offer 5000 第一个只出现一次的字符 (简单, 2021-12)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5000-第一个只出现一次的字符-简单-2021-12)
- [`剑指Offer 5100 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5100-数组中的逆序对-困难-2022-01)
- [`剑指Offer 5200 两个链表的第一个公共节点 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5200-两个链表的第一个公共节点-简单-2022-01)
- [`剑指Offer 5301 求0～n-1中缺失的数字 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5301-求0n-1中缺失的数字-简单-2022-01)
- [`剑指Offer 5302 在排序数组中查找数字 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5302-在排序数组中查找数字-简单-2022-01)
- [`剑指Offer 5400 二叉搜索树的第k大节点 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5400-二叉搜索树的第k大节点-简单-2022-01)
- [`剑指Offer 5501 求二叉树的深度 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5501-求二叉树的深度-简单-2022-01)
- [`剑指Offer 5502 判断是否为平衡二叉树 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5502-判断是否为平衡二叉树-简单-2022-01)
- [`剑指Offer 5601 数组中数字出现的次数 (中等, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5601-数组中数字出现的次数-中等-2022-01)
- [`剑指Offer 5602 数组中数字出现的次数 (中等, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5602-数组中数字出现的次数-中等-2022-01)
- [`剑指Offer 5701 和为s的两个数字 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5701-和为s的两个数字-简单-2022-01)
- [`剑指Offer 5702 和为s的连续正数序列 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5702-和为s的连续正数序列-简单-2022-01)
- [`剑指Offer 5801 翻转单词顺序 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5801-翻转单词顺序-简单-2022-01)
- [`剑指Offer 5802 左旋转字符串 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5802-左旋转字符串-简单-2022-01)
- [`剑指Offer 5901 滑动窗口的最大值 (困难, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5901-滑动窗口的最大值-困难-2022-01)
- [`剑指Offer 5902 队列的最大值 (中等, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-5902-队列的最大值-中等-2022-01)
- [`剑指Offer 6000 n个骰子的点数 (中等, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-6000-n个骰子的点数-中等-2022-01)
- [`剑指Offer 6100 扑克牌中的顺子 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-6100-扑克牌中的顺子-简单-2022-01)
- [`剑指Offer 6200 圆圈中最后剩下的数字（约瑟夫环问题） (中等, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-6200-圆圈中最后剩下的数字约瑟夫环问题-中等-2022-01)
- [`剑指Offer 6300 买卖股票的最佳时机 (中等, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-6300-买卖股票的最佳时机-中等-2022-01)
- [`剑指Offer 6400 求1~n的和 (中等, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-6400-求1n的和-中等-2022-01)
- [`剑指Offer 6500 不用加减乘除做加法 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-6500-不用加减乘除做加法-简单-2022-01)
- [`剑指Offer 6600 构建乘积数组 (中等, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-6600-构建乘积数组-中等-2022-01)
- [`剑指Offer 6700 把字符串转换成整数（atoi） (中等, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-6700-把字符串转换成整数atoi-中等-2022-01)
- [`剑指Offer 6801 二叉搜索树的最近公共祖先 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-6801-二叉搜索树的最近公共祖先-简单-2022-01)
- [`剑指Offer 6802 二叉树的最近公共祖先 (简单, 2022-01)`](algorithms/topics/合集-剑指Offer.md#剑指offer-6802-二叉树的最近公共祖先-简单-2022-01)

</details>

<details><summary><b> 合集-剑指Offer2 [3] <a href="algorithms/topics/合集-剑指Offer2.md">¶</a></b></summary>

- [`剑指Offer2 001 整数除法 (中等, 2022-02)`](algorithms/topics/合集-剑指Offer2.md#剑指offer2-001-整数除法-中等-2022-02)
- [`剑指Offer2 069 山峰数组的顶部 (简单, 2022-02)`](algorithms/topics/合集-剑指Offer2.md#剑指offer2-069-山峰数组的顶部-简单-2022-02)
- [`剑指Offer2 076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/合集-剑指Offer2.md#剑指offer2-076-数组中的第k大的数字-中等-2022-02)

</details>

<details><summary><b> 合集-热门&经典&易错 [31] <a href="algorithms/topics/合集-热门&经典&易错.md">¶</a></b></summary>

- [`LeetCode 0072 编辑距离 (困难, 2022-01)`](algorithms/topics/合集-热门&经典&易错.md#leetcode-0072-编辑距离-困难-2022-01)
- [`LeetCode 0300 最长递增子序列 (中等, 2022-01)`](algorithms/topics/合集-热门&经典&易错.md#leetcode-0300-最长递增子序列-中等-2022-01)
- [`LeetCode 0958 二叉树的完全性检验 (中等, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#leetcode-0958-二叉树的完全性检验-中等-2022-03)
- [`剑指Offer 0700 重建二叉树 (中等, 2021-11)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-0700-重建二叉树-中等-2021-11)
- [`剑指Offer 1600 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-1600-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer 2400 反转链表 (简单, 2021-11)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-2400-反转链表-简单-2021-11)
- [`剑指Offer 2900 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-2900-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer 3100 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-3100-栈的压入弹出序列-中等-2021-11)
- [`剑指Offer 3500 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-3500-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer 3600 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-3600-二叉搜索树与双向链表-中等-2021-12)
- [`剑指Offer 3800 字符串的排列（全排列） (中等, 2021-12)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-3800-字符串的排列全排列-中等-2021-12)
- [`剑指Offer 3900 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-3900-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer 4000 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-4000-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer 4900 丑数 (中等, 2021-12)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-4900-丑数-中等-2021-12)
- [`剑指Offer 5100 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-5100-数组中的逆序对-困难-2022-01)
- [`剑指Offer 6200 圆圈中最后剩下的数字（约瑟夫环问题） (中等, 2022-01)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-6200-圆圈中最后剩下的数字约瑟夫环问题-中等-2022-01)
- [`剑指Offer 6700 把字符串转换成整数（atoi） (中等, 2022-01)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-6700-把字符串转换成整数atoi-中等-2022-01)
- [`剑指Offer 6801 二叉搜索树的最近公共祖先 (简单, 2022-01)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer-6801-二叉搜索树的最近公共祖先-简单-2022-01)
- [`剑指Offer2 001 整数除法 (中等, 2022-02)`](algorithms/topics/合集-热门&经典&易错.md#剑指offer2-001-整数除法-中等-2022-02)
- [`牛客 0032 求平方根 (简单, 2022-02)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0032-求平方根-简单-2022-02)
- [`牛客 0048 在旋转过的有序数组中寻找目标值 (简单, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0048-在旋转过的有序数组中寻找目标值-简单-2022-03)
- [`牛客 0050 链表中的节点每k个一组翻转 (中等, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0050-链表中的节点每k个一组翻转-中等-2022-03)
- [`牛客 0054 三数之和 (中等, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0054-三数之和-中等-2022-03)
- [`牛客 0066 两个链表的第一个公共结点 (简单, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0066-两个链表的第一个公共结点-简单-2022-03)
- [`牛客 0067 汉诺塔问题 (中等, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0067-汉诺塔问题-中等-2022-03)
- [`牛客 0070 单链表的排序 (简单, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0070-单链表的排序-简单-2022-03)
- [`牛客 0075 数组中只出现一次的两个数字 (中等, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0075-数组中只出现一次的两个数字-中等-2022-03)
- [`牛客 0078 反转链表 (简单, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0078-反转链表-简单-2022-03)
- [`牛客 0079 丑数 (中等, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0079-丑数-中等-2022-03)
- [`牛客 0080 把二叉树打印成多行 (中等, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0080-把二叉树打印成多行-中等-2022-03)
- [`牛客 0145 01背包 (中等, 2022-03)`](algorithms/topics/合集-热门&经典&易错.md#牛客-0145-01背包-中等-2022-03)

</details>

<details><summary><b> 合集-牛客 [84] <a href="algorithms/topics/合集-牛客.md">¶</a></b></summary>

- [`牛客 0001 大数加法 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0001-大数加法-中等-2022-01)
- [`牛客 0002 重排链表 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0002-重排链表-中等-2022-01)
- [`牛客 0003 链表中环的入口结点 (简单, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0003-链表中环的入口结点-简单-2022-01)
- [`牛客 0004 判断链表中是否有环 (简单, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0004-判断链表中是否有环-简单-2022-01)
- [`牛客 0005 二叉树根节点到叶子节点的所有路径和 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0005-二叉树根节点到叶子节点的所有路径和-中等-2022-01)
- [`牛客 0006 二叉树中的最大路径和 (较难, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0006-二叉树中的最大路径和-较难-2022-01)
- [`牛客 0007 买卖股票的最好时机(一) (简单, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0007-买卖股票的最好时机一-简单-2022-01)
- [`牛客 0008 二叉树中和为某一值的路径(二) (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0008-二叉树中和为某一值的路径二-中等-2022-01)
- [`牛客 0009 二叉树中和为某一值的路径(一) (简单, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0009-二叉树中和为某一值的路径一-简单-2022-01)
- [`牛客 0010 大数乘法 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0010-大数乘法-中等-2022-01)
- [`牛客 0011 将升序数组转化为平衡二叉搜索树 (简单, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0011-将升序数组转化为平衡二叉搜索树-简单-2022-01)
- [`牛客 0012 重建二叉树 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0012-重建二叉树-中等-2022-01)
- [`牛客 0013 二叉树的最大深度 (简单, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0013-二叉树的最大深度-简单-2022-01)
- [`牛客 0014 按之字形顺序打印二叉树 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0014-按之字形顺序打印二叉树-中等-2022-01)
- [`牛客 0015 求二叉树的层序遍历 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0015-求二叉树的层序遍历-中等-2022-01)
- [`牛客 0016 对称的二叉树 (简单, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0016-对称的二叉树-简单-2022-01)
- [`牛客 0017 最长回文子串 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0017-最长回文子串-中等-2022-01)
- [`牛客 0018 顺时针旋转矩阵 (简单, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0018-顺时针旋转矩阵-简单-2022-01)
- [`牛客 0019 连续子数组的最大和 (简单, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0019-连续子数组的最大和-简单-2022-01)
- [`牛客 0020 数字字符串转化成IP地址 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0020-数字字符串转化成ip地址-中等-2022-01)
- [`牛客 0021 链表内指定区间反转 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0021-链表内指定区间反转-中等-2022-01)
- [`牛客 0022 合并两个有序的数组 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0022-合并两个有序的数组-中等-2022-01)
- [`牛客 0023 划分链表 (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0023-划分链表-中等-2022-01)
- [`牛客 0024 删除有序链表中重复的元素-II (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0024-删除有序链表中重复的元素-ii-中等-2022-01)
- [`牛客 0025 删除有序链表中重复的元素-I (中等, 2022-01)`](algorithms/topics/合集-牛客.md#牛客-0025-删除有序链表中重复的元素-i-中等-2022-01)
- [`牛客 0026 括号生成 (中等, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0026-括号生成-中等-2022-02)
- [`牛客 0027 集合的所有子集(一) (中等, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0027-集合的所有子集一-中等-2022-02)
- [`牛客 0028 最小覆盖子串 (较难, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0028-最小覆盖子串-较难-2022-02)
- [`牛客 0029 二维数组中的查找 (中等, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0029-二维数组中的查找-中等-2022-02)
- [`牛客 0030 缺失的第一个正整数 (中等, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0030-缺失的第一个正整数-中等-2022-02)
- [`牛客 0031 第一个只出现一次的字符 (简单, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0031-第一个只出现一次的字符-简单-2022-02)
- [`牛客 0032 求平方根 (简单, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0032-求平方根-简单-2022-02)
- [`牛客 0033 合并两个排序的链表 (简单, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0033-合并两个排序的链表-简单-2022-02)
- [`牛客 0034 求路径 (简单, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0034-求路径-简单-2022-02)
- [`牛客 0035 编辑距离(二) (较难, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0035-编辑距离二-较难-2022-02)
- [`牛客 0036 在两个长度相等的排序数组中找到上中位数 (较难, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0036-在两个长度相等的排序数组中找到上中位数-较难-2022-02)
- [`牛客 0037 合并区间 (中等, 2022-02)`](algorithms/topics/合集-牛客.md#牛客-0037-合并区间-中等-2022-02)
- [`牛客 0038 螺旋矩阵 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0038-螺旋矩阵-中等-2022-03)
- [`牛客 0039 N皇后问题 (较难, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0039-n皇后问题-较难-2022-03)
- [`牛客 0040 链表相加(二) (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0040-链表相加二-中等-2022-03)
- [`牛客 0041 最长无重复子数组 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0041-最长无重复子数组-中等-2022-03)
- [`牛客 0042 有重复项数字的全排列 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0042-有重复项数字的全排列-中等-2022-03)
- [`牛客 0043 没有重复项数字的全排列 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0043-没有重复项数字的全排列-中等-2022-03)
- [`牛客 0044 通配符匹配 (较难, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0044-通配符匹配-较难-2022-03)
- [`牛客 0045 实现二叉树先序、中序、后序遍历 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0045-实现二叉树先序中序后序遍历-中等-2022-03)
- [`牛客 0046 加起来和为目标值的组合(二) (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0046-加起来和为目标值的组合二-中等-2022-03)
- [`牛客 0047 数独 (较难, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0047-数独-较难-2022-03)
- [`牛客 0048 在旋转过的有序数组中寻找目标值 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0048-在旋转过的有序数组中寻找目标值-简单-2022-03)
- [`牛客 0049 最长的括号子串 (较难, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0049-最长的括号子串-较难-2022-03)
- [`牛客 0050 链表中的节点每k个一组翻转 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0050-链表中的节点每k个一组翻转-中等-2022-03)
- [`牛客 0051 合并k个已排序的链表 (较难, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0051-合并k个已排序的链表-较难-2022-03)
- [`牛客 0052 有效括号序列 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0052-有效括号序列-简单-2022-03)
- [`牛客 0053 删除链表的倒数第n个节点 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0053-删除链表的倒数第n个节点-中等-2022-03)
- [`牛客 0054 三数之和 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0054-三数之和-中等-2022-03)
- [`牛客 0055 最长公共前缀 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0055-最长公共前缀-简单-2022-03)
- [`牛客 0056 回文数字 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0056-回文数字-简单-2022-03)
- [`牛客 0057 反转数字 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0057-反转数字-简单-2022-03)
- [`牛客 0058 找到搜索二叉树中两个错误的节点 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0058-找到搜索二叉树中两个错误的节点-中等-2022-03)
- [`牛客 0059 矩阵的最小路径和 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0059-矩阵的最小路径和-中等-2022-03)
- [`牛客 0060 判断一棵二叉树是否为搜索二叉树和完全二叉树 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0060-判断一棵二叉树是否为搜索二叉树和完全二叉树-中等-2022-03)
- [`牛客 0061 两数之和 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0061-两数之和-简单-2022-03)
- [`牛客 0062 判断是不是平衡二叉树 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0062-判断是不是平衡二叉树-简单-2022-03)
- [`牛客 0063 扑克牌顺子 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0063-扑克牌顺子-简单-2022-03)
- [`牛客 0064 二叉搜索树与双向链表 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0064-二叉搜索树与双向链表-中等-2022-03)
- [`牛客 0065 斐波那契数列 (入门, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0065-斐波那契数列-入门-2022-03)
- [`牛客 0066 两个链表的第一个公共结点 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0066-两个链表的第一个公共结点-简单-2022-03)
- [`牛客 0067 汉诺塔问题 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0067-汉诺塔问题-中等-2022-03)
- [`牛客 0068 跳台阶 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0068-跳台阶-简单-2022-03)
- [`牛客 0069 链表中倒数最后k个结点 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0069-链表中倒数最后k个结点-简单-2022-03)
- [`牛客 0070 单链表的排序 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0070-单链表的排序-简单-2022-03)
- [`牛客 0071 旋转数组的最小数字 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0071-旋转数组的最小数字-简单-2022-03)
- [`牛客 0072 二叉树的镜像 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0072-二叉树的镜像-简单-2022-03)
- [`牛客 0073 数组中出现次数超过一半的数字 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0073-数组中出现次数超过一半的数字-简单-2022-03)
- [`牛客 0074 数字在升序数组中出现的次数 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0074-数字在升序数组中出现的次数-简单-2022-03)
- [`牛客 0075 数组中只出现一次的两个数字 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0075-数组中只出现一次的两个数字-中等-2022-03)
- [`牛客 0076 用两个栈实现队列 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0076-用两个栈实现队列-简单-2022-03)
- [`牛客 0077 调整数组顺序使奇数位于偶数前面(一) (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0077-调整数组顺序使奇数位于偶数前面一-中等-2022-03)
- [`牛客 0078 反转链表 (简单, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0078-反转链表-简单-2022-03)
- [`牛客 0079 丑数 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0079-丑数-中等-2022-03)
- [`牛客 0080 把二叉树打印成多行 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0080-把二叉树打印成多行-中等-2022-03)
- [`牛客 0081 二叉搜索树的第k个节点 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0081-二叉搜索树的第k个节点-中等-2022-03)
- [`牛客 0091 最长上升子序列(三) (较难, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0091-最长上升子序列三-较难-2022-03)
- [`牛客 0127 最长公共子串 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0127-最长公共子串-中等-2022-03)
- [`牛客 0145 01背包 (中等, 2022-03)`](algorithms/topics/合集-牛客.md#牛客-0145-01背包-中等-2022-03)

</details>

<details><summary><b> 合集-程序员面试金典 [2] <a href="algorithms/topics/合集-程序员面试金典.md">¶</a></b></summary>

- [`程序员面试金典 0101 判定字符是否唯一 (简单, 2022-01)`](algorithms/topics/合集-程序员面试金典.md#程序员面试金典-0101-判定字符是否唯一-简单-2022-01)
- [`程序员面试金典 0102 判定是否互为字符重排 (简单, 2022-01)`](algorithms/topics/合集-程序员面试金典.md#程序员面试金典-0102-判定是否互为字符重排-简单-2022-01)

</details>

<details><summary><b> 合集-纯数学 [1] <a href="algorithms/topics/合集-纯数学.md">¶</a></b></summary>

- [`纯数学 20220126 划分2N个点 (中等, 2022-01)`](algorithms/topics/合集-纯数学.md#纯数学-20220126-划分2n个点-中等-2022-01)

</details>

<details><summary><b>更多细分类型 ...<a href="algorithms/README.md">¶</a></b></summary>

<details><summary><b> 基础-数学 [5] <a href="algorithms/topics/基础-数学.md">¶</a></b></summary>

- [`LeetCode 0343 整数拆分 (中等, 2021-12)`](algorithms/topics/基础-数学.md#leetcode-0343-整数拆分-中等-2021-12)
- [`LeetCode 0441 排列硬币 (简单, 2021-10)`](algorithms/topics/基础-数学.md#leetcode-0441-排列硬币-简单-2021-10)
- [`剑指Offer 1401 剪绳子（整数拆分） (中等, 2021-11)`](algorithms/topics/基础-数学.md#剑指offer-1401-剪绳子整数拆分-中等-2021-11)
- [`剑指Offer 1402 剪绳子 (中等, 2021-11)`](algorithms/topics/基础-数学.md#剑指offer-1402-剪绳子-中等-2021-11)
- [`牛客 0056 回文数字 (简单, 2022-03)`](algorithms/topics/基础-数学.md#牛客-0056-回文数字-简单-2022-03)

</details>

<details><summary><b> 基础-模拟 [21] <a href="algorithms/topics/基础-模拟.md">¶</a></b></summary>

- [`LeetCode 0005 最长回文子串 (中等, 2021-10)`](algorithms/topics/基础-模拟.md#leetcode-0005-最长回文子串-中等-2021-10)
- [`LeetCode 0143 重排链表 (中等, 2022-01)`](algorithms/topics/基础-模拟.md#leetcode-0143-重排链表-中等-2022-01)
- [`LeetCode 0352 将数据流变为多个不相交区间 (困难, 2021-10)`](algorithms/topics/基础-模拟.md#leetcode-0352-将数据流变为多个不相交区间-困难-2021-10)
- [`LeetCode 0859 亲密字符串 (简单, 2021-11)`](algorithms/topics/基础-模拟.md#leetcode-0859-亲密字符串-简单-2021-11)
- [`LeetCode 0915 分割数组 (中等, 2022-01)`](algorithms/topics/基础-模拟.md#leetcode-0915-分割数组-中等-2022-01)
- [`剑指Offer 2900 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/基础-模拟.md#剑指offer-2900-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer 3900 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/基础-模拟.md#剑指offer-3900-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer 4300 1～n整数中1出现的次数 (困难, 2021-12)`](algorithms/topics/基础-模拟.md#剑指offer-4300-1n整数中1出现的次数-困难-2021-12)
- [`剑指Offer 4400 数字序列中某一位的数字 (中等, 2021-12)`](algorithms/topics/基础-模拟.md#剑指offer-4400-数字序列中某一位的数字-中等-2021-12)
- [`剑指Offer 6100 扑克牌中的顺子 (简单, 2022-01)`](algorithms/topics/基础-模拟.md#剑指offer-6100-扑克牌中的顺子-简单-2022-01)
- [`剑指Offer 6200 圆圈中最后剩下的数字（约瑟夫环问题） (中等, 2022-01)`](algorithms/topics/基础-模拟.md#剑指offer-6200-圆圈中最后剩下的数字约瑟夫环问题-中等-2022-01)
- [`剑指Offer 6300 买卖股票的最佳时机 (中等, 2022-01)`](algorithms/topics/基础-模拟.md#剑指offer-6300-买卖股票的最佳时机-中等-2022-01)
- [`剑指Offer 6700 把字符串转换成整数（atoi） (中等, 2022-01)`](algorithms/topics/基础-模拟.md#剑指offer-6700-把字符串转换成整数atoi-中等-2022-01)
- [`牛客 0001 大数加法 (中等, 2022-01)`](algorithms/topics/基础-模拟.md#牛客-0001-大数加法-中等-2022-01)
- [`牛客 0007 买卖股票的最好时机(一) (简单, 2022-01)`](algorithms/topics/基础-模拟.md#牛客-0007-买卖股票的最好时机一-简单-2022-01)
- [`牛客 0010 大数乘法 (中等, 2022-01)`](algorithms/topics/基础-模拟.md#牛客-0010-大数乘法-中等-2022-01)
- [`牛客 0017 最长回文子串 (中等, 2022-01)`](algorithms/topics/基础-模拟.md#牛客-0017-最长回文子串-中等-2022-01)
- [`牛客 0038 螺旋矩阵 (中等, 2022-03)`](algorithms/topics/基础-模拟.md#牛客-0038-螺旋矩阵-中等-2022-03)
- [`牛客 0057 反转数字 (简单, 2022-03)`](algorithms/topics/基础-模拟.md#牛客-0057-反转数字-简单-2022-03)
- [`牛客 0063 扑克牌顺子 (简单, 2022-03)`](algorithms/topics/基础-模拟.md#牛客-0063-扑克牌顺子-简单-2022-03)
- [`牛客 0073 数组中出现次数超过一半的数字 (简单, 2022-03)`](algorithms/topics/基础-模拟.md#牛客-0073-数组中出现次数超过一半的数字-简单-2022-03)

</details>

<details><summary><b> 技巧-从暴力递归到动态规划 [10] <a href="algorithms/topics/技巧-从暴力递归到动态规划.md">¶</a></b></summary>

- [`LeetCode 0091 解码方法 (中等, 2022-02)`](algorithms/topics/技巧-从暴力递归到动态规划.md#leetcode-0091-解码方法-中等-2022-02)
- [`LeetCode 0198 打家劫舍 (中等, 2022-02)`](algorithms/topics/技巧-从暴力递归到动态规划.md#leetcode-0198-打家劫舍-中等-2022-02)
- [`LeetCode 0279 完全平方数 (中等, 2022-02)`](algorithms/topics/技巧-从暴力递归到动态规划.md#leetcode-0279-完全平方数-中等-2022-02)
- [`LeetCode 0322 零钱兑换 (中等, 2022-02)`](algorithms/topics/技巧-从暴力递归到动态规划.md#leetcode-0322-零钱兑换-中等-2022-02)
- [`LeetCode 0474 一和零 (中等, 2022-02)`](algorithms/topics/技巧-从暴力递归到动态规划.md#leetcode-0474-一和零-中等-2022-02)
- [`剑指Offer 6000 n个骰子的点数 (中等, 2022-01)`](algorithms/topics/技巧-从暴力递归到动态规划.md#剑指offer-6000-n个骰子的点数-中等-2022-01)
- [`牛客 0035 编辑距离(二) (较难, 2022-02)`](algorithms/topics/技巧-从暴力递归到动态规划.md#牛客-0035-编辑距离二-较难-2022-02)
- [`牛客 0044 通配符匹配 (较难, 2022-03)`](algorithms/topics/技巧-从暴力递归到动态规划.md#牛客-0044-通配符匹配-较难-2022-03)
- [`牛客 0127 最长公共子串 (中等, 2022-03)`](algorithms/topics/技巧-从暴力递归到动态规划.md#牛客-0127-最长公共子串-中等-2022-03)
- [`牛客 0145 01背包 (中等, 2022-03)`](algorithms/topics/技巧-从暴力递归到动态规划.md#牛客-0145-01背包-中等-2022-03)

</details>

<details><summary><b> 技巧-位运算 [8] <a href="algorithms/topics/技巧-位运算.md">¶</a></b></summary>

- [`LeetCode 0029 两数相除 (中等, 2021-10)`](algorithms/topics/技巧-位运算.md#leetcode-0029-两数相除-中等-2021-10)
- [`LeetCode 0187 重复的DNA序列 (中等, 2021-10)`](algorithms/topics/技巧-位运算.md#leetcode-0187-重复的dna序列-中等-2021-10)
- [`剑指Offer 1500 二进制中1的个数 (简单, 2021-11)`](algorithms/topics/技巧-位运算.md#剑指offer-1500-二进制中1的个数-简单-2021-11)
- [`剑指Offer 5601 数组中数字出现的次数 (中等, 2022-01)`](algorithms/topics/技巧-位运算.md#剑指offer-5601-数组中数字出现的次数-中等-2022-01)
- [`剑指Offer 5602 数组中数字出现的次数 (中等, 2022-01)`](algorithms/topics/技巧-位运算.md#剑指offer-5602-数组中数字出现的次数-中等-2022-01)
- [`剑指Offer 6500 不用加减乘除做加法 (简单, 2022-01)`](algorithms/topics/技巧-位运算.md#剑指offer-6500-不用加减乘除做加法-简单-2022-01)
- [`牛客 0039 N皇后问题 (较难, 2022-03)`](algorithms/topics/技巧-位运算.md#牛客-0039-n皇后问题-较难-2022-03)
- [`牛客 0075 数组中只出现一次的两个数字 (中等, 2022-03)`](algorithms/topics/技巧-位运算.md#牛客-0075-数组中只出现一次的两个数字-中等-2022-03)

</details>

<details><summary><b> 技巧-前缀和 [2] <a href="algorithms/topics/技巧-前缀和.md">¶</a></b></summary>

- [`LeetCode 0437 路径总和III (中等, 2022-02)`](algorithms/topics/技巧-前缀和.md#leetcode-0437-路径总和iii-中等-2022-02)
- [`剑指Offer 6600 构建乘积数组 (中等, 2022-01)`](algorithms/topics/技巧-前缀和.md#剑指offer-6600-构建乘积数组-中等-2022-01)

</details>

<details><summary><b> 技巧-单调栈、单调队列 [2] <a href="algorithms/topics/技巧-单调栈、单调队列.md">¶</a></b></summary>

- [`LeetCode 0496 下一个更大元素 (简单, 2021-11)`](algorithms/topics/技巧-单调栈、单调队列.md#leetcode-0496-下一个更大元素-简单-2021-11)
- [`剑指Offer 5901 滑动窗口的最大值 (困难, 2022-01)`](algorithms/topics/技巧-单调栈、单调队列.md#剑指offer-5901-滑动窗口的最大值-困难-2022-01)

</details>

<details><summary><b> 技巧-双指针 [13] <a href="algorithms/topics/技巧-双指针.md">¶</a></b></summary>

- [`LeetCode 0005 最长回文子串 (中等, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-0005-最长回文子串-中等-2021-10)
- [`LeetCode 0011 盛最多水的容器 (中等, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-0011-盛最多水的容器-中等-2021-10)
- [`LeetCode 0015 三数之和 (中等, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-0015-三数之和-中等-2021-10)
- [`LeetCode 0016 最接近的三数之和 (中等, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-0016-最接近的三数之和-中等-2021-10)
- [`LeetCode 0042 接雨水 (困难, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-0042-接雨水-困难-2021-10)
- [`LeetCode 0167 两数之和2(输入有序数组) (简单, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-0167-两数之和2输入有序数组-简单-2021-10)
- [`LeetCode 0611 有效三角形的个数 (中等, 2021-10)`](algorithms/topics/技巧-双指针.md#leetcode-0611-有效三角形的个数-中等-2021-10)
- [`剑指Offer 2100 调整数组顺序使奇数位于偶数前面 (简单, 2021-11)`](algorithms/topics/技巧-双指针.md#剑指offer-2100-调整数组顺序使奇数位于偶数前面-简单-2021-11)
- [`剑指Offer 4800 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/技巧-双指针.md#剑指offer-4800-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer 5701 和为s的两个数字 (简单, 2022-01)`](algorithms/topics/技巧-双指针.md#剑指offer-5701-和为s的两个数字-简单-2022-01)
- [`剑指Offer 5702 和为s的连续正数序列 (简单, 2022-01)`](algorithms/topics/技巧-双指针.md#剑指offer-5702-和为s的连续正数序列-简单-2022-01)
- [`剑指Offer 5801 翻转单词顺序 (简单, 2022-01)`](algorithms/topics/技巧-双指针.md#剑指offer-5801-翻转单词顺序-简单-2022-01)
- [`牛客 0022 合并两个有序的数组 (中等, 2022-01)`](algorithms/topics/技巧-双指针.md#牛客-0022-合并两个有序的数组-中等-2022-01)

</details>

<details><summary><b> 技巧-双指针-对向双指针 [1] <a href="algorithms/topics/技巧-双指针-对向双指针.md">¶</a></b></summary>

- [`牛客 0054 三数之和 (中等, 2022-03)`](algorithms/topics/技巧-双指针-对向双指针.md#牛客-0054-三数之和-中等-2022-03)

</details>

<details><summary><b> 技巧-双指针-快慢指针 [8] <a href="algorithms/topics/技巧-双指针-快慢指针.md">¶</a></b></summary>

- [`LeetCode 0019 删除链表的倒数第N个结点 (中等, 2022-01)`](algorithms/topics/技巧-双指针-快慢指针.md#leetcode-0019-删除链表的倒数第n个结点-中等-2022-01)
- [`LeetCode 0876 链表的中间结点 (简单, 2022-01)`](algorithms/topics/技巧-双指针-快慢指针.md#leetcode-0876-链表的中间结点-简单-2022-01)
- [`剑指Offer 2200 链表中倒数第k个节点 (简单, 2021-11)`](algorithms/topics/技巧-双指针-快慢指针.md#剑指offer-2200-链表中倒数第k个节点-简单-2021-11)
- [`剑指Offer 5200 两个链表的第一个公共节点 (简单, 2022-01)`](algorithms/topics/技巧-双指针-快慢指针.md#剑指offer-5200-两个链表的第一个公共节点-简单-2022-01)
- [`牛客 0003 链表中环的入口结点 (简单, 2022-01)`](algorithms/topics/技巧-双指针-快慢指针.md#牛客-0003-链表中环的入口结点-简单-2022-01)
- [`牛客 0004 判断链表中是否有环 (简单, 2022-01)`](algorithms/topics/技巧-双指针-快慢指针.md#牛客-0004-判断链表中是否有环-简单-2022-01)
- [`牛客 0053 删除链表的倒数第n个节点 (中等, 2022-03)`](algorithms/topics/技巧-双指针-快慢指针.md#牛客-0053-删除链表的倒数第n个节点-中等-2022-03)
- [`牛客 0069 链表中倒数最后k个结点 (简单, 2022-03)`](algorithms/topics/技巧-双指针-快慢指针.md#牛客-0069-链表中倒数最后k个结点-简单-2022-03)

</details>

<details><summary><b> 技巧-双指针-滑动窗口 [4] <a href="algorithms/topics/技巧-双指针-滑动窗口.md">¶</a></b></summary>

- [`LeetCode 0003 无重复字符的最长子串 (中等, 2022-02)`](algorithms/topics/技巧-双指针-滑动窗口.md#leetcode-0003-无重复字符的最长子串-中等-2022-02)
- [`剑指Offer 5901 滑动窗口的最大值 (困难, 2022-01)`](algorithms/topics/技巧-双指针-滑动窗口.md#剑指offer-5901-滑动窗口的最大值-困难-2022-01)
- [`牛客 0028 最小覆盖子串 (较难, 2022-02)`](algorithms/topics/技巧-双指针-滑动窗口.md#牛客-0028-最小覆盖子串-较难-2022-02)
- [`牛客 0041 最长无重复子数组 (中等, 2022-03)`](algorithms/topics/技巧-双指针-滑动窗口.md#牛客-0041-最长无重复子数组-中等-2022-03)

</details>

<details><summary><b> 技巧-哈希表(Hash) [9] <a href="algorithms/topics/技巧-哈希表(Hash).md">¶</a></b></summary>

- [`LeetCode 0001 两数之和 (简单, 2021-10)`](algorithms/topics/技巧-哈希表(Hash).md#leetcode-0001-两数之和-简单-2021-10)
- [`LeetCode 0187 重复的DNA序列 (中等, 2021-10)`](algorithms/topics/技巧-哈希表(Hash).md#leetcode-0187-重复的dna序列-中等-2021-10)
- [`剑指Offer 0300 数组中重复的数字 (简单, 2021-11)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-0300-数组中重复的数字-简单-2021-11)
- [`剑指Offer 3500 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-3500-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer 4800 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-4800-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer 5000 第一个只出现一次的字符 (简单, 2021-12)`](algorithms/topics/技巧-哈希表(Hash).md#剑指offer-5000-第一个只出现一次的字符-简单-2021-12)
- [`牛客 0031 第一个只出现一次的字符 (简单, 2022-02)`](algorithms/topics/技巧-哈希表(Hash).md#牛客-0031-第一个只出现一次的字符-简单-2022-02)
- [`牛客 0061 两数之和 (简单, 2022-03)`](algorithms/topics/技巧-哈希表(Hash).md#牛客-0061-两数之和-简单-2022-03)
- [`程序员面试金典 0102 判定是否互为字符重排 (简单, 2022-01)`](algorithms/topics/技巧-哈希表(Hash).md#程序员面试金典-0102-判定是否互为字符重排-简单-2022-01)

</details>

<details><summary><b> 技巧-有限状态自动机 [1] <a href="algorithms/topics/技巧-有限状态自动机.md">¶</a></b></summary>

- [`剑指Offer 2000 表示数值的字符串 (中等, 2021-11)`](algorithms/topics/技巧-有限状态自动机.md#剑指offer-2000-表示数值的字符串-中等-2021-11)

</details>

<details><summary><b> 技巧-自底向上的递归技巧 [6] <a href="algorithms/topics/技巧-自底向上的递归技巧.md">¶</a></b></summary>

- [`LeetCode 0110 平衡二叉树 (简单, 2022-02)`](algorithms/topics/技巧-自底向上的递归技巧.md#leetcode-0110-平衡二叉树-简单-2022-02)
- [`LeetCode 0124 二叉树中的最大路径和 (困难, 2022-02)`](algorithms/topics/技巧-自底向上的递归技巧.md#leetcode-0124-二叉树中的最大路径和-困难-2022-02)
- [`LeetCode 0337 打家劫舍III (中等, 2022-02)`](algorithms/topics/技巧-自底向上的递归技巧.md#leetcode-0337-打家劫舍iii-中等-2022-02)
- [`LeetCode 0437 路径总和III (中等, 2022-02)`](algorithms/topics/技巧-自底向上的递归技巧.md#leetcode-0437-路径总和iii-中等-2022-02)
- [`剑指Offer 6802 二叉树的最近公共祖先 (简单, 2022-01)`](algorithms/topics/技巧-自底向上的递归技巧.md#剑指offer-6802-二叉树的最近公共祖先-简单-2022-01)
- [`牛客 0060 判断一棵二叉树是否为搜索二叉树和完全二叉树 (中等, 2022-03)`](algorithms/topics/技巧-自底向上的递归技巧.md#牛客-0060-判断一棵二叉树是否为搜索二叉树和完全二叉树-中等-2022-03)

</details>

<details><summary><b> 技巧-贪心 [2] <a href="algorithms/topics/技巧-贪心.md">¶</a></b></summary>

- [`LeetCode 0300 最长递增子序列 (中等, 2022-01)`](algorithms/topics/技巧-贪心.md#leetcode-0300-最长递增子序列-中等-2022-01)
- [`剑指Offer 1401 剪绳子（整数拆分） (中等, 2021-11)`](algorithms/topics/技巧-贪心.md#剑指offer-1401-剪绳子整数拆分-中等-2021-11)

</details>

<details><summary><b> 数据结构-二叉搜索树 [1] <a href="algorithms/topics/数据结构-二叉搜索树.md">¶</a></b></summary>

- [`剑指Offer 6801 二叉搜索树的最近公共祖先 (简单, 2022-01)`](algorithms/topics/数据结构-二叉搜索树.md#剑指offer-6801-二叉搜索树的最近公共祖先-简单-2022-01)

</details>

<details><summary><b> 数据结构-二叉树 [41] <a href="algorithms/topics/数据结构-二叉树.md">¶</a></b></summary>

- [`LeetCode 0098 验证二叉搜索树 (中等, 2022-03)`](algorithms/topics/数据结构-二叉树.md#leetcode-0098-验证二叉搜索树-中等-2022-03)
- [`LeetCode 0104 二叉树的最大深度 (简单, 2021-10)`](algorithms/topics/数据结构-二叉树.md#leetcode-0104-二叉树的最大深度-简单-2021-10)
- [`LeetCode 0111 二叉树的最小深度 (简单, 2021-10)`](algorithms/topics/数据结构-二叉树.md#leetcode-0111-二叉树的最小深度-简单-2021-10)
- [`LeetCode 0112 路径总和 (简单, 2022-02)`](algorithms/topics/数据结构-二叉树.md#leetcode-0112-路径总和-简单-2022-02)
- [`LeetCode 0113 路径总和II (中等, 2022-02)`](algorithms/topics/数据结构-二叉树.md#leetcode-0113-路径总和ii-中等-2022-02)
- [`LeetCode 0129 求根节点到叶节点数字之和 (中等, 2022-02)`](algorithms/topics/数据结构-二叉树.md#leetcode-0129-求根节点到叶节点数字之和-中等-2022-02)
- [`LeetCode 0257 二叉树的所有路径 (简单, 2022-02)`](algorithms/topics/数据结构-二叉树.md#leetcode-0257-二叉树的所有路径-简单-2022-02)
- [`LeetCode 0437 路径总和III (中等, 2022-02)`](algorithms/topics/数据结构-二叉树.md#leetcode-0437-路径总和iii-中等-2022-02)
- [`LeetCode 0958 二叉树的完全性检验 (中等, 2022-03)`](algorithms/topics/数据结构-二叉树.md#leetcode-0958-二叉树的完全性检验-中等-2022-03)
- [`LeetCode 0988 从叶结点开始的最小字符串 (中等, 2022-02)`](algorithms/topics/数据结构-二叉树.md#leetcode-0988-从叶结点开始的最小字符串-中等-2022-02)
- [`剑指Offer 0700 重建二叉树 (中等, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-0700-重建二叉树-中等-2021-11)
- [`剑指Offer 2600 树的子结构 (中等, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-2600-树的子结构-中等-2021-11)
- [`剑指Offer 2700 二叉树的镜像 (简单, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-2700-二叉树的镜像-简单-2021-11)
- [`剑指Offer 2800 对称的二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-2800-对称的二叉树-简单-2021-11)
- [`剑指Offer 3201 层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-3201-层序遍历二叉树-简单-2021-11)
- [`剑指Offer 3202 层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-3202-层序遍历二叉树-简单-2021-11)
- [`剑指Offer 3203 层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/数据结构-二叉树.md#剑指offer-3203-层序遍历二叉树之字形遍历-简单-2021-11)
- [`剑指Offer 3300 二叉搜索树的后序遍历序列 (中等, 2021-12)`](algorithms/topics/数据结构-二叉树.md#剑指offer-3300-二叉搜索树的后序遍历序列-中等-2021-12)
- [`剑指Offer 3400 二叉树中和为某一值的路径 (中等, 2021-12)`](algorithms/topics/数据结构-二叉树.md#剑指offer-3400-二叉树中和为某一值的路径-中等-2021-12)
- [`剑指Offer 3600 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/数据结构-二叉树.md#剑指offer-3600-二叉搜索树与双向链表-中等-2021-12)
- [`剑指Offer 3700 序列化二叉树 (困难, 2021-12)`](algorithms/topics/数据结构-二叉树.md#剑指offer-3700-序列化二叉树-困难-2021-12)
- [`剑指Offer 5400 二叉搜索树的第k大节点 (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#剑指offer-5400-二叉搜索树的第k大节点-简单-2022-01)
- [`剑指Offer 5501 求二叉树的深度 (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#剑指offer-5501-求二叉树的深度-简单-2022-01)
- [`剑指Offer 5502 判断是否为平衡二叉树 (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#剑指offer-5502-判断是否为平衡二叉树-简单-2022-01)
- [`剑指Offer 6802 二叉树的最近公共祖先 (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#剑指offer-6802-二叉树的最近公共祖先-简单-2022-01)
- [`牛客 0005 二叉树根节点到叶子节点的所有路径和 (中等, 2022-01)`](algorithms/topics/数据结构-二叉树.md#牛客-0005-二叉树根节点到叶子节点的所有路径和-中等-2022-01)
- [`牛客 0006 二叉树中的最大路径和 (较难, 2022-01)`](algorithms/topics/数据结构-二叉树.md#牛客-0006-二叉树中的最大路径和-较难-2022-01)
- [`牛客 0008 二叉树中和为某一值的路径(二) (中等, 2022-01)`](algorithms/topics/数据结构-二叉树.md#牛客-0008-二叉树中和为某一值的路径二-中等-2022-01)
- [`牛客 0009 二叉树中和为某一值的路径(一) (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#牛客-0009-二叉树中和为某一值的路径一-简单-2022-01)
- [`牛客 0011 将升序数组转化为平衡二叉搜索树 (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#牛客-0011-将升序数组转化为平衡二叉搜索树-简单-2022-01)
- [`牛客 0012 重建二叉树 (中等, 2022-01)`](algorithms/topics/数据结构-二叉树.md#牛客-0012-重建二叉树-中等-2022-01)
- [`牛客 0013 二叉树的最大深度 (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#牛客-0013-二叉树的最大深度-简单-2022-01)
- [`牛客 0014 按之字形顺序打印二叉树 (中等, 2022-01)`](algorithms/topics/数据结构-二叉树.md#牛客-0014-按之字形顺序打印二叉树-中等-2022-01)
- [`牛客 0015 求二叉树的层序遍历 (中等, 2022-01)`](algorithms/topics/数据结构-二叉树.md#牛客-0015-求二叉树的层序遍历-中等-2022-01)
- [`牛客 0016 对称的二叉树 (简单, 2022-01)`](algorithms/topics/数据结构-二叉树.md#牛客-0016-对称的二叉树-简单-2022-01)
- [`牛客 0058 找到搜索二叉树中两个错误的节点 (中等, 2022-03)`](algorithms/topics/数据结构-二叉树.md#牛客-0058-找到搜索二叉树中两个错误的节点-中等-2022-03)
- [`牛客 0062 判断是不是平衡二叉树 (简单, 2022-03)`](algorithms/topics/数据结构-二叉树.md#牛客-0062-判断是不是平衡二叉树-简单-2022-03)
- [`牛客 0064 二叉搜索树与双向链表 (中等, 2022-03)`](algorithms/topics/数据结构-二叉树.md#牛客-0064-二叉搜索树与双向链表-中等-2022-03)
- [`牛客 0072 二叉树的镜像 (简单, 2022-03)`](algorithms/topics/数据结构-二叉树.md#牛客-0072-二叉树的镜像-简单-2022-03)
- [`牛客 0080 把二叉树打印成多行 (中等, 2022-03)`](algorithms/topics/数据结构-二叉树.md#牛客-0080-把二叉树打印成多行-中等-2022-03)
- [`牛客 0081 二叉搜索树的第k个节点 (中等, 2022-03)`](algorithms/topics/数据结构-二叉树.md#牛客-0081-二叉搜索树的第k个节点-中等-2022-03)

</details>

<details><summary><b> 数据结构-堆、优先队列 [4] <a href="algorithms/topics/数据结构-堆、优先队列.md">¶</a></b></summary>

- [`剑指Offer 4000 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/数据结构-堆、优先队列.md#剑指offer-4000-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer 4100 数据流中的中位数 (困难, 2021-12)`](algorithms/topics/数据结构-堆、优先队列.md#剑指offer-4100-数据流中的中位数-困难-2021-12)
- [`剑指Offer2 076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/数据结构-堆、优先队列.md#剑指offer2-076-数组中的第k大的数字-中等-2022-02)
- [`牛客 0051 合并k个已排序的链表 (较难, 2022-03)`](algorithms/topics/数据结构-堆、优先队列.md#牛客-0051-合并k个已排序的链表-较难-2022-03)

</details>

<details><summary><b> 数据结构-字符串 [10] <a href="algorithms/topics/数据结构-字符串.md">¶</a></b></summary>

- [`LeetCode 0434 字符串中的单词数 (简单, 2021-10)`](algorithms/topics/数据结构-字符串.md#leetcode-0434-字符串中的单词数-简单-2021-10)
- [`LeetCode 0859 亲密字符串 (简单, 2021-11)`](algorithms/topics/数据结构-字符串.md#leetcode-0859-亲密字符串-简单-2021-11)
- [`剑指Offer 0500 替换空格 (简单, 2021-11)`](algorithms/topics/数据结构-字符串.md#剑指offer-0500-替换空格-简单-2021-11)
- [`剑指Offer 1900 正则表达式匹配 (困难, 2021-11)`](algorithms/topics/数据结构-字符串.md#剑指offer-1900-正则表达式匹配-困难-2021-11)
- [`剑指Offer 2000 表示数值的字符串 (中等, 2021-11)`](algorithms/topics/数据结构-字符串.md#剑指offer-2000-表示数值的字符串-中等-2021-11)
- [`剑指Offer 5802 左旋转字符串 (简单, 2022-01)`](algorithms/topics/数据结构-字符串.md#剑指offer-5802-左旋转字符串-简单-2022-01)
- [`剑指Offer 6700 把字符串转换成整数（atoi） (中等, 2022-01)`](algorithms/topics/数据结构-字符串.md#剑指offer-6700-把字符串转换成整数atoi-中等-2022-01)
- [`牛客 0001 大数加法 (中等, 2022-01)`](algorithms/topics/数据结构-字符串.md#牛客-0001-大数加法-中等-2022-01)
- [`牛客 0010 大数乘法 (中等, 2022-01)`](algorithms/topics/数据结构-字符串.md#牛客-0010-大数乘法-中等-2022-01)
- [`牛客 0055 最长公共前缀 (简单, 2022-03)`](algorithms/topics/数据结构-字符串.md#牛客-0055-最长公共前缀-简单-2022-03)

</details>

<details><summary><b> 数据结构-数组、矩阵(二维数组) [9] <a href="algorithms/topics/数据结构-数组、矩阵(二维数组).md">¶</a></b></summary>

- [`剑指Offer 2100 调整数组顺序使奇数位于偶数前面 (简单, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-2100-调整数组顺序使奇数位于偶数前面-简单-2021-11)
- [`剑指Offer 2900 顺时针打印矩阵（3种思路4个写法） (中等, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-2900-顺时针打印矩阵3种思路4个写法-中等-2021-11)
- [`剑指Offer 3000 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-3000-包含min函数的栈-简单-2021-11)
- [`剑指Offer 3100 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#剑指offer-3100-栈的压入弹出序列-中等-2021-11)
- [`牛客 0018 顺时针旋转矩阵 (简单, 2022-01)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#牛客-0018-顺时针旋转矩阵-简单-2022-01)
- [`牛客 0030 缺失的第一个正整数 (中等, 2022-02)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#牛客-0030-缺失的第一个正整数-中等-2022-02)
- [`牛客 0038 螺旋矩阵 (中等, 2022-03)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#牛客-0038-螺旋矩阵-中等-2022-03)
- [`牛客 0055 最长公共前缀 (简单, 2022-03)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#牛客-0055-最长公共前缀-简单-2022-03)
- [`牛客 0077 调整数组顺序使奇数位于偶数前面(一) (中等, 2022-03)`](algorithms/topics/数据结构-数组、矩阵(二维数组).md#牛客-0077-调整数组顺序使奇数位于偶数前面一-中等-2022-03)

</details>

<details><summary><b> 数据结构-栈、队列 [14] <a href="algorithms/topics/数据结构-栈、队列.md">¶</a></b></summary>

- [`LeetCode 0020 有效的括号 (简单, 2022-03)`](algorithms/topics/数据结构-栈、队列.md#leetcode-0020-有效的括号-简单-2022-03)
- [`剑指Offer 0600 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-0600-从尾到头打印链表-简单-2021-11)
- [`剑指Offer 0900 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-0900-用两个栈实现队列-简单-2021-11)
- [`剑指Offer 0900 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-0900-用两个栈实现队列-简单-2021-11)
- [`剑指Offer 3000 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-3000-包含min函数的栈-简单-2021-11)
- [`剑指Offer 3100 栈的压入、弹出序列 (中等, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-3100-栈的压入弹出序列-中等-2021-11)
- [`剑指Offer 3201 层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-3201-层序遍历二叉树-简单-2021-11)
- [`剑指Offer 3202 层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-3202-层序遍历二叉树-简单-2021-11)
- [`剑指Offer 3203 层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-3203-层序遍历二叉树之字形遍历-简单-2021-11)
- [`剑指Offer 5902 队列的最大值 (中等, 2022-01)`](algorithms/topics/数据结构-栈、队列.md#剑指offer-5902-队列的最大值-中等-2022-01)
- [`牛客 0014 按之字形顺序打印二叉树 (中等, 2022-01)`](algorithms/topics/数据结构-栈、队列.md#牛客-0014-按之字形顺序打印二叉树-中等-2022-01)
- [`牛客 0049 最长的括号子串 (较难, 2022-03)`](algorithms/topics/数据结构-栈、队列.md#牛客-0049-最长的括号子串-较难-2022-03)
- [`牛客 0052 有效括号序列 (简单, 2022-03)`](algorithms/topics/数据结构-栈、队列.md#牛客-0052-有效括号序列-简单-2022-03)
- [`牛客 0076 用两个栈实现队列 (简单, 2022-03)`](algorithms/topics/数据结构-栈、队列.md#牛客-0076-用两个栈实现队列-简单-2022-03)

</details>

<details><summary><b> 数据结构-线段树、树状数组 [1] <a href="algorithms/topics/数据结构-线段树、树状数组.md">¶</a></b></summary>

- [`剑指Offer 5100 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/数据结构-线段树、树状数组.md#剑指offer-5100-数组中的逆序对-困难-2022-01)

</details>

<details><summary><b> 数据结构-设计 [4] <a href="algorithms/topics/数据结构-设计.md">¶</a></b></summary>

- [`剑指Offer 0900 用两个栈实现队列 (简单, 2021-11)`](algorithms/topics/数据结构-设计.md#剑指offer-0900-用两个栈实现队列-简单-2021-11)
- [`剑指Offer 3000 包含min函数的栈 (简单, 2021-11)`](algorithms/topics/数据结构-设计.md#剑指offer-3000-包含min函数的栈-简单-2021-11)
- [`剑指Offer 4100 数据流中的中位数 (困难, 2021-12)`](algorithms/topics/数据结构-设计.md#剑指offer-4100-数据流中的中位数-困难-2021-12)
- [`剑指Offer 5902 队列的最大值 (中等, 2022-01)`](algorithms/topics/数据结构-设计.md#剑指offer-5902-队列的最大值-中等-2022-01)

</details>

<details><summary><b> 数据结构-链表 [26] <a href="algorithms/topics/数据结构-链表.md">¶</a></b></summary>

- [`LeetCode 0002 两数相加 (中等, 2021-10)`](algorithms/topics/数据结构-链表.md#leetcode-0002-两数相加-中等-2021-10)
- [`LeetCode 0019 删除链表的倒数第N个结点 (中等, 2022-01)`](algorithms/topics/数据结构-链表.md#leetcode-0019-删除链表的倒数第n个结点-中等-2022-01)
- [`LeetCode 0025 K个一组翻转链表 (困难, 2022-02)`](algorithms/topics/数据结构-链表.md#leetcode-0025-k个一组翻转链表-困难-2022-02)
- [`LeetCode 0086 分隔链表 (中等, 2021-10)`](algorithms/topics/数据结构-链表.md#leetcode-0086-分隔链表-中等-2021-10)
- [`LeetCode 0143 重排链表 (中等, 2022-01)`](algorithms/topics/数据结构-链表.md#leetcode-0143-重排链表-中等-2022-01)
- [`LeetCode 0876 链表的中间结点 (简单, 2022-01)`](algorithms/topics/数据结构-链表.md#leetcode-0876-链表的中间结点-简单-2022-01)
- [`剑指Offer 0600 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-0600-从尾到头打印链表-简单-2021-11)
- [`剑指Offer 1800 删除链表的节点 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-1800-删除链表的节点-简单-2021-11)
- [`剑指Offer 2200 链表中倒数第k个节点 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-2200-链表中倒数第k个节点-简单-2021-11)
- [`剑指Offer 2400 反转链表 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-2400-反转链表-简单-2021-11)
- [`剑指Offer 2500 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/数据结构-链表.md#剑指offer-2500-合并两个排序的链表-简单-2021-11)
- [`剑指Offer 3500 复杂链表的复制（深拷贝） (中等, 2021-12)`](algorithms/topics/数据结构-链表.md#剑指offer-3500-复杂链表的复制深拷贝-中等-2021-12)
- [`剑指Offer 5200 两个链表的第一个公共节点 (简单, 2022-01)`](algorithms/topics/数据结构-链表.md#剑指offer-5200-两个链表的第一个公共节点-简单-2022-01)
- [`牛客 0002 重排链表 (中等, 2022-01)`](algorithms/topics/数据结构-链表.md#牛客-0002-重排链表-中等-2022-01)
- [`牛客 0003 链表中环的入口结点 (简单, 2022-01)`](algorithms/topics/数据结构-链表.md#牛客-0003-链表中环的入口结点-简单-2022-01)
- [`牛客 0004 判断链表中是否有环 (简单, 2022-01)`](algorithms/topics/数据结构-链表.md#牛客-0004-判断链表中是否有环-简单-2022-01)
- [`牛客 0021 链表内指定区间反转 (中等, 2022-01)`](algorithms/topics/数据结构-链表.md#牛客-0021-链表内指定区间反转-中等-2022-01)
- [`牛客 0023 划分链表 (中等, 2022-01)`](algorithms/topics/数据结构-链表.md#牛客-0023-划分链表-中等-2022-01)
- [`牛客 0024 删除有序链表中重复的元素-II (中等, 2022-01)`](algorithms/topics/数据结构-链表.md#牛客-0024-删除有序链表中重复的元素-ii-中等-2022-01)
- [`牛客 0025 删除有序链表中重复的元素-I (中等, 2022-01)`](algorithms/topics/数据结构-链表.md#牛客-0025-删除有序链表中重复的元素-i-中等-2022-01)
- [`牛客 0033 合并两个排序的链表 (简单, 2022-02)`](algorithms/topics/数据结构-链表.md#牛客-0033-合并两个排序的链表-简单-2022-02)
- [`牛客 0040 链表相加(二) (中等, 2022-03)`](algorithms/topics/数据结构-链表.md#牛客-0040-链表相加二-中等-2022-03)
- [`牛客 0050 链表中的节点每k个一组翻转 (中等, 2022-03)`](algorithms/topics/数据结构-链表.md#牛客-0050-链表中的节点每k个一组翻转-中等-2022-03)
- [`牛客 0066 两个链表的第一个公共结点 (简单, 2022-03)`](algorithms/topics/数据结构-链表.md#牛客-0066-两个链表的第一个公共结点-简单-2022-03)
- [`牛客 0070 单链表的排序 (简单, 2022-03)`](algorithms/topics/数据结构-链表.md#牛客-0070-单链表的排序-简单-2022-03)
- [`牛客 0078 反转链表 (简单, 2022-03)`](algorithms/topics/数据结构-链表.md#牛客-0078-反转链表-简单-2022-03)

</details>

<details><summary><b> 算法-二分 [19] <a href="algorithms/topics/算法-二分.md">¶</a></b></summary>

- [`LeetCode 0004 寻找两个正序数组的中位数 (困难, 2022-02)`](algorithms/topics/算法-二分.md#leetcode-0004-寻找两个正序数组的中位数-困难-2022-02)
- [`LeetCode 0029 两数相除 (中等, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-0029-两数相除-中等-2021-10)
- [`LeetCode 0033 搜索旋转排序数组 (中等, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-0033-搜索旋转排序数组-中等-2021-10)
- [`LeetCode 0240 搜索二维矩阵2 (中等, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-0240-搜索二维矩阵2-中等-2021-10)
- [`LeetCode 0352 将数据流变为多个不相交区间 (困难, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-0352-将数据流变为多个不相交区间-困难-2021-10)
- [`LeetCode 0441 排列硬币 (简单, 2021-10)`](algorithms/topics/算法-二分.md#leetcode-0441-排列硬币-简单-2021-10)
- [`剑指Offer 0400 二维数组中的查找 (中等, 2021-11)`](algorithms/topics/算法-二分.md#剑指offer-0400-二维数组中的查找-中等-2021-11)
- [`剑指Offer 1100 旋转数组的最小数字 (简单, 2021-11)`](algorithms/topics/算法-二分.md#剑指offer-1100-旋转数组的最小数字-简单-2021-11)
- [`剑指Offer 1600 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/算法-二分.md#剑指offer-1600-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer 5301 求0～n-1中缺失的数字 (简单, 2022-01)`](algorithms/topics/算法-二分.md#剑指offer-5301-求0n-1中缺失的数字-简单-2022-01)
- [`剑指Offer 5302 在排序数组中查找数字 (简单, 2022-01)`](algorithms/topics/算法-二分.md#剑指offer-5302-在排序数组中查找数字-简单-2022-01)
- [`剑指Offer2 001 整数除法 (中等, 2022-02)`](algorithms/topics/算法-二分.md#剑指offer2-001-整数除法-中等-2022-02)
- [`剑指Offer2 069 山峰数组的顶部 (简单, 2022-02)`](algorithms/topics/算法-二分.md#剑指offer2-069-山峰数组的顶部-简单-2022-02)
- [`牛客 0029 二维数组中的查找 (中等, 2022-02)`](algorithms/topics/算法-二分.md#牛客-0029-二维数组中的查找-中等-2022-02)
- [`牛客 0032 求平方根 (简单, 2022-02)`](algorithms/topics/算法-二分.md#牛客-0032-求平方根-简单-2022-02)
- [`牛客 0036 在两个长度相等的排序数组中找到上中位数 (较难, 2022-02)`](algorithms/topics/算法-二分.md#牛客-0036-在两个长度相等的排序数组中找到上中位数-较难-2022-02)
- [`牛客 0048 在旋转过的有序数组中寻找目标值 (简单, 2022-03)`](algorithms/topics/算法-二分.md#牛客-0048-在旋转过的有序数组中寻找目标值-简单-2022-03)
- [`牛客 0071 旋转数组的最小数字 (简单, 2022-03)`](algorithms/topics/算法-二分.md#牛客-0071-旋转数组的最小数字-简单-2022-03)
- [`牛客 0074 数字在升序数组中出现的次数 (简单, 2022-03)`](algorithms/topics/算法-二分.md#牛客-0074-数字在升序数组中出现的次数-简单-2022-03)

</details>

<details><summary><b> 算法-分治 [4] <a href="algorithms/topics/算法-分治.md">¶</a></b></summary>

- [`剑指Offer 0700 重建二叉树 (中等, 2021-11)`](algorithms/topics/算法-分治.md#剑指offer-0700-重建二叉树-中等-2021-11)
- [`剑指Offer 3900 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/算法-分治.md#剑指offer-3900-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer 5100 数组中的逆序对 (困难, 2022-01)`](algorithms/topics/算法-分治.md#剑指offer-5100-数组中的逆序对-困难-2022-01)
- [`剑指Offer2 076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/算法-分治.md#剑指offer2-076-数组中的第k大的数字-中等-2022-02)

</details>

<details><summary><b> 算法-动态规划(记忆化搜索)、递推 [42] <a href="algorithms/topics/算法-动态规划(记忆化搜索)、递推.md">¶</a></b></summary>

- [`LeetCode 0005 最长回文子串 (中等, 2021-10)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0005-最长回文子串-中等-2021-10)
- [`LeetCode 0010 正则表达式匹配 (困难, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0010-正则表达式匹配-困难-2022-01)
- [`LeetCode 0053 最大子数组和 (简单, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0053-最大子数组和-简单-2022-01)
- [`LeetCode 0064 最小路径和 (中等, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0064-最小路径和-中等-2022-01)
- [`LeetCode 0070 爬楼梯 (简单, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0070-爬楼梯-简单-2022-01)
- [`LeetCode 0072 编辑距离 (困难, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0072-编辑距离-困难-2022-01)
- [`LeetCode 0091 解码方法 (中等, 2022-02)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0091-解码方法-中等-2022-02)
- [`LeetCode 0096 不同的二叉搜索树 (中等, 2022-03)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0096-不同的二叉搜索树-中等-2022-03)
- [`LeetCode 0120 三角形最小路径和 (中等, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0120-三角形最小路径和-中等-2022-01)
- [`LeetCode 0121 买卖股票的最佳时机 (简单, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0121-买卖股票的最佳时机-简单-2022-01)
- [`LeetCode 0122 买卖股票的最佳时机II (中等, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0122-买卖股票的最佳时机ii-中等-2022-01)
- [`LeetCode 0123 买卖股票的最佳时机III (困难, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0123-买卖股票的最佳时机iii-困难-2022-01)
- [`LeetCode 0152 乘积最大子数组 (中等, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0152-乘积最大子数组-中等-2022-01)
- [`LeetCode 0198 打家劫舍 (中等, 2022-02)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0198-打家劫舍-中等-2022-02)
- [`LeetCode 0213 打家劫舍II (中等, 2022-02)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0213-打家劫舍ii-中等-2022-02)
- [`LeetCode 0300 最长递增子序列 (中等, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0300-最长递增子序列-中等-2022-01)
- [`LeetCode 0322 零钱兑换 (中等, 2022-02)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0322-零钱兑换-中等-2022-02)
- [`LeetCode 0343 整数拆分 (中等, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0343-整数拆分-中等-2021-12)
- [`LeetCode 0518 零钱兑换II (中等, 2022-02)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#leetcode-0518-零钱兑换ii-中等-2022-02)
- [`剑指Offer 1001 斐波那契数列 (简单, 2021-11)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-1001-斐波那契数列-简单-2021-11)
- [`剑指Offer 1001 斐波那契数列 (简单, 2021-11)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-1001-斐波那契数列-简单-2021-11)
- [`剑指Offer 1002 跳台阶 (简单, 2021-11)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-1002-跳台阶-简单-2021-11)
- [`剑指Offer 1401 剪绳子（整数拆分） (中等, 2021-11)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-1401-剪绳子整数拆分-中等-2021-11)
- [`剑指Offer 1900 正则表达式匹配 (困难, 2021-11)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-1900-正则表达式匹配-困难-2021-11)
- [`剑指Offer 4200 连续子数组的最大和 (简单, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-4200-连续子数组的最大和-简单-2021-12)
- [`剑指Offer 4600 斐波那契数列-3（把数字翻译成字符串） (中等, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-4600-斐波那契数列-3把数字翻译成字符串-中等-2021-12)
- [`剑指Offer 4700 礼物的最大价值 (中等, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-4700-礼物的最大价值-中等-2021-12)
- [`剑指Offer 4800 最长不含重复字符的子字符串 (中等, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-4800-最长不含重复字符的子字符串-中等-2021-12)
- [`剑指Offer 4900 丑数 (中等, 2021-12)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-4900-丑数-中等-2021-12)
- [`剑指Offer 6000 n个骰子的点数 (中等, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-6000-n个骰子的点数-中等-2022-01)
- [`剑指Offer 6200 圆圈中最后剩下的数字（约瑟夫环问题） (中等, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#剑指offer-6200-圆圈中最后剩下的数字约瑟夫环问题-中等-2022-01)
- [`牛客 0017 最长回文子串 (中等, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#牛客-0017-最长回文子串-中等-2022-01)
- [`牛客 0019 连续子数组的最大和 (简单, 2022-01)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#牛客-0019-连续子数组的最大和-简单-2022-01)
- [`牛客 0034 求路径 (简单, 2022-02)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#牛客-0034-求路径-简单-2022-02)
- [`牛客 0035 编辑距离(二) (较难, 2022-02)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#牛客-0035-编辑距离二-较难-2022-02)
- [`牛客 0059 矩阵的最小路径和 (中等, 2022-03)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#牛客-0059-矩阵的最小路径和-中等-2022-03)
- [`牛客 0065 斐波那契数列 (入门, 2022-03)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#牛客-0065-斐波那契数列-入门-2022-03)
- [`牛客 0068 跳台阶 (简单, 2022-03)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#牛客-0068-跳台阶-简单-2022-03)
- [`牛客 0079 丑数 (中等, 2022-03)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#牛客-0079-丑数-中等-2022-03)
- [`牛客 0091 最长上升子序列(三) (较难, 2022-03)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#牛客-0091-最长上升子序列三-较难-2022-03)
- [`牛客 0127 最长公共子串 (中等, 2022-03)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#牛客-0127-最长公共子串-中等-2022-03)
- [`牛客 0145 01背包 (中等, 2022-03)`](algorithms/topics/算法-动态规划(记忆化搜索)、递推.md#牛客-0145-01背包-中等-2022-03)

</details>

<details><summary><b> 算法-广度优先搜索(BFS) [3] <a href="algorithms/topics/算法-广度优先搜索(BFS).md">¶</a></b></summary>

- [`剑指Offer 3201 层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/算法-广度优先搜索(BFS).md#剑指offer-3201-层序遍历二叉树-简单-2021-11)
- [`剑指Offer 3202 层序遍历二叉树 (简单, 2021-11)`](algorithms/topics/算法-广度优先搜索(BFS).md#剑指offer-3202-层序遍历二叉树-简单-2021-11)
- [`剑指Offer 3203 层序遍历二叉树（之字形遍历） (简单, 2021-11)`](algorithms/topics/算法-广度优先搜索(BFS).md#剑指offer-3203-层序遍历二叉树之字形遍历-简单-2021-11)

</details>

<details><summary><b> 算法-排序 [7] <a href="algorithms/topics/算法-排序.md">¶</a></b></summary>

- [`剑指Offer 3900 数组中出现次数超过一半的数字（摩尔投票） (简单, 2021-12)`](algorithms/topics/算法-排序.md#剑指offer-3900-数组中出现次数超过一半的数字摩尔投票-简单-2021-12)
- [`剑指Offer 4000 最小的k个数（partition操作） (简单, 2021-12)`](algorithms/topics/算法-排序.md#剑指offer-4000-最小的k个数partition操作-简单-2021-12)
- [`剑指Offer 4500 把数组排成最小的数 (中等, 2021-12)`](algorithms/topics/算法-排序.md#剑指offer-4500-把数组排成最小的数-中等-2021-12)
- [`剑指Offer 6100 扑克牌中的顺子 (简单, 2022-01)`](algorithms/topics/算法-排序.md#剑指offer-6100-扑克牌中的顺子-简单-2022-01)
- [`剑指Offer2 076 数组中的第K大的数字 (中等, 2022-02)`](algorithms/topics/算法-排序.md#剑指offer2-076-数组中的第k大的数字-中等-2022-02)
- [`牛客 0037 合并区间 (中等, 2022-02)`](algorithms/topics/算法-排序.md#牛客-0037-合并区间-中等-2022-02)
- [`程序员面试金典 0101 判定字符是否唯一 (简单, 2022-01)`](algorithms/topics/算法-排序.md#程序员面试金典-0101-判定字符是否唯一-简单-2022-01)

</details>

<details><summary><b> 算法-深度优先搜索(DFS) [15] <a href="algorithms/topics/算法-深度优先搜索(DFS).md">¶</a></b></summary>

- [`LeetCode 0111 二叉树的最小深度 (简单, 2021-10)`](algorithms/topics/算法-深度优先搜索(DFS).md#leetcode-0111-二叉树的最小深度-简单-2021-10)
- [`LeetCode 0437 路径总和III (中等, 2022-02)`](algorithms/topics/算法-深度优先搜索(DFS).md#leetcode-0437-路径总和iii-中等-2022-02)
- [`剑指Offer 0600 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-0600-从尾到头打印链表-简单-2021-11)
- [`剑指Offer 1200 矩阵中的路径 (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-1200-矩阵中的路径-中等-2021-11)
- [`剑指Offer 1200 矩阵中的路径 (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-1200-矩阵中的路径-中等-2021-11)
- [`剑指Offer 1300 机器人的运动范围 (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-1300-机器人的运动范围-中等-2021-11)
- [`剑指Offer 1700 打印从1到最大的n位数（N叉树的遍历） (中等, 2021-11)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-1700-打印从1到最大的n位数n叉树的遍历-中等-2021-11)
- [`剑指Offer 3400 二叉树中和为某一值的路径 (中等, 2021-12)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-3400-二叉树中和为某一值的路径-中等-2021-12)
- [`剑指Offer 3800 字符串的排列（全排列） (中等, 2021-12)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-3800-字符串的排列全排列-中等-2021-12)
- [`剑指Offer 5400 二叉搜索树的第k大节点 (简单, 2022-01)`](algorithms/topics/算法-深度优先搜索(DFS).md#剑指offer-5400-二叉搜索树的第k大节点-简单-2022-01)
- [`牛客 0005 二叉树根节点到叶子节点的所有路径和 (中等, 2022-01)`](algorithms/topics/算法-深度优先搜索(DFS).md#牛客-0005-二叉树根节点到叶子节点的所有路径和-中等-2022-01)
- [`牛客 0008 二叉树中和为某一值的路径(二) (中等, 2022-01)`](algorithms/topics/算法-深度优先搜索(DFS).md#牛客-0008-二叉树中和为某一值的路径二-中等-2022-01)
- [`牛客 0009 二叉树中和为某一值的路径(一) (简单, 2022-01)`](algorithms/topics/算法-深度优先搜索(DFS).md#牛客-0009-二叉树中和为某一值的路径一-简单-2022-01)
- [`牛客 0020 数字字符串转化成IP地址 (中等, 2022-01)`](algorithms/topics/算法-深度优先搜索(DFS).md#牛客-0020-数字字符串转化成ip地址-中等-2022-01)
- [`牛客 0045 实现二叉树先序、中序、后序遍历 (中等, 2022-03)`](algorithms/topics/算法-深度优先搜索(DFS).md#牛客-0045-实现二叉树先序中序后序遍历-中等-2022-03)

</details>

<details><summary><b> 算法-递归-回溯 [6] <a href="algorithms/topics/算法-递归-回溯.md">¶</a></b></summary>

- [`牛客 0026 括号生成 (中等, 2022-02)`](algorithms/topics/算法-递归-回溯.md#牛客-0026-括号生成-中等-2022-02)
- [`牛客 0027 集合的所有子集(一) (中等, 2022-02)`](algorithms/topics/算法-递归-回溯.md#牛客-0027-集合的所有子集一-中等-2022-02)
- [`牛客 0042 有重复项数字的全排列 (中等, 2022-03)`](algorithms/topics/算法-递归-回溯.md#牛客-0042-有重复项数字的全排列-中等-2022-03)
- [`牛客 0043 没有重复项数字的全排列 (中等, 2022-03)`](algorithms/topics/算法-递归-回溯.md#牛客-0043-没有重复项数字的全排列-中等-2022-03)
- [`牛客 0046 加起来和为目标值的组合(二) (中等, 2022-03)`](algorithms/topics/算法-递归-回溯.md#牛客-0046-加起来和为目标值的组合二-中等-2022-03)
- [`牛客 0047 数独 (较难, 2022-03)`](algorithms/topics/算法-递归-回溯.md#牛客-0047-数独-较难-2022-03)

</details>

<details><summary><b> 算法-递归、迭代 [16] <a href="algorithms/topics/算法-递归、迭代.md">¶</a></b></summary>

- [`LeetCode 0021 合并两个有序链表 (简单, 2021-10)`](algorithms/topics/算法-递归、迭代.md#leetcode-0021-合并两个有序链表-简单-2021-10)
- [`LeetCode 0104 二叉树的最大深度 (简单, 2021-10)`](algorithms/topics/算法-递归、迭代.md#leetcode-0104-二叉树的最大深度-简单-2021-10)
- [`剑指Offer 0600 从尾到头打印链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-0600-从尾到头打印链表-简单-2021-11)
- [`剑指Offer 1600 数值的整数次方（快速幂） (中等, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-1600-数值的整数次方快速幂-中等-2021-11)
- [`剑指Offer 1900 正则表达式匹配 (困难, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-1900-正则表达式匹配-困难-2021-11)
- [`剑指Offer 2400 反转链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-2400-反转链表-简单-2021-11)
- [`剑指Offer 2400 反转链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-2400-反转链表-简单-2021-11)
- [`剑指Offer 2500 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-2500-合并两个排序的链表-简单-2021-11)
- [`剑指Offer 2500 合并两个排序的链表 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-2500-合并两个排序的链表-简单-2021-11)
- [`剑指Offer 2600 树的子结构 (中等, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-2600-树的子结构-中等-2021-11)
- [`剑指Offer 2700 二叉树的镜像 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-2700-二叉树的镜像-简单-2021-11)
- [`剑指Offer 2800 对称的二叉树 (简单, 2021-11)`](algorithms/topics/算法-递归、迭代.md#剑指offer-2800-对称的二叉树-简单-2021-11)
- [`剑指Offer 3600 二叉搜索树与双向链表 (中等, 2021-12)`](algorithms/topics/算法-递归、迭代.md#剑指offer-3600-二叉搜索树与双向链表-中等-2021-12)
- [`剑指Offer 6400 求1~n的和 (中等, 2022-01)`](algorithms/topics/算法-递归、迭代.md#剑指offer-6400-求1n的和-中等-2022-01)
- [`牛客 0039 N皇后问题 (较难, 2022-03)`](algorithms/topics/算法-递归、迭代.md#牛客-0039-n皇后问题-较难-2022-03)
- [`牛客 0067 汉诺塔问题 (中等, 2022-03)`](algorithms/topics/算法-递归、迭代.md#牛客-0067-汉诺塔问题-中等-2022-03)

</details>


</details>

---

Notes
---
- 算法
    - 二叉树
    - [动态规划](notes/算法/动态规划)
        - [从暴力递归到动态规划](notes/算法/动态规划/从暴力递归到动态规划)
        - [自底向上的递归技巧（树形DP）](notes/算法/动态规划/自底向上的递归技巧（树形DP）)
- 机器学习
    - [基础知识](notes/机器学习/基础知识)
    - GBDT
    - [样本不平衡](notes/机器学习/样本不平衡)
- 深度学习
    - 基础知识
        - [Attention](notes/深度学习/基础知识/Attention)
        - [CNN](notes/深度学习/基础知识/CNN)
        - [RNN](notes/深度学习/基础知识/RNN)
        - [损失函数](notes/深度学习/基础知识/损失函数)
        - [正则化](notes/深度学习/基础知识/正则化)
            - [过拟合与欠拟合](notes/深度学习/基础知识/正则化/过拟合与欠拟合)
        - [激活函数](notes/深度学习/基础知识/激活函数)
    - [Transformer](notes/深度学习/Transformer)
    - [对比学习](notes/深度学习/对比学习)
- 自然语言处理
    - [预训练语言模型](notes/自然语言处理/预训练语言模型)
        - Bert
        - [Transformer系列模型](notes/自然语言处理/预训练语言模型/Transformer系列模型)
        - [词向量](notes/自然语言处理/预训练语言模型/词向量)
    - 关键词挖掘
    - 实体链接
    - 小样本学习
        - [提示学习（Prompt）](notes/自然语言处理/-小样本学习/提示学习（Prompt）)
    - 文本生成
    - 细粒度情感分析
    - [Prompt](notes/自然语言处理/Prompt)
- 搜索、广告、推荐
- 深度学习框架
    - Pytorch
- 编程语言
    - Python
        - [设计模式](notes/编程语言/Python/设计模式)
        - [语法备忘](notes/编程语言/Python/语法备忘)
        - [常见报错记录](notes/编程语言/Python/常见报错记录)
    - Java
    - CCpp
- 计算机基础
    - [RPC基础](notes/计算机基础/RPC基础)

---

Papers
---
- 实体链接

---

Books
---
- [图解人工智能](books/图解人工智能)
- [面向对象是怎样工作的（第二版）](books/面向对象是怎样工作的（第二版）)

---

Codes
---
### Work Utils [¶](src/README.md#work-utils)

- [`find_best_threshold: 搜索最佳阈值（二分类）`](src/README.md#find_best_threshold-搜索最佳阈值二分类)
- [`BertTokenizer: Bert 分词器`](src/README.md#berttokenizer-bert-分词器)
- [`ner_result_parse: NER 结果解析（基于 BIO 格式）`](src/README.md#ner_result_parse-ner-结果解析基于-bio-格式)
- [`split: 将数据按比例切分`](src/README.md#split-将数据按比例切分)
- [`XLSHelper: Excel 文件加载（基于 openpyxl）`](src/README.md#xlshelper-excel-文件加载基于-openpyxl)
- [`ImageCheck: 图片完整性检查`](src/README.md#imagecheck-图片完整性检查)
- [`get_real_ext: 获取图像文件的真实后缀`](src/README.md#get_real_ext-获取图像文件的真实后缀)


### Pytorch Models [¶](src/README.md#pytorch-models)

- [`DualNet: 双塔结构`](src/README.md#dualnet-双塔结构)
- [`SiameseNet: 孪生网络，基于双塔结构`](src/README.md#siamesenet-孪生网络基于双塔结构)
- [`SimCSE: SimCSE`](src/README.md#simcse-simcse)
- [`Bert: Bert by Pytorch`](src/README.md#bert-bert-by-pytorch)
- [`BertPretrain: Bert 预训练（MLM + NSP）`](src/README.md#bertpretrain-bert-预训练mlm-nsp)
- [`SentenceBert: Bert 句向量`](src/README.md#sentencebert-bert-句向量)
- [`BertSequenceTagging: Bert 序列标注`](src/README.md#bertsequencetagging-bert-序列标注)
- [`BertTextClassification: Bert 文本分类`](src/README.md#berttextclassification-bert-文本分类)
- [`SENet1D: SENETLayer used in FiBiNET.`](src/README.md#senet1d-senetlayer-used-in-fibinet)
- [`LayerNorm: Layer Normalization`](src/README.md#layernorm-layer-normalization)


### Pytorch Utils [¶](src/README.md#pytorch-utils)

- [`DictTensorDataset: 字典格式的 Dataset`](src/README.md#dicttensordataset-字典格式的-dataset)
- [`ToyDataLoader: 简化创建 DataLoader 的过程`](src/README.md#toydataloader-简化创建-dataloader-的过程)
- [`BertDataLoader: 简化 Bert 训练数据的加载`](src/README.md#bertdataloader-简化-bert-训练数据的加载)
- [`ContrastiveLoss: 对比损失（默认距离函数为欧几里得距离）`](src/README.md#contrastiveloss-对比损失默认距离函数为欧几里得距离)
- [`CrossEntropyLoss: 交叉熵`](src/README.md#crossentropyloss-交叉熵)
- [`TripletLoss: Triplet 损失，常用于无监督学习、few-shot 学习`](src/README.md#tripletloss-triplet-损失常用于无监督学习few-shot-学习)
- [`FGM: Fast Gradient Method (对抗训练)`](src/README.md#fgm-fast-gradient-method-对抗训练)
- [`PGM: Projected Gradient Method (对抗训练)`](src/README.md#pgm-projected-gradient-method-对抗训练)
- [`Mixup: mixup 数据增强策略`](src/README.md#mixup-mixup-数据增强策略)
- [`Trainer: Trainer 基类`](src/README.md#trainer-trainer-基类)
- [`set_seed: 设置全局随机数种子，使实验可复现`](src/README.md#set_seed-设置全局随机数种子使实验可复现)
- [`init_weights: 默认参数初始化`](src/README.md#init_weights-默认参数初始化)


### Python Utils [¶](src/README.md#python-utils)

- [`simple_argparse: 一个简化版 argparse`](src/README.md#simple_argparse-一个简化版-argparse)
- [`ArrayDict: 数组字典，支持 slice`](src/README.md#arraydict-数组字典支持-slice)
- [`ValueArrayDict: 数组字典，支持 slice，且操作 values`](src/README.md#valuearraydict-数组字典支持-slice且操作-values)
- [`BunchDict: 基于 dict 实现 Bunch 模式`](src/README.md#bunchdict-基于-dict-实现-bunch-模式)
- [`FieldBunchDict: 基于 dataclass 的 BunchDict`](src/README.md#fieldbunchdict-基于-dataclass-的-bunchdict)
- [`ls_dir_recur: 递归遍历目录下的所有文件`](src/README.md#ls_dir_recur-递归遍历目录下的所有文件)
- [`files_concat: 文件拼接`](src/README.md#files_concat-文件拼接)
- [`get_caller_name: 获取调用者的名称`](src/README.md#get_caller_name-获取调用者的名称)
- [`function_test_dn: 函数测试装饰器`](src/README.md#function_test_dn-函数测试装饰器)


---

