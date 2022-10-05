<!-- <div style="text-align: center"> -->
<div align="center">  <!-- style="text-align: center" 在 GitHub 主页不生效 -->
<!-- cacheSeconds 统一为 3600 秒 -->

# Keep on Your Studying!

[![wakatime](https://wakatime.com/badge/user/c840568d-e4b1-4c63-ade0-03856283d319.svg)](https://wakatime.com/@c840568d-e4b1-4c63-ade0-03856283d319)
[![GitHub issues](https://img.shields.io/github/issues/imhuay/studies?color=important&cacheSeconds=3600)](https://github.com/imhuay/studies/issues)
[![GitHub closed issues](https://img.shields.io/github/issues-closed-raw/imhuay/studies?color=inactive&cacheSeconds=3600)](https://github.com/imhuay/studies/issues?q=is:issue+is:closed)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/imhuay/studies?color=active&cacheSeconds=3600)](https://github.com/imhuay/studies/commits/master)
[![](https://visitor-badge.laobi.icu/badge?page_id=imhuay.studies&right_color=green&left_text=page%20views)](https://visitor-badge.laobi.icu)

<!-- 
![clones](https://raw.githubusercontent.com/imhuay/imhuay/traffic/traffic-studies/clones.svg)
![clones per week](https://raw.githubusercontent.com/imhuay/imhuay/traffic/traffic-studies/clones_per_week.svg)
![views](https://raw.githubusercontent.com/imhuay/imhuay/traffic/traffic-studies/views.svg)
![views per week](https://raw.githubusercontent.com/imhuay/imhuay/traffic/traffic-studies/views_per_week.svg)
[![GitHub last commit](https://img.shields.io/github/last-commit/imhuay/studies?cacheSeconds=3600)](https://github.com/imhuay/studies/commits)
[![GitHub Repo stars](https://img.shields.io/github/stars/imhuay/studies?style=social)](https://github.com/imhuay/studies/stargazers)
-->

</div>

## Index

{toc_algorithms}
{toc_notes}

---

{readme_algorithms}

{readme_notes}

<!-- <summary><b> TODO </b></summary> -->
### TODO

- [ ] 给 algorithms/notes 添加 README，TOC 树形目录；
- [ ] 尝试 GitHub 提供的 projects 栏：参考 [Projects · zhaoyan346a/Blog](https://github.com/zhaoyan346a/Blog/projects)
- [ ] 重构 README 生成的 Algorithms 和 Codes 两个类，并迁移至 tools 目录。
- [ ] 优化主页 README 下的 Algorithms 链接，调整为层级目录的形式（类似 Notes）

<!-- - [ ] 【`2021.11.11`】pytorch_trainer: 为 EvaluateCallback 添加各种预定义评估指标，如 acc、f1 等，目前只有 loss； -->
<!-- - [ ] 【`2021.11.11`】论文：What does BERT learn about the structure of language? —— Bert 各层的含义； -->
<!-- - [ ] 【`2021.11.10`】bert-tokenizer 自动识别 `[MASK]` 等特殊标识； -->
<!-- - [ ] 【`2021.11.07`】面试笔记：通识问题/项目问题 -->
<!-- - [ ] 【`2021.10.22`】max_batch_size 估算 -->

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
