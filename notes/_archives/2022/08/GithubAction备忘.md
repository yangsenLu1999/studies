Github Action 指南
===
- 官方文档：[Workflows - GitHub Docs](https://docs.github.com/cn/actions/using-workflows)

<!-- TOC -->
- [工作流程语法](#工作流程语法)
    - [触发器 - `on`](#触发器---on)
    - [作业 - `jobs`](#作业---jobs)
        - [执行环境 - `jobs.<job_id>.runs-on`](#执行环境---jobsjob_idruns-on)
        - [依赖执行 - `jobs.<job_id>.needs`](#依赖执行---jobsjob_idneeds)
        - [条件执行 - `jobs.<job_id>.if`](#条件执行---jobsjob_idif)
        - [矩阵策略 - `jobs.<job_id>.strategy.matrix`](#矩阵策略---jobsjob_idstrategymatrix)
        - [复用流程 - `jobs.<job_id>.uses`](#复用流程---jobsjob_iduses)
<!-- TOC -->


## 工作流程语法
> [GitHub Actions 的工作流程语法](https://docs.github.com/cn/actions/using-workflows/workflow-syntax-for-github-actions)

**示例**
```yml
name: learn-github-actions  # （可选）将出现在 GitHub 仓库的 Actions 选项卡中的工作流程名称。

on: [push]  # 指定此工作流程的触发器，更多细节：https://docs.github.com/cn/actions/using-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore

jobs:  # 该工作流的所有作业

  check-bats-version:  # 子作业的名称（自定义）
    runs-on: ubuntu-latest  # 该作业运行的环境，更多细节：https://docs.github.com/cn/actions/reference/workflow-syntax-for-github-actions#jobsjob_idruns-on
    steps:  # 该子作业的所有步骤
      - name: Checkout repository  # 该步骤的名称
        uses: actions/checkout@v3  # 复用 https://github.com/actions/checkout/tree/v3
      - name: Setup nodejs
        uses: actions/setup-node@v3  # 复用 https://github.com/actions/setup-node/tree/v3
        with:  # 附加选项
          node-version: '14'  # 设定 node 的版本
      - name: Install bats  
        run: npm install -g bats  # run 指示作业在运行器上执行命令
        run: bats -v

  build:  # 子作业
    runs-on: ubuntu-latest
    # 策略矩阵：通过创建变量来自动构建基于变量组合的多个作业
    strategy:
      matrix:  
        node: [12, 14, 16]
    steps:
      - uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node }}
    
  setup:  # 子作业
    runs-on: ubuntu-latest
    # 自作业之间默认是并行执行的，可以通过 needs 来指定依赖，更多细节：https://docs.github.com/cn/actions/using-jobs/using-jobs-in-a-workflow#defining-prerequisite-jobs
    needs: build  # 依赖 build 作业
    steps:
      - run: ./build_server.sh  # 相对于仓库主目录
```

### 触发器 - `on`
> [`on`](https://docs.github.com/cn/actions/using-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore)

### 作业 - `jobs`
> [`jobs`](https://docs.github.com/cn/actions/using-workflows/workflow-syntax-for-github-actions#jobs)

#### 执行环境 - `jobs.<job_id>.runs-on`
> [`jobs.<job_id>.runs-on`](https://docs.github.com/cn/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idruns-on)

#### 依赖执行 - `jobs.<job_id>.needs`
> [`jobs.<job_id>.needs`](https://docs.github.com/cn/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idneeds)

#### 条件执行 - `jobs.<job_id>.if`
> [`jobs.<job_id>.if`](https://docs.github.com/cn/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idif)

#### 矩阵策略 - `jobs.<job_id>.strategy.matrix`
> [`jobs.<job_id>.strategy.matrix`](https://docs.github.com/cn/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idstrategymatrix)  

#### 复用流程 - `jobs.<job_id>.uses`
> [`jobs.<job_id>.uses`](https://docs.github.com/cn/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_iduses)

