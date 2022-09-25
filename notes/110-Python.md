Python
===

<!-- TOC -->
- [Python 标准库](#python-标准库)
    - [容器的抽象基类 (`collections.abc`)](#容器的抽象基类-collectionsabc)
    - [数据类 (`dataclass`)](#数据类-dataclass)
    - [装饰器](#装饰器)
    - [备忘*](#备忘)
- [工具](#工具)
    - [Pycharm](#pycharm)
- [常见问题](#常见问题)
    - [Python 项目构建](#python-项目构建)
    - [更换国内镜像](#更换国内镜像)
    - [Solving environment: failed](#solving-environment-failed)
- [学习资料](#学习资料)
<!-- TOC -->

## Python 标准库
> [Python 标准库 — Python 文档](https://docs.python.org/zh-cn/3/library/index.html)

### 容器的抽象基类 (`collections.abc`)
> [容器的抽象基类 — Python 文档](https://docs.python.org/zh-cn/3/library/collections.abc.html#collections-abstract-base-classes)  

- 快速查询容器之间的继承关系，以及包含的抽象方法；
- **使用场景**：type hints、`isinstance()`、`issubclass()` 等；
    > 详见：[容器基类的使用](./_archives/2022/08/Python容器基类的使用.md)

### 数据类 (`dataclass`)
- [dataclass 使用记录](./_archives/2022/09/python-dataclass使用记录.md)

### 装饰器
- [装饰器的本质](./_archives/2022/05/python装饰器的本质.md)

### 备忘*
- [class method 中 `self` 的含义](./_archives/2022/06/python类方法中self的含义.md)
- [Python 函数声明中单独的正斜杠（/）和星号（*）是什么作用](./_archives/2022/07/python函数声明中单独的正斜杠和星号是什么意思.md)
- [类变量、成员变量，与注解](./_archives/2022/07/python类变量和成员变量的最佳实践)
- [requirements.txt 语法备忘](./_archives/2022/09/python-requirements语法.md)


## 工具

### Pycharm
- [Pycharm 常用配置](./_archives/2022/07/PyCharm常用配置.md)
    - [常用插件列表](./_archives/2022/07/PyCharm常用配置.md#常用插件)


## 常见问题

### Python 项目构建
- 

### 更换国内镜像
- [pip & conda 国内镜像源](./_archives/2022/06/python国内镜像源.md)

### Solving environment: failed
```shell
$ conda config --set channel_priority flexible
```


## 学习资料
- [Python 的好习惯与最佳实践 - 肥清哥哥的个人空间_bilibili](https://space.bilibili.com/374243420/channel/collectiondetail?sid=422655)
- [不基础的python基础 - 码农高天的个人空间_bilibili](https://space.bilibili.com/245645656/channel/collectiondetail?sid=346060)