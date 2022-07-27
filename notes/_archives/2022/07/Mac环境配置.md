Mac 环境配置
===

- [必备软件](#必备软件)
    - [Homebrew (brew)](#homebrew-brew)
- [常用工具](#常用工具)

## 必备软件

### Homebrew (brew)
> [Homebrew - 软件包管理工具](https://brew.sh/)

安装后需要手动添加环境变量（注意安装完成后的提示）
```shell
$ echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/huay/.zprofile
$ eval "$(/opt/homebrew/bin/brew shellenv)"
```

## 常用工具

```shell
$ brew install cmake
$ brew install pkgconfig
```
