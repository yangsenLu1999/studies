WSL 使用记录
===

<!-- TOC -->
- [备忘](#备忘)
- [配置 Python 开发环境](#配置-python-开发环境)
    - [[Linux] 安装 anaconda](#linux-安装-anaconda)
    - [Pycharm 安装](#pycharm-安装)
- [参考文档](#参考文档)
- [FAQ](#faq)
    - [fatal: unable to access 'https://github.com/xxx.git'](#fatal-unable-to-access-httpsgithubcomxxxgit)
<!-- TOC -->


## 备忘
- 从 Linux 访问 Windows 路径：`ls /mnt/d/path`（`d` 为盘符）；
- 从 Windows 访问 Linux 路径：`ls \\wsl$\Ubuntu-20.04\path` 或 `ls \\wsl.localhost\Ubuntu-20.04\path`
- 


## 配置 Python 开发环境

### [Linux] 安装 anaconda
> https://repo.anaconda.com/archive
```shell
$ wget https://repo.anaconda.com/archive/Anaconda3-20xx.xx-Linux-x86_64.sh
$ sh Anaconda3-20xx.xx-Linux-x86_64.sh
# 注意这里不要 sudo sh，否则会安装到 root 目录下
# 安装完成后
$ source /home/huay/.bashrc
$ which python
```

<details><summary><b> 安装信息 </b></summary>

```shell
Welcome to Anaconda3 2022.05

...

Do you accept the license terms? [yes|no]
[no] >>> yes

Anaconda3 will now be installed into this location:
/home/huay/anaconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home/huay/anaconda3] >>>
PREFIX=/home/huay/anaconda3
Unpacking payload ...

...

## Package Plan ##

  environment location: /home/huay/anaconda3

  added / updated specs:
    ...

...

installation finished.
Do you wish the installer to initialize Anaconda3
by running conda init? [yes|no]
[no] >>> yes
no change     /home/huay/anaconda3/condabin/conda
no change     /home/huay/anaconda3/bin/conda
no change     /home/huay/anaconda3/bin/conda-env
no change     /home/huay/anaconda3/bin/activate
no change     /home/huay/anaconda3/bin/deactivate
no change     /home/huay/anaconda3/etc/profile.d/conda.sh
no change     /home/huay/anaconda3/etc/fish/conf.d/conda.fish
no change     /home/huay/anaconda3/shell/condabin/Conda.psm1
no change     /home/huay/anaconda3/shell/condabin/conda-hook.ps1
no change     /home/huay/anaconda3/lib/python3.9/site-packages/xontrib/conda.xsh
no change     /home/huay/anaconda3/etc/profile.d/conda.csh
modified      /home/huay/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

If you'd prefer that conda's base environment not be activated on startup,
   set the auto_activate_base parameter to false:

conda config --set auto_activate_base false

Thank you for installing Anaconda3!
```

</details>

### Pycharm 安装
- 社区版无法调用 WSL 上的解释器，只能使用 Windows 上的；可能会降低性能；
- 专业版参考：[PyCharm WSL2 下开发调试_wslynn的博客](https://blog.csdn.net/qq_38992249/article/details/122387097)


## 参考文档
- [WSL 的手动安装步骤 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)
    > 这里推荐手动分步安装；文档推荐的 [安装方式](https://learn.microsoft.com/zh-cn/windows/wsl/install)（首次安装）可能会在下载 Linux 发行版的时候卡住，而手动从应用商店下载的速度则非常快；
- [设置 WSL 开发环境 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/wsl/setup/environment)


## FAQ

### fatal: unable to access 'https://github.com/xxx.git'
> [git clone 时用 https 的方式报错 - CSDN博客](https://blog.csdn.net/wang2008start/article/details/118967723)
```shell
$ git config --global https.sslVerify "true"
```
