WSL2 使用记录
===
<!--START_SECTION:badge-->

![last modify](https://img.shields.io/static/v1?label=last%20modify&message=2022-12-09%2016%3A14%3A59&color=yellowgreen&style=flat-square)

<!--END_SECTION:badge-->

<!-- TOC -->
- [安装 WSL2](#安装-wsl2)
- [开发环境](#开发环境)
    - [官方推荐设置](#官方推荐设置)
    - [Python 环境](#python-环境)
        - [安装 anaconda](#安装-anaconda)
        - [安装 Pycharm 专业版](#安装-pycharm-专业版)
    - [安装 zsh (可选)](#安装-zsh-可选)
- [WSL 常用操作](#wsl-常用操作)
- [FAQ](#faq)
    - [fatal: unable to access 'https://github.com/xxx.git'](#fatal-unable-to-access-httpsgithubcomxxxgit)
<!-- TOC -->


## 安装 WSL2
> [WSL 的手动安装步骤 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)
>> 这里推荐分步安装；官方推荐的 [首次安装](https://learn.microsoft.com/zh-cn/windows/wsl/install) 可能会在下载 Linux 发行版的时候卡住，而手动从应用商店下载的速度则非常快；

## 开发环境

### 官方推荐设置
- [设置 Windows Terminal | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/wsl/setup/environment#set-up-windows-terminal)
- [使用 Visual Studio Code | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/wsl/setup/environment#use-visual-studio-code)
- [基本 WSL 命令 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/wsl/setup/environment#basic-wsl-commands)

### Python 环境

#### 安装 anaconda
> 选择需要安装的 anaconda 版本: https://repo.anaconda.com/archive
```shell
# 下载指定版本的 anaconda
$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh

# 安装, 默认安装到 /home/<user>/anaconda3
$ sh Anaconda3-20xx.xx-Linux-x86_64.sh
# 如果使用 sudo sh，则会安装到 root 目录下

# 安装完成后激活环境
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


#### 安装 Pycharm 专业版
- PyCharm 本身是安装在 Windows 环境，这里的目的是通过 Pycharm 在 WSL 上开发；
- 社区版无法调用 WSL 上的解释器，但可以使用 Windows 端的解释器 (可能会降低性能)；
- 专业版调用 WSL：[PyCharm WSL2 下开发调试_wslynn的博客](https://blog.csdn.net/qq_38992249/article/details/122387097)
- 破解版安装：[PyCharm 破解教程 (持续更新~) - 异常教程](https://www.exception.site/essay/how-to-free-use-pycharm-2020)


### 安装 zsh (可选)
> [Installing ZSH · ohmyzsh/ohmyzsh Wiki](https://github.com/ohmyzsh/ohmyzsh/wiki/Installing-ZSH) 
>> 可能会导致部分设置失效

```shell
$ sudo apt install zsh  # install
$ zsh --version  # 
$ chsh -s $(which zsh)  # set default
$ echo $SHELL
```
- 配置 conda 环境变量
    ```shell
    # vim ~/.zshrc
    export PATH="/home/huay/anaconda3/bin:$PATH"
    conda activate
    ```
    或者
    ```shell
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/home/huay/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/home/huay/anaconda3/etc/profile.d/conda.sh" ]; then
            . "/home/huay/anaconda3/etc/profile.d/conda.sh"
        else
            export PATH="/home/huay/anaconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
    ```
- 修改主题
    ```shell
    vim ~/.zshrc # 修改 ZSH_THEME="robbyrussell"
    ```
- 推荐主题
    > [Themes · ohmyzsh/ohmyzsh Wiki](https://github.com/ohmyzsh/ohmyzsh/wiki/Themes)
    - [candy](https://github.com/ohmyzsh/ohmyzsh/wiki/Themes#candy)


## WSL 常用操作

- Windows 在 WSL 中的路径：`/mnt/d/<win_path>`（`d` 为盘符）
- WSL 在 Windows 中的路径：`\\wsl$\Ubuntu-20.04\<wsl_path>`
- 在 VSCode 中打开
    ```bash
    cd <wsl_path>
    code .
    ```
- 在 Windows 打开
    ```bash
    cd <wsl_path>
    explorer.exe .
    ``` 


## FAQ

### fatal: unable to access 'https://github.com/xxx.git'
> [git clone 时用 https 的方式报错 - CSDN博客](https://blog.csdn.net/wang2008start/article/details/118967723)
```shell
$ git config --global https.sslVerify "true"
```
