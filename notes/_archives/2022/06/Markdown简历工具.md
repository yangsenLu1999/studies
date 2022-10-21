Markdown 简历工具
===
<!--START_SECTION:badge-->

![last modify](https://img.shields.io/static/v1?label=last%20modify&message=2022-10-15%2010%3A39%3A35&color=yellowgreen&style=flat-square)

<!--END_SECTION:badge-->

<!-- TOC -->
- [GitHub 项目](#github-项目)
    - [komomoo/vuepress-theme-resume](#komomoovuepress-theme-resume)
    - [CyC2018/Markdown-Resume](#cyc2018markdown-resume)
- [网页版](#网页版)
<!-- TOC -->

## GitHub 项目

### komomoo/vuepress-theme-resume
> [komomoo/vuepress-theme-resume: 🐈 书写简洁优雅的前端程序员 markdown 简历，由 vuepress 驱动](https://github.com/komomoo/vuepress-theme-resume)

使用步骤
```shell
# 安装 Node.js ~= 16.x；安装 18.x 会报错，也可以通过 n 模块切换 node 版本
# Ubuntu, 其他发行版参考：https://github.com/nodesource/distributions/blob/master/README.md#installation-instructions
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt-get install -y nodejs
node -v
# v16.17.1

# 在不使用 sudo 的情况下全局下载一个包
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
# 添加环境变量
export PATH=~/.npm-global/bin:$PATH

# 安装 yarn
npm install --global yarn

# git clone 仓库
git clone https://github.com/komomoo/vuepress-theme-resume.git
cd vuepress-theme-resume

# git remote 自己的仓库地址，因为 fork 的仓库不能设为私有仓库，故采用这种方式
git remote set-url origin git@github.com:imhuay/vuepress-theme-resume.git
git push

# 添加原仓库地址，以便更新
git remote add author git@github.com:komomoo/vuepress-theme-resume.git
git pull author master

# 安装依赖包
yarn # 或 npm i

# 开始
yarn dev # 或 npm run dev

# 编辑位置：example/README.md
```

### CyC2018/Markdown-Resume
> [CyC2018/Markdown-Resume: ⭐️ Markdown 简历模版](https://github.com/CyC2018/Markdown-Resume)


## 网页版
- [冷熊简历](http://cv.ftqq.com/)
- [木及简历](https://www.mujicv.com/)