Markdown 备忘
===

- [VSCode 插件](#vscode-插件)
    - [自动更新目录插件](#自动更新目录插件)
- [编辑](#编辑)
    - [元素居中](#元素居中)
    - [目录结构](#目录结构)
    - [换行](#换行)
    - [图片](#图片)
    - [隐藏块](#隐藏块)
    - [HTML 表格](#html-表格)
    - [Latex 语法](#latex-语法)
    - [上标引用](#上标引用)
    - [公式对齐](#公式对齐)


## VSCode 插件

### 自动更新目录插件
- 插件名称：Markdown All in One
- 插入目录快捷键： `Shift + Command + P` (Create Table of Contents)


## 编辑

### 元素居中
将需要居中的元素替换为 html 标签，然后加上 `style="text-align: center;"` 属性；
```html
<div style="text-align: center;">

</div>
```

### 目录结构
```txt
project
├── src/
│   ├── assets/                    // 静态资源目录
│   ├── common/                    // 通用类库目录
│   ├── components/                // 公共组件目录
│   ├── store/                     // 状态管理目录
│   ├── utils/                     // 工具函数目录
│   ├── App.vue
│   └── main.ts
├── tests/                         // 单元测试目录
├── index.html
└── package.json
```

### 换行
- 法1）在行末手动插入 `<br/>`；
- 法2）在行末添加两个空格；

### 图片

- 不带链接
    ```html
    <img src="./_assets/xxx.png" height="300" />
    ```
- 带链接
    ```html
    <a href=""><img src="./_assets/xxx.png" height="300" /></a>
    ```
    - `height`用于控制图片的大小，一般不使用，图片会等比例缩放；



### 隐藏块
```html
<details><summary><b>示例：动态序列（点击展开）</b></summary> 

// 代码块，注意上下都要保留空行

</details>
<br/> <!-- 如果间隔太小，可以加一个空行 -->
```


### HTML 表格
```html
<table style="width:80%; table-layout:fixed;">
    <tr>
        <th>普通卷积</td>
        <th>空洞卷积</td>
    </tr>
    <tr>
        <td><img width="250px" src="./res/no_padding_no_strides.gif"></td>
        <td><img width="250px" src="./res/dilation.gif"></td>
    </tr>
</table>
```

### Latex 语法
- 在 markdown 内使用：行内使用 `$` 包围，如 $a+b=3$；
- 独立代码块使用 `$$` 包围，如：
$$
    a+b=3
$$
- 更多语法见 [Latex 备忘](./LaTeX备忘.md)


### 上标引用
> 非标准用法，在编辑时不支持跳转，但是转换成 HTML 页面后可以；

示例：百度[$^{[1]}$](#ref1)是一种搜索引擎

<a name="ref1"> $[1]$ </a> [百度一下](https://www.baidu.com) <br/>

### 公式对齐

$$
\begin{aligned}
    a &= 1 \\
    b &= 2
\end{aligned}
$$