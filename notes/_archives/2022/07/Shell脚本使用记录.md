Shell 脚本备忘
===

- [Tips](#tips)
    - [不使用 sh 命令，直接运行 shell 脚本的方法](#不使用-sh-命令直接运行-shell-脚本的方法)

---

## Tips

### 不使用 sh 命令，直接运行 shell 脚本的方法
> [How do I run a shell script without using "sh" or "bash" commands? - Stack Overflow](https://stackoverflow.com/questions/8779951/how-do-i-run-a-shell-script-without-using-sh-or-bash-commands)

1. 在脚本开始添加 `#!/bin/bash`；
2. 执行 `chmod u+x $script_path`；
3. （可选）添加环境变量，使脚本全局可用，`export PATH=$PATH:$script_directory`；
    > 如果不行添加，可以把脚本保存到 `/usr/local/bin`；
