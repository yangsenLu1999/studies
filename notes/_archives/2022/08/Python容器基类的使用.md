Python 容器基类的使用
===

容器基类列表：[collections.abc — Python 文档](https://docs.python.org/zh-cn/3/library/collections.abc.html)


## 使用场景

### 判断一个具体类或实例是否具有某一特定的接口

```python
from typing import *

# 判断能否 len(obj)，即判断是否实现了 __len__
isinstance(obj, Sized)

# 判断能否 obj[index]
isinstance(obj, Sequence)

# 判断是否可哈希
issubclass(cls, Hashable)
```