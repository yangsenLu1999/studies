#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time:
    2022-08-06 16:44
Author:
    huayang (imhuay@163.com)
Subject:
    custom_docker
"""
import os
import sys
import json
import time
import doctest

from typing import *
from pathlib import Path
from itertools import islice
from collections import defaultdict

src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.append(str(src_path))
    import huaytools  # noqa
    from huaytools.utils import StrUtils
else:
    raise Exception


class CustomDockerFile:
    """"""
    docker_file_temp = r'''
    FROM {base_docker}
    
    
    '''

    save_dir = Path(r'./docker_files')
    save_dir.mkdir(exist_ok=True)

    def save(self):
        """"""


class __Test:
    """"""

    def __init__(self):
        """"""
        for k, v in self.__class__.__dict__.items():
            if k.startswith('_test') and isinstance(v, Callable):
                print(f'\x1b[32m=== Start "{k}" {{\x1b[0m')
                start = time.time()
                v(self)
                print(f'\x1b[32m}} End "{k}" - Spend {time.time() - start:3f}s===\x1b[0m\n')

    def _test_doctest(self):  # noqa
        """"""
        import doctest
        doctest.testmod()

    def _test_xxx(self):  # noqa
        """"""
        # print(f'{Path(__file__).parent.parent / "src"}')
        print(BuildDockerFile.docker_file_temp)
        print(StrUtils.remove_min_indent(BuildDockerFile.docker_file_temp))


if __name__ == '__main__':
    """"""
    __Test()
