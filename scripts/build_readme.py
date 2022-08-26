#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time:
    2022-08-22 20:14
Author:
    huayang (imhuay@163.com)
Subject:
    build_readme
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


class BuildReadme:
    """"""
    D_SRC_NAME = 'src'
    D_NOTES_NAME = 'notes'

    cur_dir = Path(__file__).parent
    repo_path = cur_dir.parent
    src_path = repo_path / D_SRC_NAME
    sys.path.append(str(src_path))

    notes_path = repo_path / D_NOTES_NAME

    def __init__(self):
        """"""
        self.create_notes_toc()

    def create_notes_toc(self):
        """"""
        # load top notes markdowns
        fs = [p for p in self.notes_path.iterdir() if '-' in p.stem]
        fs = sorted(fs, key=lambda p: p.stem)
        # print(list(fs))

        toc_ls = []
        for p in fs:
            toc_ls.append(self._parse(p))

    def _parse(self, p: Path) -> str:
        """"""

    # def foo(self):
    #     from huaytools.utils import PythonUtils
    #     print(PythonUtils)


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
        br = BuildReadme()


if __name__ == '__main__':
    """"""
    __Test()
