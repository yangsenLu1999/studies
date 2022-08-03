#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time:
    2022-08-03 16:45
Author:
    HuaYang (imhuay@163.com)
Subject:
    Print Utils
"""
import os
import sys
import json
import time

from typing import *
from pathlib import Path
from collections import defaultdict

COLOR = BACKCOLOR = Literal['black', 'red', 'green', 'yellow', 'blue', 'white', 'gray']
# https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
COLOR_MAPPING = {
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'white': 37,
    'gray': 90
}
MODE = Literal['normal', 'bold', 'light', 'italic', 'underline']
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_(Select_Graphic_Rendition)_parameters
BACKCOLOR_MAPPING = {
    'black': 40,
    'red': 41,
    'green': 42,
    'yellow': 43,
    'blue': 44,
    'white': 47,
    'gray': 100
}
MODE_MAPPING = {
    'normal': 0,
    'bold': 1,
}
SUFFIX = '\033[0m'


class PrintUtils:
    """"""

    @staticmethod
    def cprint(*args,
               color: Union[int, COLOR] = 'red',
               backcolor: Union[int, BACKCOLOR] = None,
               mode: Union[int, MODE] = 'normal',
               **kwargs):
        """
        基本格式：{\033[显示方式;前景色;背景色m}text{\033[0m}

        Args:
            *args:
            color: 前景色
            backcolor: 背景色
            mode: 显式模式
            **kwargs:

        References:
            - [Python：print显示颜色 - 博客园](https://www.cnblogs.com/hanfe1/p/10664942.html)
            - https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
            - https://en.wikipedia.org/wiki/ANSI_escape_code#Colors

        """
        prefix = '\033['
        if isinstance(color, str):
            color = COLOR_MAPPING[color]
        if mode is not None:
            if isinstance(mode, str):
                mode = MODE_MAPPING[mode]
            prefix += f'{mode};'
        prefix += f'{color}'
        if backcolor is not None:
            if isinstance(backcolor, str):
                backcolor = BACKCOLOR_MAPPING[backcolor]
            prefix += f';{backcolor}'
        prefix += 'm'

        if len(args) == 1:
            args = (f'{prefix}{args[0]}{SUFFIX}',)
        else:
            args = (f'{prefix}{args[0]}',) + args[1:-1] + (f'{args[-1]}{SUFFIX}',)
        print(*args, **kwargs)


cprint = PrintUtils.cprint


class __RunWrapper:
    """"""

    def __init__(self):
        """"""
        for k, v in self.__class__.__dict__.items():
            if k.startswith('demo') and isinstance(v, Callable):
                print(f'=== Start "{k}" {{')
                start = time.time()
                v(self)
                print(f'}} End "{k}" - Spend {time.time() - start:f}s===\n')

    def demo_doctest(self):  # noqa
        """"""
        import doctest
        doctest.testmod()

    def demo_xxx(self):  # noqa
        """"""
        cprint(1, '2', [3, 4], color='black', backcolor='white')
        cprint(1, '2', [3, 4])  # default red
        cprint(1, '2', [3, 4], color='green')
        cprint(1, '2', [3, 4], color='yellow')
        cprint(1, '2', [3, 4], color='blue')
        cprint(1, '2', [3, 4], color='white')
        cprint(1, '2', [3, 4], color='gray')


if __name__ == '__main__':
    """"""
    __RunWrapper()
