#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time:
    2022-08-03 10:59
Author:
    HuaYang (imhuay@163.com)
Subject:
    A simple implementation of a tire tree
"""
import os
import sys
import json
import time

from typing import *
from pathlib import Path
from collections import defaultdict


class TireNode:
    """"""

    def __init__(self):
        """"""
        self.nodes = dict()
        self.is_end = False
        self.count = 0


class TireTree(TireNode):

    def insert(self, seq: Sequence):
        """"""
        cur = self
        for p in seq:
            if not cur.nodes.get(p, None):
                node = TireNode()
                cur.nodes[p] = node
            cur = cur.nodes[p]
        cur.is_end = True
        self.count += 1

    def search(self, seq: Sequence):
        """"""
        cur = self
        for p in seq:
            cur = cur.nodes.get(p, None)
            if cur is None:
                return False
        return cur.is_end

    def find_all_subseq(self, seq: Sequence) -> List[List]:
        ret = []
        for start_idx in range(len(seq)):
            sub_ret = []
            cur = self
            seq_suf = seq[start_idx:]
            for i, p in enumerate(seq_suf):
                cur: TireNode = cur.nodes.get(p, None)
                if cur is None:
                    break
                if cur.is_end:
                    sub_ret.append(tuple(seq_suf[:i + 1]))

            if sub_ret:
                ret.append(sub_ret)
        return ret


class __Test:
    """"""

    def __init__(self):
        """"""
        for k, v in self.__class__.__dict__.items():
            if k.startswith('_test') and isinstance(v, Callable):
                print(f'=== Start "{k}" {{')
                start = time.time()
                v(self)
                print(f'}} End "{k}" - Spend {time.time() - start:f}s===\n')

    def _test_doctest(self):  # noqa
        """"""
        import doctest
        doctest.testmod()

    def _test_xxx(self):  # noqa
        """"""
        phrases = ['apple', 'apple inc', 'nlp', 'machine', 'machine learning']
        tire = TireTree()
        for p in phrases:
            tokens = p.split()
            tire.insert(tokens)

        assert tire.search('apple inc'.split())

        txt = "I'm studying machine learning ."
        ret = tire.find_all_subseq(txt.split())
        print(ret)


if __name__ == '__main__':
    """"""
    __Test()
