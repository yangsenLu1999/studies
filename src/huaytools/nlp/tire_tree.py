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

NodeValue = TypeVar(Any)


class TireNode:
    """"""

    def __init__(self, value):
        """"""
        self.value: NodeValue = value
        self.nodes: Dict[NodeValue, 'TireNode'] = dict()
        self.is_end: bool = False
        self.count: int = 0

    @property
    def nodes_count(self) -> int:
        return len(self.nodes)


class TireTree(TireNode):
    # info field name
    _F_SEQ = 'seq'  # 至当前节点的序列，下简称序列
    _F_IS_END = 'is_end'  # 序列是否是一个完整的序列
    _F_COUNT = 'count'  # 序列在语料中出现的计数
    _F_SUB_NODE_COUNT = 'sub_node_count'  # 直接子节点的计数
    _F_ALL_SUB_NODE_COUNT = 'all_sub_node_count'  # 全部子节点的计数

    _default_info = {
        _F_SEQ: tuple(),
        _F_IS_END: False,
        _F_COUNT: 0,
        _F_SUB_NODE_COUNT: 0,
        _F_ALL_SUB_NODE_COUNT: 0,
    }

    # 遍历结果
    _traversal: List = None

    def __init__(self):
        super().__init__(value=None)

    def insert(self, seq: Sequence):
        """"""
        cur = self
        for it in seq:
            if not cur.nodes.get(it, None):
                node = TireNode(it)
                cur.nodes[it] = node
            cur = cur.nodes[it]
        cur.is_end = True
        cur.count += 1

    def search(self, seq: Sequence):
        """"""
        cur = self
        for p in seq:
            cur = cur.nodes.get(p, None)
            if cur is None:
                return False
        return cur.is_end

    def find_all_subseq(self, seq: Sequence) -> List[List]:
        """找出所有完整的子串"""
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

    @property
    def traversal(self) -> List[Dict]:
        """遍历所有节点，记录相关信息"""
        if self._traversal is None:
            self._traversal = self._traverse()
        return self._traversal

    def _traverse(self):
        _traversal = []

        def dfs(node: TireNode, seq: List):
            if not node:
                return self._default_info

            info = self._default_info.copy()
            _traversal.append(info)

            info[TireTree._F_COUNT] = node.count
            info[TireTree._F_IS_END] = node.is_end
            info[TireTree._F_SUB_NODE_COUNT] = len(node.nodes)
            info[TireTree._F_ALL_SUB_NODE_COUNT] = len(node.nodes)

            for k, v in node.nodes.items():
                seq.append(v.value)
                sub_info = dfs(v, seq)
                sub_info[TireTree._F_SEQ] = tuple(seq)
                info[TireTree._F_ALL_SUB_NODE_COUNT] += sub_info[TireTree._F_ALL_SUB_NODE_COUNT]
                seq.pop()

            return info

        dfs(self, [])
        return _traversal


class __Test:
    """"""

    def __init__(self):
        """"""
        phrases = ['a b', 'a b c', 'a b d', 'a b c',
                   'apple', 'apple inc',
                   'nlp',
                   'machine', 'machine learning', 'machine learning', 'machine learning system',
                   'machine system']
        tire = TireTree()
        for p in phrases:
            tokens = p.split()
            tire.insert(tokens)

        self.tire = tire

        # try:
        #     from huaytools.utils import cprint
        # except ImportError:
        #     cprint = print

        for k, v in self.__class__.__dict__.items():
            if k.startswith('_test') and isinstance(v, Callable):
                print(f'\x1b[31m=== Start "{k}" {{\x1b[0m')
                start = time.time()
                v(self)
                print(f'\x1b[31m}} End "{k}" - Spend {time.time() - start:f}s===\x1b[0m\n')

    def _test_doctest(self):  # noqa
        """"""
        import doctest
        doctest.testmod()

    def _test_find_all_subseq(self):  # noqa
        """"""
        tire = self.tire
        assert tire.search('apple inc'.split())

        txt = "I'm studying machine learning ."
        ret = tire.find_all_subseq(txt.split())
        print(ret)

    def _test_traversal(self):
        """"""
        for it in self.tire.traversal:
            print(it)


if __name__ == '__main__':
    """"""
    __Test()
