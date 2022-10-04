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

from huaytools.utils import get_logger
from github import Github

logger = get_logger()

try:
    fp = Path(__file__)
    sys.path.insert(0, f'{fp.parent.parent / "src"}')
except:  # noqa
    exit(1)
else:
    from readme.args import args
    from readme.algorithms import Algorithms
    from readme.utils import ReadmeUtils


class ENV:
    """"""
    github_token = ''


class BuildReadme:
    """"""

    def __init__(self):
        """"""
        self.fp_repo = args.fp_repo
        self.fp_repo_readme_main = args.fp_repo_readme_main
        self.algo = Algorithms()

    def _update_homepage(self):
        """"""
        with open(self.fp_repo_readme_main, encoding='utf8') as f:
            tmp = f.read()

        readme_homepage = tmp.format(readme_algorithms=self.algo.readme)
        with open(args.fp_repo_readme, 'w', encoding='utf8') as f:
            f.write(readme_homepage)

    def build(self):
        # build algorithms
        self.algo.build()

        # last
        self._update_homepage()


if __name__ == '__main__':
    """"""
    br = BuildReadme()
    br.build()
    logger.info(f'Update README Success!')
