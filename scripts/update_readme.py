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
import subprocess

from typing import *
from pathlib import Path
from itertools import islice
from collections import defaultdict

from huaytools.utils import get_logger

# from github import Github

logger = get_logger()

try:
    fp = Path(__file__)
    sys.path.insert(0, f'{fp.parent.parent / "src"}')
except:  # noqa
    exit(1)
else:
    from readme.algorithms import Algorithms
    from readme.notes import Notes
    from readme.utils import ReadmeUtils, args


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
        self.note = Notes()

        # process

    commit_info = 'Auto-Update README'

    def git_push(self):
        os.system('git add -u')
        code, data = subprocess.getstatusoutput(f'git commit -m "{self.commit_info}"')
        if code == 0:
            # os.system('git push')
            data = subprocess.getoutput('git push')
            print(data)
        else:
            print(data)

    def build(self):
        # build algorithms
        self.algo.build()
        self.note.build()

        # last
        self._update_homepage()

    def _update_homepage(self):
        """"""
        with open(self.fp_repo_readme_main, encoding='utf8') as f:
            tmp = f.read()

        readme_homepage = tmp.format(toc_algorithms=self.algo.readme_toc,
                                     toc_notes=self.note.readme_toc,
                                     toc_recent=self.note.repo_recent_toc,
                                     readme_algorithms=self.algo.readme_concat,
                                     readme_notes=self.note.readme_concat)
        with open(args.fp_repo_readme, 'w', encoding='utf8') as f:
            f.write(readme_homepage)


if __name__ == '__main__':
    """"""
    br = BuildReadme()
    br.build()
    br.git_push()
    # logger.info(f'Update README Success!')
