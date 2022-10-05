#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time:
    2022-10-03 23:53
Author:
    huayang (imhuay@163.com)
Subject:
    utils
"""
from __future__ import annotations

import os
# import sys
# import json
# import unittest

# from typing import *
# from collections import defaultdict
from pathlib import Path

from huaytools.utils import get_logger

logger = get_logger()


class ReadmeUtils:
    GIT_ADD_TEMP = 'git add "{fp}"'

    @staticmethod
    def git_add(fp: Path):
        """不再使用，通过 git add -u 代替"""
        command = ReadmeUtils.GIT_ADD_TEMP.format(fp=fp.resolve())
        code = os.system(command)
        ReadmeUtils._git_log(code, command)

    GIT_MV_TEMP = 'git mv "{old_fp}" "{new_fp}"'

    @staticmethod
    def git_mv(old_fp: Path, new_fp: Path):
        command = ReadmeUtils.GIT_MV_TEMP.format(old_fp=old_fp.resolve(),
                                                 new_fp=new_fp.resolve())
        code = os.system(command)
        ReadmeUtils._git_log(code, command)

    @staticmethod
    def _git_log(code, command):
        if code == 0:
            logger.info(command)
        else:
            logger.error(command)


class args:  # noqa
    """"""
    _fp_cur_file = Path(__file__)
    # repo
    fp_repo = Path(_fp_cur_file.parent / '../..').resolve()
    fp_repo_readme = fp_repo / 'README.md'
    fp_repo_readme_main = fp_repo / 'README_main.md'
    fp_repo_readme_notes = fp_repo / 'README_notes.md'

    # algorithms
    fp_algorithms = Path(fp_repo / 'algorithms')
    fp_algorithms_readme = fp_algorithms / 'README.md'
    fp_algorithms_problems = fp_algorithms / 'problems'
    fp_algorithms_property = fp_algorithms / 'properties.yml'

    # notes
    fp_notes = Path(fp_repo / 'notes')
    fp_notes_readme = fp_notes / 'README.md'
    fp_notes_readme_main = fp_notes / 'README_main.md'
    fp_notes_readme_temp = fp_notes / 'README_template.md'
    fp_notes_property = fp_notes / 'properties.yml'
