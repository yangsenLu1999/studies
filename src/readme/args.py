#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time:
    2022-09-28 20:30
Author:
    huayang (imhuay@163.com)
Subject:
    consts
"""
from __future__ import annotations

import os
# import sys
# import json
# import unittest

# from typing import *
# from collections import defaultdict
from pathlib import Path


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
