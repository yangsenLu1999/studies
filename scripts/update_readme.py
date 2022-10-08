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
    from readme.build import BuildReadme

if __name__ == '__main__':
    """"""
    br = BuildReadme()
    br.pipeline()
    # logger.info(f'Update README Success!')
