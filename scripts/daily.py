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

try:
    sys.path.insert(0, f'{Path(__file__).parent.parent / "src"}')
except:  # noqa
    exit(1)
else:
    from readme.build import BuildReadme
    from readme.utils import ReadmeUtils, args

logger = get_logger()
last_modify_tmp = '![last modify](https://img.shields.io/static/v1?label=last%20modify&message={datetime}&color=yellowgreen&style=flat-square)'  # noqa


def update_note_last_modify(fp: Path):
    datetime = ReadmeUtils.get_file_last_commit_date(fp)
    datetime = f'{datetime[:10]}%20{datetime[11:19]}'
    with fp.open(encoding='utf8') as f:
        new_txt = ReadmeUtils.replace_tag_content('badge', f.read(), last_modify_tmp.format(datetime=datetime))
    with fp.open('w', encoding='utf8') as f:
        f.write(new_txt)


def main():
    for dp, _, fns in os.walk(args.fp_notes_archives):
        for fn in fns:
            fp = Path(dp) / fn
            if fp.suffix != '.md':
                continue
            update_note_last_modify(fp)
            # sys.exit(1)


if __name__ == '__main__':
    """"""
    main()
    br = BuildReadme()
    br.build()
    br.git_push()
    logger.info(f'Update Success!')
