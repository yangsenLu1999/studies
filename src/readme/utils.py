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
import re
# import sys
# import json
# import unittest
import subprocess

# from typing import *
# from collections import defaultdict
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, fields

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

    @staticmethod
    def get_file_first_commit_date(fp, return_datetime=False) -> str | datetime:
        code, date_str = subprocess.getstatusoutput(f'git log --follow --format=%ad --date=iso-strict {fp} | tail -1')
        if code != 0:
            raise ValueError(f'{ReadmeUtils.get_file_first_commit_date.__name__}: {fp}')
        if return_datetime:
            return datetime.fromisoformat(date_str)
        return date_str

    @staticmethod
    def get_file_last_commit_date(fp, return_datetime=False) -> str | datetime:
        code, date_str = subprocess.getstatusoutput(f'git log --follow --format=%ad --date=iso-strict {fp} | head -1')
        if code != 0:
            raise ValueError(f'{ReadmeUtils.get_file_last_commit_date.__name__}: {fp}')
        if return_datetime:
            return datetime.fromisoformat(date_str)
        return date_str

    # RE_WAKATIME = re.compile(r'<!--START_SECTION:waka-->[\s\S]+<!--END_SECTION:waka-->')

    # @staticmethod
    # def extract_wakatime(txt) -> str:
    #     return ReadmeUtils.RE_WAKATIME.search(txt).group()

    SECTION_START = '<!--START_SECTION:{tag}-->'
    SECTION_END = '<!--END_SECTION:{tag}-->'
    SECTION_ANNOTATION = r'<!--{tag}\n([\s\S]+)\n-->'

    @staticmethod
    def replace_tag_content(tag, txt, content) -> str:
        """"""
        tag_begin = ReadmeUtils.SECTION_START.format(tag=tag)
        tag_end = ReadmeUtils.SECTION_END.format(tag=tag)
        re_pattern = re.compile(fr'{tag_begin}[\s\S]+{tag_end}')
        repl = f'{tag_begin}\n\n{content}\n\n{tag_end}'
        return re_pattern.sub(repl, txt, count=1)

    @staticmethod
    def get_tag_content(tag, txt) -> str:
        """"""
        tag_begin = ReadmeUtils.SECTION_START.format(tag=tag)
        tag_end = ReadmeUtils.SECTION_END.format(tag=tag)
        re_pattern = re.compile(fr'{tag_begin}[\s\S]+{tag_end}')
        return re_pattern.search(txt).group()

    @staticmethod
    def get_annotation(tag, txt) -> str | None:
        """"""
        re_pattern = re.compile(ReadmeUtils.SECTION_ANNOTATION.format(tag=tag))
        m = re_pattern.search(txt)
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def get_annotation_info(txt) -> str | None:
        """"""
        return ReadmeUtils.get_annotation('info', txt)


@dataclass
class ReadmeTag:
    index: str = None
    recent: str = None
    algorithms: str = None
    notes: str = None
    waka: str = None

    def __post_init__(self):
        for f in fields(self):
            setattr(self, f.name, f.name)


readme_tag = ReadmeTag()


class args:  # noqa
    """"""
    _fp_cur_file = Path(__file__)
    # repo
    fp_repo = Path(_fp_cur_file.parent / '../..').resolve()
    fp_repo_readme = fp_repo / 'README.md'

    # algorithms
    fp_algorithms = Path(fp_repo / 'algorithms')
    fp_algorithms_readme = fp_algorithms / 'README.md'
    fp_algorithms_problems = fp_algorithms / 'problems'
    fp_algorithms_property = fp_algorithms / 'properties.yml'
    algorithms_readme_title = 'Algorithm Coding'

    # notes
    fp_notes = Path(fp_repo / 'notes')
    fp_notes_archives = fp_notes / '_archives'
    fp_notes_readme = fp_notes / 'README.md'
    fp_notes_readme_temp = fp_notes / 'README_template.md'
    fp_notes_property = fp_notes / 'properties.yml'
    notes_top_limit = 5


TEMP_main_readme_notes_recent_toc = '''## Recently 📖
{toc_top}
{toc_recent}
'''
TEMP_main_readme_algorithms_concat = '''## {title}

{toc}
'''

TEMP_algorithm_toc_td_category = '<td width="1000" valign="top">\n\n{sub_toc}\n\n</td>'
TEMP_algorithm_toc_table = '''<table>  <!-- invalid: frame="void", style="width: 100%; border: none; background: none" -->
<tr>
<td colspan="2" valign="top" width="1000">

{toc_hot}

</td>
<td colspan="2" rowspan="3" valign="top" width="1000">

{toc_subject}

</td>
</tr>
<tr></tr>
<tr>
<td colspan="2" valign="top">

{toc_level}

</td>
</tr>
<tr></tr>
<tr>  <!-- loop TMP_TOC_TD_CATEGORY -->

{toc_category}

</tr>
</table>'''
TEMP_algorithm_readme = '''# {title}

{toc}

---

{sub_toc}'''
