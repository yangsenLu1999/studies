#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time:
    2022-09-29 13:43
Author:
    huayang (imhuay@163.com)
Subject:
    update_problem_info
"""
from __future__ import annotations

import os
import re
# import sys
import json
# import unittest

# from typing import *
# from collections import defaultdict
from pathlib import Path

from huaytools_local.utils import NoIndentJSONEncoder

fp_problems = Path(r'../../algorithms/problems')
RE_INFO = re.compile(r'<!--(.*?)-->', flags=re.DOTALL)

new_info_temp = '''<!--
{info}
-->'''

for dp, _, fns in os.walk(fp_problems):
    for fn in fns:
        fp = Path(dp) / fn

        with fp.open(encoding='utf8') as f:
            txt = f.read()

        info_str = RE_INFO.search(txt).group(1)
        info = json.loads(info_str)
        new_info = {
            'category': NoIndentJSONEncoder.wrap(info['category']),
            'source': info['source'],
            'level': '困难' if info['level'] == '较难' else info['level'],
            'number': info['number'],
            'name': info['name'],
            'company': NoIndentJSONEncoder.wrap(info['company']),
        }
        new_info_str = new_info_temp.format(info=json.dumps(new_info, indent=4, ensure_ascii=False,
                                                            cls=NoIndentJSONEncoder))

        new_txt = RE_INFO.sub(new_info_str, txt, count=1)
        with fp.open('w', encoding='utf8') as f:
            f.write(new_txt)
