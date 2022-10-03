#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time:
    2022-09-28 20:19
Author:
    huayang (imhuay@163.com)
Subject:
    algo
"""
from __future__ import annotations

# import os
# import sys
# import unittest
import json
import os
import re

from typing import ClassVar
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

from huaytools.utils import get_logger, MarkdownUtils

from readme.args import args
from readme.utils import ReadmeUtils


@dataclass
class ProblemInfo:
    category: list[str]
    source: str
    number: str
    level: str
    name: str
    company: list[str]
    file_path: Path

    # field_name
    F_CATEGORY: ClassVar[str] = 'category'
    F_SOURCE: ClassVar[str] = 'source'
    F_NUMBER: ClassVar[str] = 'number'
    F_LEVEL: ClassVar[str] = 'level'
    F_NAME: ClassVar[str] = 'name'
    F_COMPANY: ClassVar[str] = 'company'
    F_PATH: ClassVar[str] = f'file_path'

    @property
    def head_name(self):
        return self.file_path.stem


@dataclass(unsafe_hash=True)
class TagTypeInfo:
    name: str
    level: int
    show_name: str


class TagType:
    hot: ClassVar[TagTypeInfo] = TagTypeInfo('hot', 0, 'Hot 🔥')
    level: ClassVar[TagTypeInfo] = TagTypeInfo(ProblemInfo.F_LEVEL, 1, 'Level 📈')
    subject: ClassVar[TagTypeInfo] = TagTypeInfo(ProblemInfo.F_SOURCE, 2, 'Subject 📓')
    category: ClassVar[TagTypeInfo] = TagTypeInfo(ProblemInfo.F_CATEGORY, 3, 'Category')


@dataclass(unsafe_hash=True)
class CategoryInfo:
    name: str
    level: int
    show_name: str = None


# class Categories:
#     base: ClassVar[CategoryInfo] = CategoryInfo('base', 0, '基础')
#     data_struct: ClassVar[CategoryInfo] = CategoryInfo('data_struct', 1, '数据结构')
#     algorithm: ClassVar[CategoryInfo] = CategoryInfo('algorithm', 2, '算法')
#     trick: ClassVar[CategoryInfo] = CategoryInfo('trick', 3, '技巧')
category_map = {
    '基础': CategoryInfo('base', 0, '基础'),
    '数据结构': CategoryInfo('data_struct', 1, '数据结构'),
    '算法': CategoryInfo('algorithm', 2, '算法'),
    '技巧': CategoryInfo('trick', 3, '技巧'),
}


@dataclass
class TagInfo:
    _tag_name: str
    tag_type: TagTypeInfo = None
    collects: list[ProblemInfo] = field(default_factory=list)

    @property
    def tag_count(self):
        return len(self.collects)

    EMPTY: ClassVar[str] = ''
    SEP: ClassVar[str] = '-'

    @property
    def tag_category(self):
        """格式：tag_category-tag_name"""
        if self.SEP in self._tag_name:
            return self._tag_name.split(self.SEP, maxsplit=1)[0]
        else:
            return self.EMPTY

    @property
    def tag_name(self):
        if self.SEP in self._tag_name:
            return self._tag_name.split(self.SEP, maxsplit=1)[1]
        else:
            return self._tag_name

    @property
    def tag_head(self):
        return f'{self.tag_name} ({self.tag_count})'


# 修改 tag 的类型
TAG2TYPE_MODIFY = {
    '热门&经典&易错': TagType.hot,
    'LeetCode Hot 100': TagType.hot,
}
# 加入热门标签
EX_HOT_TAGS = [
    # 'LeetCode',
    '剑指Offer',
    '牛客'
]

# 本地有效，GitHub 无效
# sp_div = '''<div>
# <div style="float: left; width: 50%; ">
#
# {toc_hot}
#
# </div>
# <div style="float: right; width: 50%; ">
#
# {toc_subject}
#
# </div>
# <div style="width: 50%; ">
#
# {toc_level}
#
# </div>
# </div>'''

# GitHub 上 style 失效: style="width: 100%; border: none; background: none"
TMP_TOC_TD_CATEGORY = '<td width="1000" valign="top">\n\n{sub_toc}\n\n</td>'
TMP_TOC_TABLE = '''<table>  <!-- frame="void" 无效 -->
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

README_TITLE = 'Algorithms'
TMP_README = '''# {title}

{toc}

---

{sub_toc}'''

TMP_README_CONCAT = '''## {title}

{toc}
'''


class Algorithms:
    """"""
    logger = get_logger()
    # AUTO_GENERATED_STR = '<!-- Auto-generated -->'
    RE_INFO = re.compile(r'<!--(.*?)-->', flags=re.DOTALL)

    def __init__(self):
        """"""
        # attrs
        self.title = self.__class__.__name__
        self.fp_algo = args.fp_algorithms
        self.fp_algo_readme = args.fp_algorithms_readme
        self.fp_repo_readme_algorithms = args.fp_repo_readme_algorithms
        self.fp_problems = self.fp_algo / 'problems'
        with open(self.fp_algo / 'tag2topic.json') as f:
            self.tag2topic = json.load(f)

        # pipeline
        self._extract_infos()
        self._collect_tags()
        # self.generate_readme()

    problems_infos: list[ProblemInfo] = []

    def _extract_infos(self):
        """"""
        for dp, _, fns in os.walk(self.fp_problems):
            for fn in fns:
                fp = Path(dp) / fn  # each problem.md
                if fp.suffix != '.md':
                    continue
                info = self._extract_info(fp)
                self._update_info(info, fp)
                self._try_update_title(info)
                self.problems_infos.append(ProblemInfo(**info))

    def _update_info(self, info, fp):
        """"""
        # update standard tag
        new_tags = []
        for tag in info[ProblemInfo.F_CATEGORY]:
            new_tags.append(self.tag2topic[tag.lower()])
        info[ProblemInfo.F_CATEGORY] = new_tags

        # add fp
        fp = self._try_rename(fp, info)
        info[ProblemInfo.F_PATH] = fp

    TEMPLATE_PROBLEM_TITLE = '## {src}_{no}_{title}（{level}, {date}）'

    def _try_update_title(self, info) -> bool:
        """"""
        fp = info[ProblemInfo.F_PATH]
        new_title = self.TEMPLATE_PROBLEM_TITLE.format(src=info[ProblemInfo.F_SOURCE],
                                                       no=info[ProblemInfo.F_NUMBER],
                                                       title=info[ProblemInfo.F_NAME],
                                                       level=info[ProblemInfo.F_LEVEL],
                                                       date='-'.join(str(fp.parent).split('/')[-2:]))
        with fp.open(encoding='utf8') as f:
            lines = f.read().split('\n')

        updated = True
        if not lines[0].startswith('##'):
            lines.insert(0, new_title)
        elif lines[0] != new_title:
            lines[0] = new_title
        else:
            updated = False

        if updated:
            with fp.open('w', encoding='utf8') as f:
                f.write('\n'.join(lines))
                # ReadmeUtils.git_add(fp)

        return updated

    TEMPLATE_PROBLEM_FILENAME = '{src}_{no}_{level}_{title}.md'

    def _try_rename(self, fp: Path, info) -> Path:
        """"""
        new_fn = self.TEMPLATE_PROBLEM_FILENAME.format(src=info[ProblemInfo.F_SOURCE],
                                                       no=info[ProblemInfo.F_NUMBER],
                                                       level=info[ProblemInfo.F_LEVEL],
                                                       title=info[ProblemInfo.F_NAME])
        if new_fn != fp.name:
            # fp = fp.rename(fp.parent / new_fn)
            new_fp = fp.parent / new_fn
            # ReadmeUtils.git_add(fp)
            ReadmeUtils.git_mv(fp, new_fp)
            fp = new_fp
        return fp

    NUMBER_WIDTH = 5

    def _extract_info(self, fp) -> dict:
        """"""
        fp = Path(fp)
        with fp.open(encoding='utf8') as f:
            txt = f.read()

        try:
            info_str = self.RE_INFO.search(txt).group(1)
            info = json.loads(info_str)
            return info
        except:  # noqa
            self.logger.info(fp)

    # GIT_ADD_TEMP = 'git add "{fp}"'
    #
    # def _git_add(self, fp):
    #     """"""
    #     command = self.GIT_ADD_TEMP.format(fp=fp)
    #     # self.logger.info(command)
    #     os.system(command)

    tag_infos: dict[str, TagInfo] = dict()

    def _collect_tags(self):
        """"""

        def _add(_tag, _type, _info):
            _type = TAG2TYPE_MODIFY.get(_tag, _type)
            if _tag not in self.tag_infos:
                self.tag_infos[_tag] = TagInfo(_tag)
                self.tag_infos[_tag].tag_type = _type
            else:
                assert self.tag_infos[_tag].tag_type == _type
            self.tag_infos[_tag].collects.append(_info)

        for problems_info in self.problems_infos:
            _add(problems_info.source, TagType.subject, problems_info)
            _add(problems_info.level, TagType.level, problems_info)
            for cat in problems_info.category:
                _add(cat, TagType.category, problems_info)

        # sort
        for info in self.tag_infos.values():
            info.collects.sort(key=lambda i: (i.source, i.number))

    hot_toc: list[str]
    type2tags: dict[TagTypeInfo, list[TagInfo]] = defaultdict(list)

    @staticmethod
    def _get_toc_tag_line(tag_info):
        """"""
        return f'- [{tag_info.tag_head}](#{MarkdownUtils.slugify(tag_info.tag_head)})'

    def _generate_sub_toc(self, tag_type: TagTypeInfo):
        """"""
        sub_toc = [f'### {tag_type.show_name}']
        for tag_info in self.type2tags[tag_type]:
            sub_toc.append(self._get_toc_tag_line(tag_info))
        return sub_toc

    def _generate_category_toc(self):
        """tag格式：category-tag_name"""
        category2problems: dict[CategoryInfo, list[TagInfo]] = defaultdict(list)
        for tag_info in self.type2tags[TagType.category]:
            assert tag_info.tag_category != ''
            category2problems[category_map[tag_info.tag_category]].append(tag_info)

        # toc = [f'## {TagType.category.show_name}']
        toc = []
        for tag_category in sorted(category2problems.keys(), key=lambda k: k.level):
            # toc.append(f'### {tag_category.show_name}')
            sub_toc = [f'### {tag_category.show_name}']
            tag_infos = category2problems[tag_category]
            for tag_info in tag_infos:
                sub_toc.append(self._get_toc_tag_line(tag_info))

            toc.append(TMP_TOC_TD_CATEGORY.format(sub_toc='\n'.join(sub_toc)))
        return toc

    def build(self):
        """"""
        for tag, tag_info in self.tag_infos.items():
            self.type2tags[tag_info.tag_type].append(tag_info)
        for tag_infos in self.type2tags.values():
            tag_infos.sort(key=lambda i: (i.tag_category, -i.tag_count, i.tag_name))

        # sub toc
        self.hot_toc = toc_hot = self._generate_sub_toc(TagType.hot)
        for tag in EX_HOT_TAGS:
            tag_info = self.tag_infos[tag]
            toc_hot.append(self._get_toc_tag_line(tag_info))
        toc_level = self._generate_sub_toc(TagType.level)
        toc_subject = self._generate_sub_toc(TagType.subject)
        toc_category = self._generate_category_toc()

        contents = []
        for tag_type in sorted(self.type2tags.keys(), key=lambda i: i.level):
            tag_infos = self.type2tags[tag_type]
            for tag_info in tag_infos:
                contents.append(f'### {tag_info.tag_head}')
                for problem_info in tag_info.collects:
                    contents.append('- [`{name}`]({path})'.format(
                        name=problem_info.head_name,
                        path=problem_info.file_path.relative_to(self.fp_algo)
                    ))
                contents.append('')

        toc = TMP_TOC_TABLE.format(toc_hot='\n'.join(toc_hot),
                                   toc_subject='\n'.join(toc_subject),
                                   toc_level='\n'.join(toc_level),
                                   toc_category='\n'.join(toc_category))
        sub_toc = '\n'.join(contents)
        readme = TMP_README.format(title=self.title, toc=toc, sub_toc=sub_toc)

        with self.fp_algo_readme.open('w', encoding='utf8') as f:
            f.write(readme)

        toc_concat = toc.replace('(#', f'({self.fp_algo.name}/README.md#')
        readme_concat = TMP_README_CONCAT.format(title=self.title, toc=toc_concat)
        with self.fp_repo_readme_algorithms.open('w', encoding='utf8') as f:
            f.write(readme_concat)


if __name__ == '__main__':
    """"""
    algo = Algorithms()
    algo.build()
