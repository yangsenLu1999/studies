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

import yaml

from huaytools.utils import MarkdownUtils

from readme.utils import ReadmeUtils, args


@dataclass
class ProblemInfo:
    tags: list[str]
    source: str
    number: str
    level: str
    name: str
    companies: list[str]
    file_path: Path

    # field_name
    F_TAGS: ClassVar[str] = 'tags'
    F_SOURCE: ClassVar[str] = 'source'
    F_NUMBER: ClassVar[str] = 'number'
    F_LEVEL: ClassVar[str] = 'level'
    F_NAME: ClassVar[str] = 'name'
    F_COMPANIES: ClassVar[str] = 'companies'
    F_PATH: ClassVar[str] = f'file_path'

    @property
    def head_name(self):
        return self.file_path.stem


@dataclass(unsafe_hash=True)
class TagType:
    name: str
    level: int
    show_name: str


@dataclass(unsafe_hash=True)
class AlgoType:
    name: str
    level: int
    show_name: str | None = None


@dataclass(unsafe_hash=True)
class TagInfo:
    tag_name: str
    tag_types: list[TagType] = field(hash=False)
    algo_type: AlgoType | None = field(default=None, hash=False)
    collects: list[ProblemInfo] = field(default_factory=list, hash=False)

    @property
    def tag_count(self):
        return len(self.collects)

    @property
    def tag_head(self):
        return f'{self.tag_name} ({self.tag_count})'

    EMPTY: ClassVar[str] = ''

    # SEP: ClassVar[str] = '-'

    @property
    def tag_category(self):
        """格式：tag_category-tag_name"""
        if self.algo_type is not None:
            return self.algo_type.name
        else:
            return self.EMPTY


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


class Property:

    def __init__(self, fp_property_yaml: Path):
        with fp_property_yaml.open(encoding='utf8') as f:
            self.properties = yaml.safe_load(f.read())

        self._load_tags()

    tags: dict[str, TagInfo]
    _tag2info: dict[str, TagInfo] = dict()

    def _load_tags(self):
        self.tags = tags = dict()
        # common_tags
        for k, v in self.properties['tags']['common'].items():
            tag_name = v.get('name', k)
            if tag_name not in self._tag2info:
                tag_types = list()
                if isinstance(v['type'], str):
                    v['type'] = [v['type']]
                for t in v['type']:
                    tag_types.append(self.tag_types[t])
                if isinstance(tag_types, str):
                    tag_types = [tag_types]
                tags[k.lower()] = self._tag2info[tag_name] = TagInfo(tag_name=tag_name,
                                                                     tag_types=tag_types)
            else:
                tags[k.lower()] = self._tag2info[tag_name]

        # algo_tags
        for k, v in self.properties['tags']['algo'].items():
            tag_name = v.get('name', k)
            if tag_name not in self._tag2info:
                tag_types = [self.tag_types['algorithm']]
                ex_tag_types = v.get('type', None)
                if ex_tag_types:
                    if isinstance(ex_tag_types, str):
                        ex_tag_types = [ex_tag_types]
                    for t in ex_tag_types:
                        tag_types.append(self.tag_types[t])
                tags[k.lower()] = self._tag2info[tag_name] = TagInfo(tag_name=tag_name,
                                                                     tag_types=tag_types,
                                                                     algo_type=self.algo_types[v['algo_type']])
            else:
                tags[k.lower()] = self._tag2info[tag_name]

    _tag_types: dict[str, TagType] | None = None

    @property
    def tag_types(self):
        if self._tag_types is None:
            self._tag_types = dict()
            for k, v in self.properties['tag_types'].items():
                v['name'] = k
                self._tag_types[k] = TagType(**v)
        return self._tag_types

    @property
    def tt_hot(self):
        return self.tag_types['hot']

    @property
    def tt_level(self):
        return self.tag_types['level']

    @property
    def tt_subject(self):
        return self.tag_types['subject']

    @property
    def tt_algorithm(self):
        return self.tag_types['algorithm']

    _algo_types: dict[str, AlgoType] | None = None

    @property
    def algo_types(self):
        if self._algo_types is None:
            self._algo_types = dict()
            for k, v in self.properties['algo_types'].items():
                v['name'] = k
                self._algo_types[k] = AlgoType(**v)
        return self._algo_types


class Algorithms:
    """"""
    # AUTO_GENERATED_STR = '<!-- Auto-generated -->'
    _RE_INFO = re.compile(r'<!--(.*?)-->', flags=re.DOTALL)
    _tag_infos: dict[str, TagInfo]
    _problems_infos: list[ProblemInfo]
    _type2tags: dict[TagType, list[TagInfo]]

    def __init__(self):
        """"""
        # attrs
        self.title = README_TITLE
        self._fp_algo = args.fp_algorithms
        self._fp_algo_readme = args.fp_algorithms_readme
        self._fp_problems = args.fp_algorithms_problems
        self.property = Property(args.fp_algorithms_property)
        self._tag_infos = self.property.tags

        # pipeline
        self._extract_infos()
        self._collect_tags()

    def _extract_infos(self):
        """"""
        self._problems_infos = []
        for dp, _, fns in os.walk(self._fp_problems):
            for fn in fns:
                fp = Path(dp) / fn  # each problem.md
                if fp.suffix != '.md':
                    continue
                info = self._extract_info(fp)
                self._update_info(info, fp)
                self._try_update_title(info)
                self._problems_infos.append(ProblemInfo(**info))

    def _update_info(self, info, fp):
        """"""
        # update standard tag
        # new_tags = []
        # for tag in info[ProblemInfo.F_TAGS]:
        #     new_tags.append(self.tag2topic[tag.lower()])
        # info[ProblemInfo.F_TAGS] = new_tags

        # add fp
        fp = self._try_rename(fp, info)
        info[ProblemInfo.F_PATH] = fp

    _TMP_PROBLEM_TITLE = '## {src}_{no}_{title}（{level}, {date}）'

    def _try_update_title(self, info) -> bool:
        """"""
        fp = info[ProblemInfo.F_PATH]
        new_title = self._TMP_PROBLEM_TITLE.format(src=info[ProblemInfo.F_SOURCE],
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

    _TMP_PROBLEM_FILENAME = '{src}_{no}_{level}_{title}.md'

    def _try_rename(self, fp: Path, info) -> Path:
        """"""
        new_fn = self._TMP_PROBLEM_FILENAME.format(src=info[ProblemInfo.F_SOURCE],
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

    def _extract_info(self, fp) -> dict:
        """"""
        fp = Path(fp)
        with fp.open(encoding='utf8') as f:
            txt = f.read()

        try:
            info_str = self._RE_INFO.search(txt).group(1)  # type:ignore
            info = json.loads(info_str)
        except:  # noqa
            raise ValueError(fp)

        return info

    def _collect_tags(self):

        def _add(_tag, _info):
            self._tag_infos[_tag.lower()].collects.append(_info)

        for problems_info in self._problems_infos:
            _add(problems_info.source, problems_info)
            _add(problems_info.level, problems_info)
            for tag in problems_info.tags:
                _add(tag, problems_info)

        # sort
        for info in self._tag_infos.values():
            info.collects.sort(key=lambda i: (i.source, i.number))

    @staticmethod
    def _get_toc_tag_line(tag_info):
        """"""
        return f'- [{tag_info.tag_head}](#{MarkdownUtils.slugify(tag_info.tag_head)})'

    def _generate_sub_toc(self, tag_type: TagType):
        """"""
        sub_toc = [f'### {tag_type.show_name}']
        for tag_info in self._type2tags[tag_type]:
            sub_toc.append(self._get_toc_tag_line(tag_info))
        return sub_toc

    def _generate_algo_toc(self):
        algo2problems: dict[AlgoType, list[TagInfo]] = defaultdict(list)
        for tag_info in self._type2tags[self.property.tt_algorithm]:
            assert tag_info.tag_category != ''
            algo2problems[tag_info.algo_type].append(tag_info)

        # toc = [f'## {TagType.category.show_name}']
        toc = []
        for tag_category in sorted(algo2problems.keys(), key=lambda k: k.level):
            # toc.append(f'### {tag_category.show_name}')
            sub_toc = [f'### {tag_category.show_name}']
            tag_infos = algo2problems[tag_category]
            for tag_info in tag_infos:
                sub_toc.append(self._get_toc_tag_line(tag_info))

            toc.append(TMP_TOC_TD_CATEGORY.format(sub_toc='\n'.join(sub_toc)))
        return toc

    readme_toc: str
    readme_concat: str

    def build(self):
        self._type2tags = dict()
        tmp = defaultdict(set)
        for tag, tag_info in self._tag_infos.items():
            for tag_type in tag_info.tag_types:
                tmp[tag_type].add(tag_info)
        for k, tag_infos in tmp.items():
            self._type2tags[k] = sorted(tag_infos, key=lambda i: (-i.tag_count, i.tag_name))

        # sub toc
        toc_hot = self._generate_sub_toc(self.property.tt_hot)
        toc_level = self._generate_sub_toc(self.property.tt_level)
        toc_subject = self._generate_sub_toc(self.property.tt_subject)
        toc_algo = self._generate_algo_toc()

        contents = []
        for tag_type in sorted(self._type2tags.keys(), key=lambda i: i.level):
            # 因为 hot 标签只作为附加标签，所以跳过，防止重复生成
            if tag_type is self.property.tt_hot:
                continue
            tag_infos = self._type2tags[tag_type]
            for tag_info in tag_infos:
                contents.append(f'### {tag_info.tag_head}')
                for problem_info in tag_info.collects:
                    contents.append('- [`{name}`]({path})'.format(
                        name=problem_info.head_name,
                        path=problem_info.file_path.relative_to(self._fp_algo)
                    ))
                contents.append('')

        toc = TMP_TOC_TABLE.format(toc_hot='\n'.join(toc_hot),
                                   toc_subject='\n'.join(toc_subject),
                                   toc_level='\n'.join(toc_level),
                                   toc_category='\n'.join(toc_algo))
        sub_toc = '\n'.join(contents)
        readme = TMP_README.format(title=self.title, toc=toc, sub_toc=sub_toc)

        with self._fp_algo_readme.open('w', encoding='utf8') as f:
            f.write(readme)

        toc_concat = toc.replace('(#', f'({self._fp_algo.name}/README.md#')
        self.readme_toc = f'- [{README_TITLE}](#{MarkdownUtils.slugify(README_TITLE)})'
        self.readme_concat = TMP_README_CONCAT.format(title=self.title, toc=toc_concat)
        # with self.fp_repo_readme_algorithms.open('w', encoding='utf8') as f:
        #     f.write(readme_concat)


if __name__ == '__main__':
    """"""
    algo = Algorithms()
    algo.build()
