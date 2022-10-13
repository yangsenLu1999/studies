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

from readme.utils import (
    args,
    ReadmeUtils,
    TEMP_algorithm_toc_td_category,
    TEMP_algorithm_toc_table,
    TEMP_algorithm_readme,
    TEMP_main_readme_algorithms_concat
)

from readme._base import Readme


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
    collects: list[Problem] = field(default_factory=list, hash=False)

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


class _Property:

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


properties = _Property(args.fp_algorithms_property)


@dataclass()
class Problem:
    _path: Path

    # property
    _info = None
    _file_name = None
    _title = None
    _last_commit_time = None

    # ClassVar
    # _TEMP_PROBLEM_FILENAME: ClassVar[str] = '{src}_{no}_{level}_{name}.md'
    # _TEMP_TITLE: ClassVar[str] = '{name} ({src}, {level})'
    _TAG_BADGE = 'badge'

    def __post_init__(self):
        self.update_file()

    def update_file(self):
        # try update title
        with self.path.open(encoding='utf8') as f:
            txt = f.read()

        lns = txt.split('\n', maxsplit=1)
        if lns[0].startswith('##'):
            lns[0] = self.head
        else:
            lns.insert(0, self.head)

        if not ReadmeUtils.get_tag_content(self._TAG_BADGE, txt):
            lns.insert(1, ReadmeUtils.get_tag_begin(self._TAG_BADGE))
            lns.insert(2, ReadmeUtils.get_tag_end(self._TAG_BADGE))

        txt = ReadmeUtils.replace_tag_content(self._TAG_BADGE,
                                              '\n'.join(lns),
                                              self.badge_content)
        with self.path.open('w', encoding='utf8') as f:
            f.write(txt)

    @property
    def badge_content(self):
        return '\n'.join([ReadmeUtils.get_last_modify_badge_url(self.path),
                          ReadmeUtils.get_badge_url('source', message=self.source, color='green'),
                          ReadmeUtils.get_badge_url('level', message=self.level, color='yellow'),
                          ReadmeUtils.get_badge_url('tags', message=self.message_tags, color='orange')])

    # @property
    # def message_source(self):
    #     return f'{self.source}-{self.level}'

    @property
    def message_tags(self):
        def _get_tag(_tag):
            if properties.tags[_tag.lower()].algo_type is None:
                return properties.tags[_tag.lower()].tag_name
            else:
                return _tag

        return ', '.join([_get_tag(tag) for tag in self.tags])

    @property
    def path(self):
        if self.file_name != self._path.name:
            new_path = self._path.parent / self.file_name
            ReadmeUtils.git_mv(self._path, new_path)
            self._path = new_path
        return self._path

    @property
    def title(self):
        # if self._title is None:
        #     self._title = '{name} ({src}, {level})'.format(name=self.name,
        #                                                    src=self.source,
        #                                                    level=self.level)
        # return self._title
        return self.name

    @property
    def head(self):
        return f'## {self.title}'

    @property
    def last_commit_time(self):
        if self._last_commit_time is None:
            ReadmeUtils.get_file_last_commit_date(self.path)
        return self._last_commit_time

    @property
    def file_name(self):
        if self._file_name is None:
            self._file_name = '{src}_{no}_{level}_{name}.md'.format(src=self.source,
                                                                    no=self.number,
                                                                    level=self.level,
                                                                    name=re.sub(r'\s+', '', self.name))
        return self._file_name

    @property
    def info(self):
        if self._info is None:
            with self._path.open(encoding='utf8') as f:
                txt = f.read()
            try:
                info_str = ReadmeUtils.get_annotation_info(txt)
                self._info = yaml.safe_load(info_str.strip())
            except:  # noqa
                raise ValueError(self._path)
        return self._info

    _F_TAGS: ClassVar[str] = 'tags'
    _F_SOURCE: ClassVar[str] = 'source'
    _F_NUMBER: ClassVar[str] = 'number'
    _F_LEVEL: ClassVar[str] = 'level'
    _F_NAME: ClassVar[str] = 'name'
    _F_COMPANIES: ClassVar[str] = 'companies'

    @property
    def tags(self) -> list[str]:
        return self.info[Problem._F_TAGS]

    @property
    def source(self) -> str:
        return self.info[Problem._F_SOURCE]

    @property
    def number(self) -> str:
        return self.info[Problem._F_NUMBER]

    @property
    def level(self) -> str:
        return self.info[Problem._F_LEVEL]

    @property
    def name(self) -> str:
        return self.info[Problem._F_NAME]

    @property
    def companies(self) -> list[str]:
        return self.info[Problem._F_COMPANIES]


class AlgorithmsReadme(Readme):

    def __init__(self):
        """"""

    def build(self):
        """"""


# @dataclass
# class ProblemInfo:
#     tags: list[str]
#     source: str
#     number: str
#     level: str
#     name: str
#     companies: list[str]
#     file_path: Path
#
#     # field_name
#     F_TAGS: ClassVar[str] = 'tags'
#     F_SOURCE: ClassVar[str] = 'source'
#     F_NUMBER: ClassVar[str] = 'number'
#     F_LEVEL: ClassVar[str] = 'level'
#     F_NAME: ClassVar[str] = 'name'
#     F_COMPANIES: ClassVar[str] = 'companies'
#     F_PATH: ClassVar[str] = f'file_path'
#
#     @property
#     def head_name(self):
#         return self.file_path.stem


class Algorithms:
    """"""
    # AUTO_GENERATED_STR = '<!-- Auto-generated -->'
    _RE_INFO = re.compile(r'<!--(.*?)-->', flags=re.DOTALL)
    _RE_INFO_YAML = re.compile(r'<!--info(.*?)-->', flags=re.DOTALL)
    _tag_infos: dict[str, TagInfo]
    _problems: list[Problem]
    _type2tags: dict[TagType, list[TagInfo]]

    def __init__(self):
        """"""
        # attrs
        self.title = args.algorithms_readme_title
        self._fp_algo = args.fp_algorithms
        self._fp_algo_readme = args.fp_algorithms_readme
        self._fp_problems = args.fp_algorithms_problems
        self.properties = properties
        self._tag_infos = self.properties.tags

        # pipeline
        self._extract_infos()
        self._collect_tags()

    def _extract_infos(self):
        """"""
        self._problems = []
        for dp, _, fns in os.walk(self._fp_problems):
            for fn in fns:
                fp = Path(dp) / fn  # each problem.md
                if fp.suffix != '.md':
                    continue
                self._problems.append(Problem(fp))

    def _collect_tags(self):

        def _add(_tag, _info):
            self._tag_infos[_tag.lower()].collects.append(_info)

        for p in self._problems:
            _add(p.source, p)
            _add(p.level, p)
            for tag in p.tags:
                _add(tag, p)

        # sort
        for info in self._tag_infos.values():
            info.collects.sort(key=lambda i: (i.source, i.number))

    @staticmethod
    def _get_toc_tag_line(tag_info):
        """"""
        return f'- [{tag_info.tag_head}](#{MarkdownUtils.slugify(tag_info.tag_name)})'

    def _generate_sub_toc(self, tag_type: TagType):
        """"""
        sub_toc = [f'### {tag_type.show_name}']
        # if tag_type is self.properties.tt_level:
        #     sub_toc[0] += " " + ReadmeUtils.get_badge_url('total', len(self._problems), 'success')
        for tag_info in self._type2tags[tag_type]:
            sub_toc.append(self._get_toc_tag_line(tag_info))
        return sub_toc

    def _generate_algo_toc(self):
        algo2problems: dict[AlgoType, list[TagInfo]] = defaultdict(list)
        for tag_info in self._type2tags[self.properties.tt_algorithm]:
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

            toc.append(TEMP_algorithm_toc_td_category.format(sub_toc='\n'.join(sub_toc)))
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
        toc_hot = self._generate_sub_toc(self.properties.tt_hot)
        toc_level = self._generate_sub_toc(self.properties.tt_level)
        toc_subject = self._generate_sub_toc(self.properties.tt_subject)
        toc_algo = self._generate_algo_toc()

        contents = []
        for tag_type in sorted(self._type2tags.keys(), key=lambda i: i.level):
            # 因为 hot 标签只作为附加标签，所以跳过，防止重复生成
            if tag_type is self.properties.tt_hot:
                continue
            tag_infos = self._type2tags[tag_type]
            for tag_info in tag_infos:
                contents.append(f'### {tag_info.tag_name}')
                contents.append(ReadmeUtils.get_badge_url('total', len(tag_info.collects), 'success'))
                for p in tag_info.collects:
                    contents.append('- [`{name}`]({path})'.format(
                        name=p.path.stem,
                        path=p.path.relative_to(self._fp_algo)
                    ))
                contents.append('')

        toc = TEMP_algorithm_toc_table.format(toc_hot='\n'.join(toc_hot),
                                              toc_subject='\n'.join(toc_subject),
                                              toc_level='\n'.join(toc_level),
                                              toc_category='\n'.join(toc_algo))
        sub_toc = '\n'.join(contents)
        readme = TEMP_algorithm_readme.format(title=self.title, toc=toc, sub_toc=sub_toc)

        with self._fp_algo_readme.open('w', encoding='utf8') as f:
            f.write(readme)

        toc_concat = toc.replace('(#', f'({self._fp_algo.name}/README.md#')
        self.readme_toc = f'- [{self.title}](#{MarkdownUtils.slugify(self.title)})'
        self.readme_concat = TEMP_main_readme_algorithms_concat.format(title=self.title, toc=toc_concat)
        # with self.fp_repo_readme_algorithms.open('w', encoding='utf8') as f:
        #     f.write(readme_concat)


if __name__ == '__main__':
    """"""
    algo = Algorithms()
    algo.build()
