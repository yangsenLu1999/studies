#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time:
    2022-10-05 13:15
Author:
    huayang (imhuay@163.com)
Subject:
    notes
"""
from __future__ import annotations

# import os
# import sys
# import json
# import unittest
import re
import yaml

from typing import ClassVar
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field

from readme.utils import args


# TMP_subject_toc = '''### {title}
#
# {toc}
# '''


@dataclass(unsafe_hash=True)
class SubjectId:
    id: str
    name: str


class RE:
    note_name = re.compile(r'(\d{3})-(.*?).md')
    note_info = re.compile(r'<!--info(.*?)-->', flags=re.DOTALL)
    note_toc = re.compile(r'<!-- TOC -->(.*?)<!-- TOC -->', flags=re.DOTALL)
    note_content = re.compile(r'<!-- CONTENT -->(.*?)<!-- CONTENT -->', flags=re.DOTALL)


@dataclass
class SubjectInfo:
    _prefix: str
    name: str
    path: Path
    _txt: str = field(default=None, hash=False)
    _toc: str = field(default=None, hash=False)
    _info: dict = field(default=None, hash=False)

    # subject_ids: ClassVar[dict[str, SubjectId]]

    @property
    def subject_id(self) -> str:
        return self._prefix[0]

    @property
    def subject_number(self) -> str:
        return self._prefix[1:]

    @property
    def txt(self):
        if self._txt is None:
            with self.path.open(encoding='utf8') as f:
                self._txt = f.read().strip()
        return self._txt

    @property
    def toc(self) -> str:
        if self._toc is None:
            m = RE.note_toc.search(self.txt)
            if not m:
                raise ValueError(self.path)
            self._toc = m.group(1).strip().replace('(#', f'({self.path.name}#')
            # self._toc = TMP_subject_toc.format(title=self.name, toc=_toc)
        return self._toc

    @property
    def info(self) -> dict:
        if self._info is None:
            m = RE.note_info.search(self.txt)
            if not m:
                raise ValueError(self.path)
            self._info = yaml.safe_load(m.group(1).strip())
        return self._info

    @property
    def toc_id(self):
        return self.info['toc_id']


class Property:

    def __init__(self, fp_property_yaml):
        with fp_property_yaml.open(encoding='utf8') as f:
            self.properties = yaml.safe_load(f.read())

        self._load_subject_ids()

    subject_ids: dict[str, SubjectId]

    def _load_subject_ids(self):
        self.subject_ids = dict()
        for k, v in self.properties['subject_ids'].items():
            k = str(k)
            self.subject_ids[k] = SubjectId(k, **v)


class Notes:

    def __init__(self):
        """"""
        self._fp_notes = args.fp_notes
        self._fp_notes_readme = args.fp_notes_readme
        self._fp_notes_readme_temp = args.fp_notes_readme_temp
        self.property = Property(args.fp_notes_property)

        self._load_note_indexes()

    subjects: list[SubjectInfo]
    cate2subjects: dict[SubjectId, list[SubjectInfo]]  # no-use

    def _load_note_indexes(self):
        self.subjects = []
        self.cate2subjects = defaultdict(list)
        for path in self._fp_notes.iterdir():
            if not RE.note_name.match(path.name):
                continue
            _prefix, name = path.stem.split('-')
            _subject = SubjectInfo(_prefix, name, path)
            self.subjects.append(_subject)
            sid = self.property.subject_ids[_subject.subject_id]
            self.cate2subjects[sid].append(_subject)

        # sort
        for v in self.cate2subjects.values():
            v.sort(key=lambda s: s.subject_number)

    readme_toc: str
    readme_concat: str

    def build(self):
        with self._fp_notes_readme_temp.open(encoding='utf8') as f:
            tmp = f.read()

        contents = {s.toc_id: s.toc for s in self.subjects}
        readme = tmp.format(**contents)

        with self._fp_notes_readme.open('w', encoding='utf8') as f:
            f.write(readme)

        self.readme_toc = self.get_readme_toc()
        self.readme_concat = self.get_readme_concat()

    def get_readme_toc(self):
        with self._fp_notes_readme.open(encoding='utf8') as f:
            return RE.note_toc.search(f.read()).group(1).strip()

    def get_readme_concat(self):
        with self._fp_notes_readme.open(encoding='utf8') as f:
            content = RE.note_content.search(f.read()).group(1).strip()
            return content.replace('](', f']({self._fp_notes.name}/')


if __name__ == '__main__':
    """"""
    note = Notes()
    note.build()
