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

import os
# import os
# import sys
# import json
# import unittest
import re
import subprocess

import yaml

from typing import ClassVar
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field

from readme.utils import args, ReadmeUtils, TEMP_main_readme_notes_recent_toc


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


# def _load_note_info(fp, txt):
#     m = RE.note_info.search(txt)
#     if not m:
#         raise ValueError(fp)
#     return yaml.safe_load(m.group(1).strip())


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


@dataclass
class NoteInfo:
    path: Path
    _info: dict = None
    _title: str = None
    _first_commit_date: str = None

    @property
    def title(self):
        if self._title is None:
            with self.path.open(encoding='utf8') as f:
                for ln in f:
                    self._title = ln.strip()
                    break

            if not self._title:
                self._title = f'Untitled-{self.path_relative_to_repo}'
        return self._title

    @property
    def info(self):
        if self._info is None:
            with self.path.open(encoding='utf8') as f:
                try:
                    m = RE.note_info.search(f.read())
                except:  # noqa
                    raise ValueError(self.path)
                if m:
                    self._info = yaml.safe_load(m.group(1).strip())
                else:
                    self._info = dict()
        return self._info

    @property
    def first_commit_date(self) -> str:
        if self._first_commit_date is None:
            self._first_commit_date = ReadmeUtils.get_file_first_commit_date(self.path)
        return self._first_commit_date

    @property
    def date(self):
        return self.first_commit_date[:10]

    @property
    def is_top(self):
        return self.info.get('top', False)

    @property
    def path_relative_to_repo(self):
        return self.path.relative_to(args.fp_repo)

    @property
    def toc_line_relative_to_repo(self):
        """"""
        if self.is_top:
            return f'- [`{self.date}` {self.title} 📌]({self.path_relative_to_repo})'
        else:
            return f'- [`{self.date}` {self.title}]({self.path_relative_to_repo})'

    @property
    def sort_key(self):
        # if self.title is None:
        #     raise ValueError(self.path)
        return self.first_commit_date, self.title


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
        self._fp_notes_archives = args.fp_notes_archives
        self._fp_notes_readme = args.fp_notes_readme
        self._fp_notes_readme_temp = args.fp_notes_readme_temp
        self._top_limit = args.notes_top_limit
        # self._recent_limit = args.notes_recent_limit
        self.property = Property(args.fp_notes_property)

        self._load_note_indexes()
        self._load_all_notes()

    subjects: list[SubjectInfo]
    cate2subjects: dict[SubjectId, list[SubjectInfo]]  # no-use
    fp2date: dict[Path, str]
    # recent_notes: list[Path]
    notes: list[NoteInfo] = []
    _notes_top: list[NoteInfo] = []
    _notes_recent: list[NoteInfo] = []

    @property
    def recent_limit(self):
        return

    @property
    def notes_top(self):
        return self._notes_top[:self._top_limit]

    @property
    def notes_recent(self):
        recent_limit = len(self.readme_toc.split('\n'))
        return self._notes_recent[:recent_limit - len(self.notes_top)]

    def _load_all_notes(self):
        for dp, _, fns in os.walk(self._fp_notes_archives):
            for fn in fns:
                fp = Path(dp) / fn
                if fp.suffix != '.md':
                    continue
                note_i = NoteInfo(fp)
                self.notes.append(note_i)
                if note_i.is_top:
                    self._notes_top.append(note_i)
                else:
                    self._notes_recent.append(note_i)

        self._notes_top.sort(key=lambda x: x.sort_key, reverse=True)
        self._notes_recent.sort(key=lambda x: x.sort_key, reverse=True)

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
    repo_recent_toc: str

    def build(self):
        with self._fp_notes_readme_temp.open(encoding='utf8') as f:
            tmp = f.read()

        contents = {s.toc_id: s.toc for s in self.subjects}
        readme = tmp.format(**contents)

        with self._fp_notes_readme.open('w', encoding='utf8') as f:
            f.write(readme)

        self.readme_toc = self._get_readme_toc()
        self.readme_concat = self._get_readme_concat()
        self.repo_recent_toc = self._build_repo_recent_toc()

    def _get_readme_toc(self):
        with self._fp_notes_readme.open(encoding='utf8') as f:
            return RE.note_toc.search(f.read()).group(1).strip()

    def _get_readme_concat(self):
        with self._fp_notes_readme.open(encoding='utf8') as f:
            content = RE.note_content.search(f.read()).group(1).strip()
            return content.replace('](', f']({self._fp_notes.name}/')

    def _build_repo_recent_toc(self):
        return TEMP_main_readme_notes_recent_toc.format(
            toc_top='\n'.join([n.toc_line_relative_to_repo for n in self.notes_top]),
            toc_recent='\n'.join([n.toc_line_relative_to_repo for n in self.notes_recent])
        )


if __name__ == '__main__':
    """"""
    note = Notes()
    note.build()
