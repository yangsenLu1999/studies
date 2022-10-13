#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time:
    2022-10-13 20:58
Author:
    huayang (imhuay@163.com)
Subject:
    _base
"""
from __future__ import annotations

# import os
# import sys
# import json
# import unittest

# from typing import *
# from pathlib import Path
# from collections import defaultdict


class Readme:

    def build(self):
        raise NotImplementedError


class Build:

    def __init__(self, *readmes: Readme):
        self.readmes = readmes

    def build(self):
        [r.build() for r in self.readmes]
