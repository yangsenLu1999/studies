#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time:
    2022-08-01 17:32
Author:
    HuaYang (imhuay@163.com)
Subject:
    A wrapper of AutoPhrase：命令行调用 AutoPhrase 的简单包装
References:
    - https://github.com/shangjingbo1226/AutoPhrase
    - pip install autophrase
"""
import os
import sys
import json
import time
import shutil

from typing import *
from pathlib import Path
from collections import defaultdict

from huaytools.utils import get_logger
from huaytools.utils.singleton import singleton

_logger = get_logger()

_Official_Language = ['EN', 'CN', 'JA', 'ES', 'AR', 'FR', 'IT', 'RU']
_ENV_AUTOPHRASE = 'AUTOPHRASE'

Language = Literal['EN', 'CN', 'JA', 'ES', 'AR', 'FR', 'IT', 'RU', 'OTHER']
LanguageArabic = 'AR'
"""
EN: English
FR: French
DE: German
ES: Spanish
IT: Italian
PT: Portuguese
RU: Russian
AR: Arabic
CN: Chinese
JA: Japanese
"""
LIB_PATH = None
_TokenizeMode = Literal['train', 'test', 'translate']


# @singleton
class AutoPhrase:
    """"""
    _lib_path: ClassVar
    _tokenizer: ClassVar[str] = None

    def __init__(self,
                 model_dir,
                 fp_train_data=None,
                 fp_stopwords=None,
                 fp_seed_words=None,
                 fp_quality_seed_words=None,
                 language: Language = 'CN',
                 enable_postag: bool = False,
                 highlight_multi: float = 0.5,
                 highlight_single: float = 0.8,
                 lowercase: bool = True,
                 thread: int = 10,
                 min_sup: int = 10,
                 lib_path: Union[str, Path] = None):
        """"""
        self._lib_path = lib_path
        self._model_dir = Path(model_dir)
        self._fp_train_data = fp_train_data
        self._fp_stopwords = fp_stopwords
        self._fp_seed_words = fp_seed_words
        self._fp_quality_seed_words = fp_quality_seed_words
        self._language = language
        self._lowercase = lowercase
        self._thread = thread

        self._check_args()

        if self._language == LanguageArabic:
            self.tagger_model = self.lib_path + "/tools/models/arabic.tagger"

        self._fp_tmp = self._model_dir / 'tmp'
        self._fp_tokenized_train = self._fp_tmp / 'tokenized_train.txt'
        self._fp_tokenized_stopwords = self._fp_tmp / 'tokenized_stopwords.txt'
        self._fp_tokenized_seed_words = self._fp_tmp / 'tokenized_seed_words.txt'
        self._fp_tokenized_quality_seed_words = self._fp_tmp / 'tokenized_quality_seed_words.txt'
        self._fp_token_mapping = self._fp_tmp / 'token_mapping.txt'

    def _check_args(self):
        assert self._fp_train_data is not None or self._model_dir.exists()
        if self._model_dir.exists():
            assert self._model_dir.is_dir()
        if self._language not in _Official_Language:
            _logger.warning(f'{self._language} is not official supported language')
        assert self._thread > 0

    def train(self, ):
        """"""
        self._tokenize_train_data()
        self._tokenize_stopwords()
        self._tokenize_seed_words()
        self._tokenize_quality_seed_words()
        # TODO: continue

    def _tokenize_train_data(self):
        """"""
        _logger.info(AutoPhrase._get_start_log())
        command = self._get_tokenize_command(self._fp_train_data,
                                             self._fp_tokenized_train,
                                             self._fp_token_mapping,
                                             mode='train',
                                             lowercase=self._lowercase,
                                             language=self._language)
        self._run_command(command)

    def _tokenize_stopwords(self):
        """"""
        _logger.info(AutoPhrase._get_start_log())
        command = self._get_tokenize_command(self._fp_stopwords,
                                             self._fp_tokenized_stopwords,
                                             self._fp_token_mapping,
                                             mode='test',
                                             lowercase=self._lowercase,
                                             language=self._language)
        self._run_command(command)

    def _tokenize_seed_words(self):
        """"""
        _logger.info(AutoPhrase._get_start_log())
        command = self._get_tokenize_command(self._fp_seed_words,
                                             self._fp_tokenized_seed_words,
                                             self._fp_token_mapping,
                                             mode='test',
                                             lowercase=self._lowercase,
                                             language=self._language)
        self._run_command(command)

    def _tokenize_quality_seed_words(self):
        """"""
        command = self._get_tokenize_command(self._fp_quality_seed_words,
                                             self._fp_tokenized_quality_seed_words,
                                             self._fp_token_mapping,
                                             mode='test',
                                             lowercase=self._lowercase,
                                             language=self._language)
        self._run_command(command)

    @classmethod
    def load(cls, model_path):
        """"""

    def _get_tokenize_command(self,
                              input_file,
                              output_file,
                              mapping_file,
                              mode: _TokenizeMode,
                              lowercase: bool = True,
                              language: Language = None):
        """"""
        case_sensitive = 'N' if lowercase else 'Y'
        command = f'java {self.tokenizer}' \
                  f' -m {mode}' \
                  f' -i {input_file}' \
                  f' -o {output_file}' \
                  f' -t {mapping_file}' \
                  f' -c {case_sensitive}' \
                  f' -thread {self._thread}'

        if language == LanguageArabic:
            command = f'{command} -tagger_model {self.tagger_model}'

        return command

    @staticmethod
    def _run_command(command):
        return os.system(command)

    def tokenize(self,
                 input_file,
                 output_file,
                 mapping_file,
                 mode: _TokenizeMode,
                 language: Language = 'CN',
                 lowercase: bool = True,
                 thread: int = 10):
        """"""
        case_sensitive = 'N' if lowercase else 'Y'
        command = f'java {self.tokenizer}' \
                  f' -m {mode}' \
                  f' -i {input_file}' \
                  f' -o {output_file}' \
                  f' -t {mapping_file}' \
                  f' -c {case_sensitive}' \
                  f' -thread {thread}'

        if language == LanguageArabic:
            command = f'{command} -tagger_model {self.tagger_model}'

    @property
    def lib_path(self):
        if self._lib_path is None:
            self._lib_path = AutoPhrase.find_lib_path()
        return self._lib_path

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            tokenizer_path = f'{self.lib_path}:' \
                             f'{self.lib_path}/tools/tokenizer/lib/*:' \
                             f'{self.lib_path}/tools/tokenizer/resources/:' \
                             f'{self.lib_path}/tools/tokenizer/build/'
            self._tokenizer = f'-cp {tokenizer_path} Tokenizer'
        return self._tokenizer

    @staticmethod
    def find_lib_path() -> Path:
        """"""

        def try_import():
            try:
                import autophrase  # noqa
            except ImportError:
                _p = None
            else:
                _p = Path(Path(autophrase.__file__).parent)
            return _p

        if _ENV_AUTOPHRASE in os.environ:
            lib_path = os.environ[_ENV_AUTOPHRASE]
        elif p := try_import():
            lib_path = p
        else:
            raise ValueError(f'Not find autophrase. '
                             f'Please `pip install autophrase` or '
                             f'download from "https://github.com/shangjingbo1226/AutoPhrase"')
        return Path(lib_path)

    @staticmethod
    def _get_start_log() -> str:
        from huaytools.utils import get_caller_name
        return f'=== Start {get_caller_name()} ==='


class __RunWrapper:
    """"""

    def __init__(self):
        """"""
        for k, v in self.__class__.__dict__.items():
            if k.startswith('demo') and isinstance(v, Callable):
                print(f'=== Start "{k}" {{')
                start = time.time()
                v(self)
                print(f'}} End "{k}" - Spend {time.time() - start:5f}s===\n')

    def demo_doctest(self):  # noqa
        """"""
        import doctest
        doctest.testmod()

    def _demo_AutoPhrase(self):  # noqa
        """"""
        # import autophrase
        # p = Path(autophrase.__file__)
        # print(p.parent)
        a = AutoPhrase('', language='c')

    def _demo_download(self):  # noqa
        """"""
        from huaytools.utils import cprint
        cprint('Downloading pretrained model ...')
        import urllib
        DBLP_MODEL = 'https://github.com/CS512-Autophrase-Demo/AutoPhrase/blob/master/models/DBLP/segmentation.model?raw=true'
        segmentation_model = Path('./tmp/segmentation.model')
        segmentation_model.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DBLP_MODEL, segmentation_model)


if __name__ == '__main__':
    """"""
    __RunWrapper()
    # print(os.environ['AUTOPHRASE'])
