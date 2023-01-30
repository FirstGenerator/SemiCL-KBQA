# coding: utf-8

import os
import sys
import logging
import inspect
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from conf.setting import BartSetting
from conf.config import ROOT_DIR, LOG_DIR, LOG_LEVEL, LOG_FILE_HANDLER, IS_DEBUG_MODE, DEVICE_NUM

if not hasattr(sys.modules[__name__], '__file__'):
    __file__ = inspect.getfile(inspect.currentframe())

logging.getLogger("elasticsearch").setLevel(logging.WARNING)


class Level:
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARN = logging.WARNING
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET


name_2_level = {
    'CRITICAL': logging.CRITICAL,
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}

MAX_BYTES = 50 * 1024 * 1024
BACKUP_COUNT = 10
WHEN = 'D'
INTERVAL = 1
FMT = '%(asctime)s %(levelname)8s %(filename)28s %(funcName)32s %(lineno)4s - %(message)s'


class MyLogger:
    def __init__(self, log_dir=None, log_level=logging.DEBUG):
        self._log_dir = log_dir if log_dir else os.path.join(os.path.dirname(__file__), 'logs')
        if not os.path.isdir(self._log_dir):
            os.makedirs(self._log_dir)
        time = datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M:%S')
        name = f'{BartSetting.model_strategy}_{BartSetting.work_dir_suffix}_{DEVICE_NUM}_{time}.log'
        self._logger = logging.getLogger(name)
        self._logger.setLevel(log_level)

        formatter = logging.Formatter(FMT)
        if IS_DEBUG_MODE:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(stream_handler)

        path = os.path.abspath(os.path.join(self._log_dir, name))
        if LOG_FILE_HANDLER == 'TimedRotatingFileHandler':
            file_handler = TimedRotatingFileHandler(path, when=WHEN, interval=INTERVAL, backupCount=BACKUP_COUNT,
                                                    encoding='utf-8')
        elif LOG_FILE_HANDLER == 'RotatingFileHandler':
            file_handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8')
        else:
            raise ValueError('None valid FileHandler')
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        self._bind_methods()

    def get_logger(self):
        return self._logger

    def add_handler(self, hdlr):
        self._logger.addHandler(hdlr)

    def remove_handler(self, hdlr):
        self._logger.removeHandler(hdlr)

    def _bind_methods(self):
        self.debug = getattr(self._logger, 'debug')
        self.info = getattr(self._logger, 'info')
        self.warn = getattr(self._logger, 'warning')
        self.warning = getattr(self._logger, 'warning')
        self.exception = getattr(self._logger, 'exception')
        self.error = getattr(self._logger, 'error')
        self.critical = getattr(self._logger, 'critical')


logger = MyLogger(log_dir=os.path.join(ROOT_DIR, LOG_DIR), log_level=name_2_level[LOG_LEVEL])
