# -*- coding: utf-8 -*-
from typing import List

from utils.logger import logger
from utils.visdom_client import visdom_client


def verify(func):
    def wrapper(self, *args, **kwargs):
        if self.visdom_client:  # 检验visdom_client是否可用
            return func(self, *args, **kwargs)
        # logger.warning('visdom_client连接失败')
    return wrapper


class Visualizer:
    def __init__(self):
        self.visdom_client = visdom_client

    @verify
    def set_line_chart(self, x, y, win, title, update='append', xlabel=None, ylabel=None, legend: List = None):
        opts = {'title': title, 'xlabel': xlabel, 'ylabel': ylabel}
        if legend:
            opts['legend'] = legend
        self.visdom_client.line(X=x, Y=y, win=win, update=update, opts=opts)