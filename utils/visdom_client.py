# -*- coding: utf-8 -*-
from visdom import Visdom

from conf.config import ENV, USE_INCOMING_SOCKET, OTHER_CONFIG
from utils.logger import logger


class VisdomClient(Visdom):

    def __init__(self, env, use_incoming_socket, **kwargs):
        super(VisdomClient, self).__init__(
            env=env,
            use_incoming_socket=use_incoming_socket,
            **kwargs
        )


try:
    visdom_client = VisdomClient(ENV, USE_INCOMING_SOCKET, **OTHER_CONFIG)
    assert visdom_client.check_connection(), '服务连接失败'
except Exception as e:
    logger.warning('Failed to connect to visualization service， error: {}'.format(e))
    visdom_client = None
