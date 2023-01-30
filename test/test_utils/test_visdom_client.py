import time
import torch as t
import numpy as np

from utils.visdom_client import visdom_client

#-------------------------------
for global_steps in range(10):
    #----------------------------
    #  randomly get loss and acc （设置显示数据）
    loss = 0.1 * np.random.randn() + 1
    acc = 0.1 * np.random.randn() + 0.5
    #  update window image （传递数据到监听窗口进行画图）
    visdom_client.line([[loss, acc]], [global_steps], win='train', update='append', opts=dict(title='loss&acc', legend=['loss', 'acc']))
    #-----------------------------
    #  delay time 0. 5s
    time.sleep(0.5)
