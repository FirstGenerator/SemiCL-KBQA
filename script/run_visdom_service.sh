#!/bin/bash

if [ "$1" == "" ]; then
    PORT=8097
else
    PORT=$1
fi
echo $PORT
FILE=visdom_server.log
PID=`lsof -i:$PORT | awk '{print $1 " " $2}'`
if [ "$PID" == "" ]; then
    nohup python -m visdom.server -port $PORT > ./$FILE 2>&1 &
    echo "vimdom服务启动成功"
else
    echo "vimdom服务已启动"
fi
