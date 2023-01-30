#!/bin/bash

if [ "$1" == "" ]; then
    PORT=8097
else
    PORT=$1
fi

lsof -i:$PORT | awk '{print $2}' | sed -n '2, $p' | xargs kill -9
