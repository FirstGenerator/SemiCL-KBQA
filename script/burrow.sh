#!/bin/bash
IP=202.120.165.136
PORT=8097

lsof -i:$PORT | awk '{print $2}' | sed -n '2, $p' | xargs kill -9
ssh -t -p 1231 ubuntu@$IP -fNL $PORT:$IP:$PORT
