#!/bin/sh

if (($# < 1)); then
  echo "Usage: sh $0 front or back" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate KQAPro # 或者bash窗口输入 kqap打开虚拟环境

OWN_FOLDER_DIR=$(cd "$(dirname "$0")" || exit;pwd)

# 启动可视化在线服务
VIMPORT=8097
sh $OWN_FOLDER_DIR/run_visdom_service.sh $VIMPORT

cd $OWN_FOLDER_DIR/../ || exit

SHOW=$1
TIME=$(date "+%Y%m%d-%H%M%S")

if [ $SHOW = "front" ]; then  # 前台训练
  python -m Bart_Program.train
elif [ $SHOW = "back" ]; then  # 后台训练
  FILE=$TIME"_train.log"
  nohup python -m Bart_Program.train  > ./$FILE 2>&1 &
  tail -f ./$FILE

elif [ $SHOW = "front-a" ]; then  # 前台训练
  python -m Bart_Program.train_multi_adapter
elif [ $SHOW = "back-a" ]; then  # 后台训练
  FILE=$TIME"_train.log"
  nohup python -m Bart_Program.train_multi_adapter  > ./$FILE 2>&1 &
  tail -f ./$FILE

elif [ $SHOW = "eval" ]; then  # 后台训练
  nohup python -m Bart_Program.eval_bart > ./eval.log 2>&1 &
  tail -f eval.log
elif [ $SHOW = "data" ]; then  # 构建数据
  python -m Bart_Program.preprocess
fi
