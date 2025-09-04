#!/bin/bash
# chmod +x infer_hbb.sh
set -e  # 遇到错误立即退出

# 日志目录（自动创建）
LOGDIR="./logs_hbb"
OUTPUT_DIR_HBB="./output_hbb"
mkdir -p "$LOGDIR"
mkdir -p "$OUTPUT_DIR_HBB"

# 日志文件名（自动加日期）
LOGFILE="$LOGDIR/run_$(date +'%Y-%m-%d_%H-%M-%S').log"

echo "======================================"
echo " Run started at $(date)"
echo " Logs will be saved to $LOGFILE"
echo "======================================"

# 把后续所有输出都重定向到日志文件，同时在终端显示
exec > >(tee -a "$LOGFILE") 2>&1

# 编译阶段
# echo "[INFO] 开始编译源代码..."
# cd ./src
# cmake .
# make
# echo "[INFO] 编译完成."

# # 推理阶段
echo "[INFO] 开始推理..."
cd ./out
./main_hbb
echo "[INFO] 推理完成."

# # 精度评估阶段
# echo "[INFO] 开始评估 mAP..."
# cd ..
# python3 val_obb.py
# echo "[INFO] 评估完成."

# echo "======================================"
# echo " Run finished at $(date)"
# echo " Logs saved to $LOGFILE"
# echo "======================================"
