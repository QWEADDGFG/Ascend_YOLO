#!/bin/bash

set -e  # 出错就退出

# 提交信息，带上日期
msg="update $(date +'%Y%m%d %H:%M:%S')"
if [ -n "$1" ]; then
  msg="$1"
fi

echo "[INFO] 添加所有更改..."
git add .

echo "[INFO] 提交: $msg"
git commit -m "$msg" || echo "[WARN] 没有更改需要提交"

echo "[INFO] 推送到远端 main 分支..."
git push origin main

echo "[INFO] ✅ 推送完成"
