import os
import numpy as np
from pathlib import Path
import torch
import sys

# 优先用工程目录下的 metrics.py
sys.path.insert(0, "/home/HwHiAiUser/gp/YOLO")
from metrics import box_iou, DetMetrics

# ================== 配置 ==================
pred_dir = Path("/home/HwHiAiUser/gp/YOLO/output/labels")
gt_dir   = Path("/home/HwHiAiUser/gp/YOLO/DATASETS/IRSTD_1K/txts_test")

NUM_CLASSES = 1   # 类别数
# =========================================


def load_gt_label(txt_path):
    """读取 GT 标签: class x y w h"""
    if not txt_path.exists() or os.path.getsize(txt_path) == 0:
        return np.zeros((0, 5))
    return np.loadtxt(txt_path, ndmin=2)


def load_pred_label(txt_path):
    """读取预测标签: class x y w h conf"""
    if not txt_path.exists() or os.path.getsize(txt_path) == 0:
        return np.zeros((0, 6))
    return np.loadtxt(txt_path, ndmin=2)


def xywh2xyxy(x):
    """YOLO 格式 (cx,cy,w,h) -> xyxy"""
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


# ================== 评估 ==================
all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []

for gt_file in os.listdir(gt_dir):
    if not gt_file.endswith(".txt"):
        continue
    gt_path = gt_dir / gt_file
    pred_path = pred_dir / gt_file

    gt_labels = load_gt_label(gt_path)        # [N,5]
    pred_labels = load_pred_label(pred_path)  # [M,6]

    # 解析 GT
    if gt_labels.shape[0]:
        gt_cls = gt_labels[:, 0]
        gt_boxes = xywh2xyxy(gt_labels[:, 1:5])
    else:
        gt_cls = np.array([])
        gt_boxes = np.zeros((0, 4))

    # 解析预测
    if pred_labels.shape[0]:
        pred_cls = pred_labels[:, 0]
        pred_boxes = xywh2xyxy(pred_labels[:, 1:5])
        pred_conf = pred_labels[:, 5]
    else:
        pred_cls = np.array([])
        pred_boxes = np.zeros((0, 4))
        pred_conf = np.array([])

    # IoU 计算
    if len(gt_boxes) and len(pred_boxes):
        iou = box_iou(torch.tensor(gt_boxes), torch.tensor(pred_boxes))
    else:
        iou = torch.zeros((len(gt_boxes), len(pred_boxes)))

    # TP 记录（IoU=0.5:0.95）
    correct = np.zeros((pred_boxes.shape[0], 10), dtype=bool)
    for i in range(10):
        thresh = 0.5 + i * 0.05
        matches = iou >= thresh
        for p in range(pred_boxes.shape[0]):
            if matches[:, p].any():
                correct[p, i] = True

    all_tp.append(correct)
    all_conf.append(pred_conf)
    all_pred_cls.append(pred_cls)
    all_target_cls.append(gt_cls)

# 拼接
tp = np.concatenate(all_tp, 0) if all_tp else np.zeros((0, 10))
conf = np.concatenate(all_conf, 0) if all_conf else np.array([])
pred_cls = np.concatenate(all_pred_cls, 0) if all_pred_cls else np.array([])
target_cls = np.concatenate(all_target_cls, 0) if all_target_cls else np.array([])

# 计算 mAP
metrics = DetMetrics(names={i: str(i) for i in range(NUM_CLASSES)})
metrics.process(tp, conf, pred_cls, target_cls)

print("========== 评估结果 ==========")
print("Precision:", metrics.box.p)
print("Recall:", metrics.box.r)
print("mAP@0.5:", metrics.box.map50)
print("mAP@0.5:0.95:", metrics.box.map)
print("================================")
