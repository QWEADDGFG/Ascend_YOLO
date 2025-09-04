import os
import numpy as np
from pathlib import Path
import torch
import sys

# 如果你有自定义的 batch_probiou 和 OBBMetrics，可按需要调整导入路径
sys.path.insert(0, "./")
from metrics import batch_probiou, OBBMetrics

# ================== 配置 ==================
GT_DIR = Path("/home/HwHiAiUser/gp/DATASETS/MVRSD/txt")
PRED_DIR = Path("./output_obb/labels")
IMG_W, IMG_H = 640, 640
LABELS = ["AFV", "CV", "LMV", "MCV", "SMV"]
LABELS_IDX = {name: i for i, name in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)
# =========================================


def load_gt_obb_label(txt_path):
    """读取GT OBB标签: class x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标)
       GT 中 class 是数字索引（0..）或是类名，也都支持。
    """
    if not txt_path.exists() or os.path.getsize(txt_path) == 0:
        return np.zeros((0, 9))
    lines = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 9:
            continue
        cls = parts[0]
        # 支持类名或类索引
        try:
            cls_idx = int(float(cls))
        except Exception:
            cls_idx = LABELS_IDX.get(cls, None)
            if cls_idx is None:
                raise ValueError(f"未知的GT类名: {cls} in {txt_path}")
        coords = list(map(float, parts[1:9]))
        out.append([cls_idx] + coords)
    return np.array(out, dtype=float) if out else np.zeros((0, 9))


def load_pred_obb_label(txt_path):
    """读取预测OBB标签: x1 y1 x2 y2 x3 y3 x4 y4 CLASS CONF
       CLASS 可以是类名或类别索引；CONF 为 float。
       返回 (N, 10) 的 numpy 数组：8 coords (float), cls_idx (int), conf (float)
    """
    if not txt_path.exists() or os.path.getsize(txt_path) == 0:
        return np.zeros((0, 10))
    lines = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 10:
            # 若顺序是 coords + conf + class_name，可以尝试另一种解析
            # 但按照你给出的示例，最后是 CLASS CONF
            continue
        try:
            coords = list(map(float, parts[0:8]))
        except Exception as e:
            raise ValueError(f"无法将前8个坐标解析为float: {ln}") from e

        cls_field = parts[8]
        conf_field = parts[9]

        # 解析 class：支持类名或数字
        try:
            cls_idx = int(float(cls_field))
        except Exception:
            cls_idx = LABELS_IDX.get(cls_field, None)
            if cls_idx is None:
                raise ValueError(f"未知的预测类名: {cls_field} in {txt_path}")

        # 解析置信度
        try:
            conf = float(conf_field)
        except Exception:
            # 有时格式是 "SMV 0.912109" 的话 parts[8] = 'SMV' parts[9] = '0.912109' 正常解析
            raise ValueError(f"无法解析预测置信度为float: {conf_field} in {txt_path}")

        out.append(coords + [cls_idx, conf])
    return np.array(out, dtype=float) if out else np.zeros((0, 10))


def normalize_pred_coords(pred_coords, img_w, img_h):
    """将预测的像素坐标归一化到[0,1]"""
    normalized = pred_coords.copy()
    # x坐标归一化
    normalized[:, [0, 2, 4, 6]] /= img_w
    # y坐标归一化
    normalized[:, [1, 3, 5, 7]] /= img_h
    return normalized


def obb8_to_obb5(obb8):
    """
    将8点OBB格式(x1,y1,x2,y2,x3,y3,x4,y4)转换为5点格式(cx,cy,w,h,angle)
    """
    if obb8.shape[0] == 0:
        return np.zeros((0, 5))
    x1, y1, x2, y2, x3, y3, x4, y4 = obb8.T
    cx = (x1 + x2 + x3 + x4) / 4.0
    cy = (y1 + y2 + y3 + y4) / 4.0
    edge1_len = np.hypot(x2 - x1, y2 - y1)
    edge2_len = np.hypot(x3 - x2, y3 - y2)
    w = edge1_len
    h = edge2_len
    angle = np.arctan2(y2 - y1, x2 - x1)
    return np.column_stack([cx, cy, w, h, angle])


def calculate_obb_iou(gt_obb5, pred_obb5):
    if len(gt_obb5) == 0 or len(pred_obb5) == 0:
        return torch.zeros((len(gt_obb5), len(pred_obb5)))
    gt_tensor = torch.tensor(gt_obb5, dtype=torch.float32)
    pred_tensor = torch.tensor(pred_obb5, dtype=torch.float32)
    iou = batch_probiou(gt_tensor, pred_tensor)
    return iou


# ================== 评估 ==================
all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []

print("开始OBB评估...")
processed_files = 0

for gt_file in os.listdir(GT_DIR):
    if not gt_file.endswith(".txt"):
        continue

    gt_path = GT_DIR / gt_file
    pred_path = PRED_DIR / gt_file

    gt_labels = load_gt_obb_label(gt_path)        # [N, 9]: class + 8点坐标 (归一化)
    pred_labels = load_pred_obb_label(pred_path)  # [M, 10]: 8点坐标 + cls_idx + conf

    # 解析GT
    if gt_labels.shape[0] > 0:
        gt_cls = gt_labels[:, 0].astype(int)
        gt_coords = gt_labels[:, 1:9]  # 已归一化
        gt_obb5 = obb8_to_obb5(gt_coords)
    else:
        gt_cls = np.array([], dtype=int)
        gt_obb5 = np.zeros((0, 5))

    # 解析预测
    if pred_labels.shape[0] > 0:
        pred_coords = pred_labels[:, :8]  # 像素坐标
        pred_cls = pred_labels[:, 8].astype(int)
        pred_conf = pred_labels[:, 9].astype(float)

        # 归一化预测坐标
        pred_coords_norm = normalize_pred_coords(pred_coords, IMG_W, IMG_H)
        pred_obb5 = obb8_to_obb5(pred_coords_norm)
    else:
        pred_cls = np.array([], dtype=int)
        pred_conf = np.array([], dtype=float)
        pred_obb5 = np.zeros((0, 5))

    # 计算IoU
    iou = calculate_obb_iou(gt_obb5, pred_obb5)

    # 计算TP（IoU阈值从0.5到0.95，步长0.05）
    correct = np.zeros((pred_obb5.shape[0], 10), dtype=bool)

    if len(gt_obb5) > 0 and len(pred_obb5) > 0:
        for i in range(10):
            thresh = 0.5 + i * 0.05
            matches = iou >= thresh

            # 为每个预测框找匹配的GT（这里只判断是否有匹配，未做一对一最大匹配）
            for p in range(pred_obb5.shape[0]):
                # 匹配的GT中类别需相同
                matched = matches[:, p]
                if matched.any():
                    # 检查匹配的 GT 中是否存在相同类别
                    gt_indices = np.where(matched)[0]
                    # 若 GT 中包含类别信息，则只在相同类别中计为 TP
                    if len(gt_cls) > 0:
                        # 如果有任一匹配GT类别等于 pred_cls[p]，视作匹配
                        if any(gt_cls[gi] == pred_cls[p] for gi in gt_indices):
                            correct[p, i] = True
                    else:
                        correct[p, i] = True

    # 收集结果
    all_tp.append(correct)
    all_conf.append(pred_conf)
    all_pred_cls.append(pred_cls)
    all_target_cls.append(gt_cls)

    processed_files += 1
    if processed_files % 50 == 0:
        print(f"已处理 {processed_files} 个文件...")

# 拼接所有结果
tp = np.concatenate(all_tp, 0) if all_tp else np.zeros((0, 10))
conf = np.concatenate(all_conf, 0) if all_conf else np.array([])
pred_cls = np.concatenate(all_pred_cls, 0) if all_pred_cls else np.array([])
target_cls = np.concatenate(all_target_cls, 0) if all_target_cls else np.array([])

print(f"总共处理了 {processed_files} 个文件")
print(f"预测框总数: {len(pred_cls)}")
print(f"GT框总数: {len(target_cls)}")

# 计算OBB mAP
names_dict = {i: LABELS[i] for i in range(NUM_CLASSES)}
metrics = OBBMetrics(names=names_dict)
metrics.process(tp, conf, pred_cls, target_cls)

print("\n========== OBB评估结果 ==========")
print(f"类别标签: {LABELS}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

# 分类别结果
print("\n========== 各类别详细结果 ==========")
print(f"{'Class':<8} {'P':<8} {'R':<8} {'mAP50':<8} {'mAP50-95':<8}")
print("-" * 45)
for i in range(NUM_CLASSES):
    if i < len(metrics.box.p):
        p, r, ap50, ap = metrics.class_result(i)
        print(f"{LABELS[i]:<8} {p:.3f}    {r:.3f}    {ap50:.3f}    {ap:.3f}")
    else:
        print(f"{LABELS[i]:<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")

print("=" * 50)
print("OBB评估完成！")
