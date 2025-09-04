# YOLO部署到华为昇腾AI处理器

## 1. 准备工作

### 1.1 准备环境
```bash
npu-smi info
+--------------------------------------------------------------------------------------------------------+
| npu-smi 23.0.rc3                                 Version: 23.0.rc3                                     |
+-------------------------------+-----------------+------------------------------------------------------+
| NPU     Name                  | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
| Chip    Device                | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
+===============================+=================+======================================================+
| 0       310B1                 | OK              | 9.2          53                15    / 15            |
| 0       0                     | NA              | 0            3841 / 11577                            |
+===============================+=================+======================================================+
```
```bash

.
├── LICENSE
├── README.md
├── convert.py
├── git.sh
├── infer_hbb.sh
├── infer_obb.sh
├── metrics.py
├── model
│   ├── YOLO11n_p2_hbb_IRSTD_1K_512.om
│   ├── YOLO11s_base_obb_MVRSD_640.om
│   ├── aipp512.cfg
│   └── aipp640.cfg
├── run_compile.sh
├── src
│   ├── CMakeLists.txt
│   ├── YOLO_hbb.cpp
│   └── YOLO_obb.cpp
├── val_hbb.py
└── val_obb.py

2 directories, 17 files
```

### 1.2 准备数据集与模型
```shell
- HBB数据集，IRSTD_1K，红外弱小目标数据集，类别只有SO一类，在PC机上训练得到 YOLO11n_p2_hbb_IRSTD_1K_512.onnx模型。
- OBB数据集，MVRSD，旋转框多类别车辆数据集，类别有5类，在PC机上训练得到 YOLO11s_base_obb_MVRSD_640.onnx模型。
- 视频数据集，[TODO]
```

```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples

if __name__ == '__main__':
    model = YOLO('/data/gy/gp/Huawei/yolo11obb/runs/train/yolo11sobb_MVRSD2/weights/best.pt')
    model.export(format='onnx', simplify=True, opset=11, dynamic=True, imgsz=640, nms=False)
```

## 2. 模型转换

### 2.1 onnx转换为om

```shell
atc --model=YOLO11n_p2_hbb_IRSTD_1K_512.onnx --framework=5 --output=YOLO11n_p2_hbb_IRSTD_1K_512 --input_shape="images:1,3,512,512"  --soc_version=Ascend310B1  --insert_op_conf=aipp512.cfg
atc --model=YOLO11s_base_obb_MVRSD_640.onnx --framework=5 --output=YOLO11s_base_obb_MVRSD_640 --input_shape="images:1,3,640,640"  --soc_version=Ascend310B1  --insert_op_conf=aipp640.cfg
```
aipp640.cfg做了三件事：
1. YUV420SP → RGB 颜色空间转换；
2. 裁剪成 640×640；
3. 归一化到 [0,1]（通过除以 255）。

```cfg
aipp_op{
    aipp_mode:static                 # AIPP 模式（static 静态模式）
    input_format : YUV420SP_U8       # 输入图像格式（NV12/YUV420SP）
    src_image_size_w : 640           # 输入图像宽度
    src_image_size_h : 640           # 输入图像高度

    csc_switch : true                # 是否启用颜色空间转换（YUV → RGB）
    rbuv_swap_switch : false         # 是否交换 U/V 通道
    # CSC 转换矩阵（从 YUV 到 RGB 的转换系数）
    matrix_r0c0 : 256
    matrix_r0c1 : 0
    matrix_r0c2 : 359
    matrix_r1c0 : 256
    matrix_r1c1 : -88
    matrix_r1c2 : -183
    matrix_r2c0 : 256
    matrix_r2c1 : 454
    matrix_r2c2 : 0
    # YUV 偏置（常用于去除色度偏移）
    input_bias_0 : 0
    input_bias_1 : 128
    input_bias_2 : 128

    crop: true                       # 是否开启裁剪
    load_start_pos_h : 0             # 裁剪起始点（高度方向）
    load_start_pos_w : 0             # 裁剪起始点（宽度方向）
    crop_size_w : 640                # 裁剪宽度
    crop_size_h : 640                # 裁剪高度

    # 归一化参数（min/max 值 & 方差倒数）
    min_chn_0 : 0                    # 通道 0 的最小值
    min_chn_1 : 0                    # 通道 1 的最小值
    min_chn_2 : 0                    # 通道 2 的最小值
    var_reci_chn_0: 0.0039215686274509803921568627451   # 通道 0 方差倒数 = 1/255
    var_reci_chn_1: 0.0039215686274509803921568627451   # 通道 1 方差倒数 = 1/255
    var_reci_chn_2: 0.0039215686274509803921568627451   # 通道 2 方差倒数 = 1/255
}

```
### 2.2 测试集图片格式要求

```bash
[ERROR]  unsupported format, only Baseline JPEG
[ERROR]  ReadJpeg failed, errorCode is 1
[ERROR]  ProcessInput image failed, errorCode is 1
```

原因：Progressive JPEG 不是 Baseline，某些库（如老旧 OpenCV、嵌入式系统）不支持

解决办法：
python convert.py 

## 3. 编译推理
### 3.1 编译
```bash
chmod +x run_compile.sh
./run_compile.sh
[INFO] 开始编译源代码...
-- The C compiler identification is GNU 11.3.0
-- The CXX compiler identification is GNU 11.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- set INC_PATH: /usr/local/Ascend/ascend-toolkit/latest
-- set LIB_PATH: /usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub
-- set THIRDPART: /usr/local/Ascend/ascend-toolkit/latest/thirdpart
-- Configuring done
-- Generating done
-- Build files have been written to: /home/HwHiAiUser/gp/Ascend_YOLO/src
[ 25%] Building CXX object CMakeFiles/main_hbb.dir/YOLO_hbb.cpp.o
[ 50%] Linking CXX executable /home/HwHiAiUser/gp/Ascend_YOLO/out/main_hbb
[ 50%] Built target main_hbb
[ 75%] Building CXX object CMakeFiles/main_obb.dir/YOLO_obb.cpp.o
[100%] Linking CXX executable /home/HwHiAiUser/gp/Ascend_YOLO/out/main_obb
[100%] Built target main_obb
[INFO] 编译完成.
```

### 3.2 推理
```bash
./run_infer_hbb.sh
./run_infer_obb.sh
```

## 4. 精度结果解析
```bash
python val_hbb.py
========== 评估结果 ==========
Precision: [    0.90542]
Recall: [    0.81481]
mAP@0.5: 0.8786495761295028
mAP@0.5:0.95: 0.4567271853301881
================================

python val_obb.py
开始OBB评估...
已处理 50 个文件...
已处理 100 个文件...
已处理 150 个文件...
已处理 200 个文件...
已处理 250 个文件...
已处理 300 个文件...
已处理 350 个文件...
已处理 400 个文件...
已处理 450 个文件...
已处理 500 个文件...
已处理 550 个文件...
已处理 600 个文件...
总共处理了 600 个文件
预测框总数: 7302
GT框总数: 6227

========== OBB评估结果 ==========
类别标签: ['AFV', 'CV', 'LMV', 'MCV', 'SMV']
Precision: [     0.8092     0.92974     0.81982     0.37506     0.83685]
Recall: [    0.83908     0.96576     0.85463     0.47059     0.77915]
mAP@0.5: 0.7942
mAP@0.5:0.95: 0.7782

========== 各类别详细结果 ==========
Class    P        R        mAP50    mAP50-95
---------------------------------------------
AFV      0.809    0.839    0.884    0.877
CV       0.930    0.966    0.981    0.975
LMV      0.820    0.855    0.891    0.846
MCV      0.375    0.471    0.362    0.342
SMV      0.837    0.779    0.853    0.851
==================================================
OBB评估完成！


```

## 5 常见问题
[Q]：
检测框混乱，重叠，置信度大于1
[A]：
原因：
PyTorch: starting from '/data/gy/gp/Huawei/ultralytics-main/runs/train/yolov8s-repvit/weights/best.pt' with input shape (1, 3, 512, 512) BCHW and output shape(s) (1, 5, 5376) (39.5 MB) 
PyTorch: starting from '/data/gy/gp/Huawei/YOLOv11_Final_20250705/dataconfig/irstd1k_v11shbb_640_nextvit/weights/best.pt' with input shape (1, 3, 512, 512) BCHW and output shape(s) (1, 5376, 6) (73.6 MB)
模型输出output shape(s) 不一致，不同模型的输出不一致

(1, 3, 512, 512) -> (1, 5, 5376)
batch = 1
每个预测有 5376 个 anchor/grid
每个预测输出 5 个值（比如 x, y, w, h, obj）
channel-first

(1, 3, 512, 512) -> (1, 5376, 6)
batch = 1
有 5376 个预测框
每个预测框输出 6 个值（比如 x, y, w, h, conf, cls）
prediction-first

另外，对于旋转框，其输出格式为：([cx, cy, w, h, cls1..clsN, angle])