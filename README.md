## Ascend YOLO
#### 教程
1. 模型转换：
在PC机上执行如下python脚本将.pt模型转化成.onnx模型:

```python
# 环境要求：onnx onnxsim onnxruntime onnxruntime-gpu
# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/data/gy/gp/Huawei/YOLOv11_Final_20250705/dataconfig/irstd1k_v11shbb_640_nextvit/weights/best.pt')
    model.export(format='onnx', simplify=True, opset=11, dynamic=True, int8=True, imgsz=512, nms=True)
```

2. 将.onnx模型上传到Atlas_200I_DK_A2开发板上，执行如下命令将.onnx模型转化成.om模型：
```bash
atc --model=best0818.onnx --framework=5 --output=best0818 --input_shape="images:1,3,512,512"  --soc_version=Ascend310B1  --insert_op_conf=aipp.cfg
atc --model=bestv8s.onnx --framework=5 --output=bestv8s --input_shape="images:1,3,512,512"  --soc_version=Ascend310B1  --insert_op_conf=aipp.cfg
```
其中的:
--soc_version=Ascend310P3可以通过npu-smi info命令进行查看，
(base) root@davinci-mini:/home/HwHiAiUser/gp/YOLO# npu-smi info
+--------------------------------------------------------------------------------------------------------+
| npu-smi 23.0.rc3                                 Version: 23.0.rc3                                     |
+-------------------------------+-----------------+------------------------------------------------------+
| NPU     Name                  | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
| Chip    Device                | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
+===============================+=================+======================================================+
| 0       310B1                 | OK              | 10.7         58                15    / 15            |
| 0       0                     | NA              | 0            8983 / 11577                            |
+===============================+=================+======================================================+
这里打印的是 310B1 则，--soc_version 为 Ascend前缀加上310B1，即Ascend310B1。

--input_shape="images:1,3,512,512" 表示NCHW，即批处理为1，通道为3，图片大小为512x512 。

3. chmod +x run_infer.sh
执行 ./run_infer.sh
输出结果：
/home/HwHiAiUser/gp/YOLO/logs/run_2025-08-20_10-44-58.log

#### Q&A
1. 
[Q]：
```bash
[ERROR]  unsupported format, only Baseline JPEG
[ERROR]  ReadJpeg failed, errorCode is 1
[ERROR]  ProcessInput image failed, errorCode is 1
```
[A]：
原因：
Progressive JPEG 不是 Baseline，某些库（如老旧 OpenCV、嵌入式系统）不支持
解决办法：
python /home/HwHiAiUser/gp/YOLO/convert.py /home/HwHiAiUser/gp/YOLO/DATASETS/IRSTD_1K/imgs_test /home/HwHiAiUser/gp/YOLO/DATASETS/IRSTD_1K/imgs_test_jpg --quality 100

2. 
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

解决办法：
```cpp
    // //模型输出(1,5,5376) --> 64*64+32*32+16*16
    // for (size_t i = 0; i < modelOutputBoxNum; ++i)
    // {
    //     float maxValue = 0;
    //     size_t maxIndex = 0;
    //     for (size_t j = 0; j < classNum; ++j)
    //     {
    //         float value = classBuff[(offset + j) * modelOutputBoxNum + i];
    //         if (value > maxValue)
    //         {
    //             // index of class
    //             maxIndex = j;
    //             maxValue = value;
    //         }
    //     }

    //     if (maxValue > confidenceThreshold)
    //     {
    //         BoundBox box;
    //         box.x = classBuff[i] * srcWidth / modelWidth_;
    //         box.y = classBuff[yIndex * modelOutputBoxNum + i] * srcHeight / modelHeight_;
    //         box.width = classBuff[widthIndex * modelOutputBoxNum + i] * srcWidth / modelWidth_;
    //         box.height = classBuff[heightIndex * modelOutputBoxNum + i] * srcHeight / modelHeight_;
    //         box.score = maxValue;
    //         box.classIndex = maxIndex;
    //         box.index = i;
    //         if (maxIndex < classNum)
    //         {
    //             boxes.push_back(box);
    //         }
    //     }
    // }

    // 模型输出(1,5376,6)
    for (size_t i = 0; i < modelOutputBoxNum; ++i)
    {
        float x = classBuff[i * 6 + 0] * srcWidth  / modelWidth_;
        float y = classBuff[i * 6 + 1] * srcHeight / modelHeight_;
        float w = classBuff[i * 6 + 2] * srcWidth  / modelWidth_;
        float h = classBuff[i * 6 + 3] * srcHeight / modelHeight_;
        float conf = classBuff[i * 6 + 4];
        float cls  = classBuff[i * 6 + 5];  // 注意：这里直接输出类别ID（浮点数），需要转 int
    
        if (conf > confidenceThreshold)
        {
            BoundBox box;
            box.x = x;
            box.y = y;
            box.width = w;
            box.height = h;
            box.score = conf;
            box.classIndex = static_cast<size_t>(cls);
            box.index = i;
            boxes.push_back(box);
        }
    }
```