#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include <chrono>
#include <fstream>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <cmath>
#include <tuple>
#include <numeric>

#include <string>
using namespace std;

const string label[] = {"AFV","CV","LMV","MCV","SMV"};

using namespace std;
using namespace cv;



// 兼容 C++11 
#if __cplusplus < 201402L
namespace std {
    template<typename T, typename... Args>
    unique_ptr<T> make_unique(Args&&... args) {
        return unique_ptr<T>(new T(forward<Args>(args)...));
    }
}
#endif

// 若模型角度输出方式不同，请设置该宏：
// 0 = 模型直接输出角度（弧度）
// 1 = 模型输出 sin(angle) 和 cos(angle) 两个通道（建议使用 atan2 恢复）
#ifndef MODEL_ANGLE_MODE
#define MODEL_ANGLE_MODE 0
#endif

// OBB检测框结构
struct OBBBoundingBox {
    float cx, cy, width, height; // 中心点坐标和宽高
    float angle;                 // 旋转角度（弧度），规范化后
    float confidence;
    size_t classIndex;
    size_t index;
    
    // 计算四个角点
    vector<cv::Point2f> getCornerPoints() const {
        // 使用角度表示： angle 为矩形相对于 x 轴的逆时针角（弧度）
        // 注意：不同实现定义可能不同。此处假设 angle=0 表示边与 x 轴平行，angle 正值表示逆时针旋转
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        float w_half = width / 2.0f;
        float h_half = height / 2.0f;
        
        // 四个角点相对于中心的偏移（按顺时针或逆时针顺序）
        // 我们按左上、右上、右下、左下 顺序（坐标系：x 向右，y 向下）
        // 在图像坐标系中，y 向下，所以逆时针的角度定义可能需要适配（若角度方向错误可改为 -angle）
        vector<cv::Point2f> points(4);
        // 以中心为原点，先计算未旋转时四点
        // (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)
        // 然后按旋转矩阵旋转 (cos, -sin; sin, cos) 注意 y 向下时需谨慎
        float dx1 = -w_half * cos_a + h_half * sin_a;
        float dy1 = -w_half * sin_a - h_half * cos_a;
        
        float dx2 = w_half * cos_a + h_half * sin_a;
        float dy2 = w_half * sin_a - h_half * cos_a;
        
        float dx3 = w_half * cos_a - h_half * sin_a;
        float dy3 = w_half * sin_a + h_half * cos_a;
        
        float dx4 = -w_half * cos_a - h_half * sin_a;
        float dy4 = -w_half * sin_a + h_half * cos_a;
        
        points[0] = cv::Point2f(cx + dx1, cy + dy1); // 左上
        points[1] = cv::Point2f(cx + dx2, cy + dy2); // 右上
        points[2] = cv::Point2f(cx + dx3, cy + dy3); // 右下
        points[3] = cv::Point2f(cx + dx4, cy + dy4); // 左下
        
        return points;
    }
    
    OBBBoundingBox() : cx(0), cy(0), width(0), height(0), angle(0), 
                       confidence(0), classIndex(0), index(0) {}
};

// 推理配置类
class InferenceConfig {
public:
    string modelPath;
    string inputDir;
    string outputImgDir;
    string outputTxtDir;
    int32_t modelWidth;
    int32_t modelHeight;
    float confidenceThreshold;
    float nmsThreshold;
    size_t modelOutputBoxNum;
    size_t classNum;
    
    InferenceConfig() : 
        modelWidth(640), modelHeight(640),
        confidenceThreshold(0.001), nmsThreshold(0.45),
        modelOutputBoxNum(8400), classNum(5) {}
};

// 工具类
class Utils {
public:
    static bool sortByConfidence(const OBBBoundingBox& a, const OBBBoundingBox& b) {
        return a.confidence > b.confidence;
    }
    
    static void createDirectory(const string& path) {
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            mkdir(path.c_str(), 0777);
            ACLLITE_LOG_INFO("Created directory: %s", path.c_str());
        }
    }
    
    static vector<string> getImagePaths(const string& dirPath) {
        vector<string> imagePaths;
        DIR* dir = opendir(dirPath.c_str());
        if (!dir) {
            ACLLITE_LOG_ERROR("Cannot open directory: %s", dirPath.c_str());
            return imagePaths;
        }
        
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            string name = entry->d_name;
            if (name != "." && name != ".." && name != ".keep") {
                string fullPath = dirPath + "/" + name;
                size_t dot = name.find_last_of(".");
                if (dot == string::npos) continue;
                string ext = name.substr(dot + 1);
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp") {
                    imagePaths.push_back(fullPath);
                }
            }
        }
        closedir(dir);
        return imagePaths;
    }
    
    static string getFileNameWithoutExt(const string& path) {
        size_t lastSlash = path.find_last_of("/");
        size_t lastDot = path.find_last_of(".");
        return path.substr(lastSlash + 1, lastDot - lastSlash - 1);
    }
    
    // 角度规范化到 [-pi/2, pi/2)
    static float normalizeAngle(float angle) {
        // 将 angle 规范到 [-pi/2, pi/2)
        // 先到 [-pi, pi)
        angle = std::fmod(angle, (float)M_PI);
        if (angle < -M_PI) angle += 2.0f * M_PI;
        if (angle >= M_PI) angle -= 2.0f * M_PI;
        // 现在到 (-pi, pi]
        // 若 angle >= pi/2 或 angle < -pi/2，则可以通过交换宽高并减/加 pi/2 来标准化
        if (angle >= M_PI/2.0f) angle -= M_PI;
        if (angle < -M_PI/2.0f) angle += M_PI;
        // 最终在 [-pi/2, pi/2)
        return angle;
    }
};

// OBB后处理器
class OBBPostProcessor {
private:
    size_t modelOutputBoxNum_;
    size_t classNum_;
    
public:
    OBBPostProcessor(size_t boxNum, size_t classNum) 
        : modelOutputBoxNum_(boxNum), classNum_(classNum) {}
    
    vector<OBBBoundingBox> parseOutput(float* outputData, size_t outputSize,
                                      int srcWidth, int srcHeight,
                                      int modelWidth, int modelHeight,
                                      float confidenceThreshold) {
        vector<OBBBoundingBox> boxes;
        
        // 输出格式 (示例): 通道顺序假定为 [cx_chan, cy_chan, w_chan, h_chan, cls1..clsN, angle_chan?]
        // 你需根据模型输出实际通道顺序调整索引
        for (size_t i = 0; i < modelOutputBoxNum_; ++i) {
            float maxValue = 0.0f;
            size_t maxIndex = 0;
            
            // 找最大类别置信度（假设类别通道从第5个开始）
            for (size_t j = 0; j < classNum_; ++j) {
                float value = outputData[(4 + j) * modelOutputBoxNum_ + i];
                if (value > maxValue) {
                    maxValue = value;
                    maxIndex = j;
                }
            }
            
            if (maxValue > confidenceThreshold) {
                OBBBoundingBox box;
                
                // 获取回归参数（根据你的模型需要做缩放）
                float cx = outputData[0 * modelOutputBoxNum_ + i] * srcWidth / modelWidth;
                float cy = outputData[1 * modelOutputBoxNum_ + i] * srcHeight / modelHeight;
                float w = outputData[2 * modelOutputBoxNum_ + i] * srcWidth / modelWidth;
                float h = outputData[3 * modelOutputBoxNum_ + i] * srcHeight / modelHeight;
                
                float raw_angle = 0.0f;
#if MODEL_ANGLE_MODE == 0
                // 直接弧度输出
                raw_angle = outputData[ (4 + classNum_) * modelOutputBoxNum_ + i ];
#elif MODEL_ANGLE_MODE == 1
                // 角度以 sin, cos 两通道输出：假设 sin 在通道 a，cos 在通道 a+1
                // 这里假设角度通道起始索引为 4 + classNum_，sin在该通道，cos在下一个通道
                {
                    float sin_v = outputData[(4 + classNum_) * modelOutputBoxNum_ + i];
                    float cos_v = outputData[(4 + classNum_ + 1) * modelOutputBoxNum_ + i];
                    raw_angle = atan2f(sin_v, cos_v); // -pi .. pi
                }
#else
#error "Unsupported MODEL_ANGLE_MODE"
#endif
                
                // 规范化角度到 [-pi/2, pi/2)
                float angle = Utils::normalizeAngle(raw_angle);
                
                // 有时模型会把长边和短边一起回归，angle 指明长边方向。如果角度导致宽高颠倒（w<h 但角度表示为90度等），
                // 需要在解析时根据模型定义调整（此处只规范化角度，不交换宽高）
                
                box.cx = cx;
                box.cy = cy;
                box.width = w;
                box.height = h;
                box.angle = angle;
                box.confidence = maxValue;
                box.classIndex = maxIndex;
                box.index = i;
                
                boxes.push_back(box);
            }
        }
        return boxes;
    }
};

// OBB NMS处理器 - 使用 ProbIoU 风格距离
class OBBNMSProcessor {
private:
    // 获取协方差矩阵的参数 (a, b, c)
    static std::tuple<float, float, float> getCovarianceMatrix(const OBBBoundingBox& box) {
        // 将矩形视为均匀分布，协方差近似为 width^2/12, height^2/12 在旋转后的坐标系中
        float w = box.width;
        float h = box.height;
        float r = box.angle;  // 旋转角度（弧度）
        
        float cos_r = std::cos(r);
        float sin_r = std::sin(r);
        
        float w_sq = (w * w) / 12.0f;
        float h_sq = (h * h) / 12.0f;
        
        // 旋转到世界坐标系的协方差矩阵元素
        float a = w_sq * cos_r * cos_r + h_sq * sin_r * sin_r; // var_xx
        float b = w_sq * sin_r * sin_r + h_sq * cos_r * cos_r; // var_yy
        float c = (w_sq - h_sq) * cos_r * sin_r;               // cov_xy
        
        return std::make_tuple(a, b, c);
    }

public:
    // ProbIoU计算（基于 Bhattacharyya / Hellinger 距离）
    static float calculateProbIOU(const OBBBoundingBox& box1, const OBBBoundingBox& box2, 
                                  bool useCIoU = false, float eps = 1e-7f) {
        // 中心点
        float x1 = box1.cx;
        float y1 = box1.cy;
        float x2 = box2.cx;
        float y2 = box2.cy;
        
        // 协方差
        std::tuple<float, float, float> cov1 = getCovarianceMatrix(box1);
        std::tuple<float, float, float> cov2 = getCovarianceMatrix(box2);
        
        float a1 = std::get<0>(cov1);
        float b1 = std::get<1>(cov1);
        float c1 = std::get<2>(cov1);
        
        float a2 = std::get<0>(cov2);
        float b2 = std::get<1>(cov2);
        float c2 = std::get<2>(cov2);
        
        // 坐标差：注意符号，常见 ProbIoU 推导中使用 dx = x2 - x1, dy = y2 - y1
        float dx = x2 - x1;
        float dy = y2 - y1;
        
        // 分母稳定化
        float denom_a = (a1 + a2);
        float denom_b = (b1 + b2);
        float denom_c = (c1 + c2);
        float denominator = denom_a * denom_b - denom_c * denom_c + eps;
        
        // t1, t2, t3 参考推导
        float t1 = ((denom_a * dy * dy + denom_b * dx * dx) / denominator) * 0.25f;
        float t2 = ((denom_c * dx * dy) / denominator) * 0.5f;
        
        float det1 = std::max(a1 * b1 - c1 * c1, 0.0f);
        float det2 = std::max(a2 * b2 - c2 * c2, 0.0f);
        float sqrt_dets = std::sqrt(det1 * det2) + eps;
        float t3_inner = denominator / (4.0f * sqrt_dets) + eps;
        float t3 = 0.5f * std::log(t3_inner);
        
        float bd = t1 + t2 + t3;
        bd = std::max(eps, std::min(bd, 100.0f));
        
        float hd = std::sqrt(1.0f - std::exp(-bd) + eps);
        float iou = 1.0f - hd;
        if (iou < 0.0f) iou = 0.0f;
        if (iou > 1.0f) iou = 1.0f;
        
        if (useCIoU) {
            float w1 = box1.width;
            float h1 = box1.height;
            float w2 = box2.width;
            float h2 = box2.height;
            float aspect1 = std::atan2(w1, h1);
            float aspect2 = std::atan2(w2, h2);
            float v = (4.0f / (M_PI * M_PI)) * (aspect2 - aspect1) * (aspect2 - aspect1);
            float alpha = v / (v + 1.0f - iou + eps);
            return iou - alpha * v;
        }
        
        return iou;
    }

    static float calculateOBBIOU(const OBBBoundingBox& box1, const OBBBoundingBox& box2) {
        return calculateProbIOU(box1, box2, false);
    }

    static vector<OBBBoundingBox> applyNMS(vector<OBBBoundingBox>& boxes, 
                                          float nmsThreshold) {
        vector<OBBBoundingBox> result;
        sort(boxes.begin(), boxes.end(), Utils::sortByConfidence);
        
        while (!boxes.empty()) {
            OBBBoundingBox best = boxes[0];
            result.push_back(best);
            
            // 对剩余框依据 ProbIoU 与分类进行筛除
            vector<OBBBoundingBox> remaining;
            for (size_t i = 1; i < boxes.size(); ++i) {
                if (boxes[i].classIndex != best.classIndex) {
                    remaining.push_back(boxes[i]);
                    continue;
                }
                float iou = calculateOBBIOU(best, boxes[i]);
                if (iou <= nmsThreshold) {
                    remaining.push_back(boxes[i]);
                } else {
                    // 被抑制 -- 可以记录日志用于调试
                    // ACLLITE_LOG_INFO("Suppressed box idx %zu by idx %zu with iou %f", boxes[i].index, best.index, iou);
                }
            }
            boxes.swap(remaining);
        }
        return result;
    }
};


// 结果保存器
class OBBResultSaver {
public:
    static void saveResults(const vector<OBBBoundingBox>& boxes, 
                           const string& imagePath,
                           const string& outputImgDir,
                           const string& outputTxtDir,
                           int srcWidth, int srcHeight) {
        string fileName = Utils::getFileNameWithoutExt(imagePath);
        string outputImagePath = outputImgDir + "/" + fileName + ".jpg";
        string outputTxtPath = outputTxtDir + "/" + fileName + ".txt";
        
        // 保存TXT文件 (DOTA格式)
        saveTxtFile(boxes, outputTxtPath, srcWidth, srcHeight);
        
        // 保存可视化图像
        saveVisualization(boxes, imagePath, outputImagePath);
    }
    
private:
    static void saveTxtFile(const vector<OBBBoundingBox>& boxes, 
                           const string& txtPath, int srcWidth, int srcHeight) {
        ofstream txtFile(txtPath);
        if (!txtFile.is_open()) {
            ACLLITE_LOG_ERROR("Cannot open output TXT file: %s", txtPath.c_str());
            return;
        }
        
        for (const auto& box : boxes) {
            // 获取四个角点
            vector<cv::Point2f> points = box.getCornerPoints();
            
            // DOTA格式：x1 y1 x2 y2 x3 y3 x4 y4 category confidence
            txtFile << points[0].x << " " << points[0].y << " "
                    << points[1].x << " " << points[1].y << " "
                    << points[2].x << " " << points[2].y << " "
                    << points[3].x << " " << points[3].y << " "
                    << label[box.classIndex] << " "
                    << box.confidence << endl;
        }
        txtFile.close();
    }
    
    static void saveVisualization(const vector<OBBBoundingBox>& boxes,
                                 const string& imagePath,
                                 const string& outputPath) {
        cv::Mat srcImage = cv::imread(imagePath);
        if (srcImage.empty()) {
            ACLLITE_LOG_ERROR("Cannot read image: %s", imagePath.c_str());
            return;
        }
        
        const vector<cv::Scalar> colors = {
            cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
            cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
            cv::Scalar(128, 0, 0), cv::Scalar(0, 128, 0), cv::Scalar(0, 0, 128),
            cv::Scalar(128, 128, 0), cv::Scalar(128, 0, 128), cv::Scalar(0, 128, 128),
            cv::Scalar(64, 64, 64), cv::Scalar(192, 192, 192), cv::Scalar(255, 128, 0)
        };
        
        for (size_t i = 0; i < boxes.size(); ++i) {
            const auto& box = boxes[i];
            vector<cv::Point2f> points = box.getCornerPoints();
            
            cv::Scalar color = colors[box.classIndex % colors.size()];
            vector<cv::Point> intPoints;
            for (const auto& p : points) {
                intPoints.push_back(cv::Point(static_cast<int>(round(p.x)), static_cast<int>(round(p.y))));
            }
            
            // 绘制旋转多边形
            const cv::Point* pts = intPoints.data();
            int npts = static_cast<int>(intPoints.size());
            polylines(srcImage, &pts, &npts, 1, true, color, 2);
            
            circle(srcImage, Point(static_cast<int>(round(box.cx)), static_cast<int>(round(box.cy))), 3, color, -1);
            
            string className = (box.classIndex < 5) ? label[box.classIndex] : "unknown";
            // 格式化置信度到小数点后3位
            char confBuf[32];
            snprintf(confBuf, sizeof(confBuf), "%.3f", box.confidence);
            string markString = className + ":" + confBuf;
            
            putText(srcImage, markString, 
                    Point(static_cast<int>(round(box.cx - box.width/4)), 
                          static_cast<int>(round(box.cy - box.height/4 - 10))),
                    FONT_HERSHEY_COMPLEX, 0.5, color, 1);
        }
        
        imwrite(outputPath, srcImage);
    }
};

// 主推理类
class YOLOOBBInference {
private:
    InferenceConfig config_;
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    unique_ptr<OBBPostProcessor> postProcessor_;
    
public:
    YOLOOBBInference(const InferenceConfig& config) : config_(config) {
        postProcessor_ = std::make_unique<OBBPostProcessor>(config_.modelOutputBoxNum, config_.classNum);
    }
    
    ~YOLOOBBInference() {
        releaseResources();
    }
    
    bool initialize() {
        if (aclResource_.Init() != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("ACL resource initialization failed");
            return false;
        }
        
        if (aclrtGetRunMode(&runMode_) != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("Get run mode failed");
            return false;
        }
        
        if (imageProcess_.Init() != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("Image process initialization failed");
            return false;
        }
        
        if (model_.Init(config_.modelPath.c_str()) != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("Model initialization failed");
            return false;
        }
        
        ACLLITE_LOG_INFO("Model initialized successfully");
        ACLLITE_LOG_INFO("Input size: %dx%d", config_.modelWidth, config_.modelHeight);
        ACLLITE_LOG_INFO("Output boxes: %zu", config_.modelOutputBoxNum);
        ACLLITE_LOG_INFO("Classes: %zu", config_.classNum);
        ACLLITE_LOG_INFO("Using ProbIoU for NMS processing");
        
        return true;
    }
    
    void runInference() {
        // 创建输出目录
        Utils::createDirectory(config_.outputImgDir);
        Utils::createDirectory(config_.outputTxtDir);
        
        // 获取图像路径
        vector<string> imagePaths = Utils::getImagePaths(config_.inputDir);
        if (imagePaths.empty()) {
            ACLLITE_LOG_ERROR("No images found in directory: %s", config_.inputDir.c_str());
            return;
   }
        
        ACLLITE_LOG_INFO("Found %zu images to process", imagePaths.size());
        
        double totalTime = 0.0;
        size_t processedImages = 0;
        
        for (size_t i = 0; i < imagePaths.size(); ++i) {
            auto start = chrono::steady_clock::now();
            
            if (processImage(imagePaths[i])) {
                auto end = chrono::steady_clock::now();
                double elapsed = chrono::duration<double>(end - start).count();
                
                if (i == 0) {
                    ACLLITE_LOG_INFO("Warmup image processed in %f s, fps: %f", elapsed, 1.0/elapsed);
                } else {
                    totalTime += elapsed;
                    processedImages++;
                    ACLLITE_LOG_INFO("Image %zu processed in %f s, fps: %f", i, elapsed, 1.0/elapsed);
                }
            } else {
                ACLLITE_LOG_ERROR("Failed to process image: %s", imagePaths[i].c_str());
            }
        }
        
        if (processedImages > 0) {
            double avgFps = processedImages / totalTime;
            ACLLITE_LOG_INFO("Processed %zu images, average FPS: %f", processedImages, avgFps);
        }
    }
    
private:
    bool processImage(const string& imagePath) {
        // 图像预处理
        if (!preprocessImage(imagePath)) {
            return false;
        }
        
        // 推理
        vector<InferenceOutput> inferOutputs;
        if (!runModelInference(inferOutputs)) {
            return false;
        }
        
        // 后处理
        return postprocessResults(inferOutputs, imagePath);
    }
    
    bool preprocessImage(const string& imagePath) {
        ImageData image;
        if (ReadJpeg(image, imagePath) != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("Failed to read image: %s", imagePath.c_str());
            return false;
        }
        
        ImageData imageDevice;
        if (CopyImageToDevice(imageDevice, image, runMode_, MEMORY_DVPP) != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("Failed to copy image to device");
            return false;
        }
        
        ImageData yuvImage;
        if (imageProcess_.JpegD(yuvImage, imageDevice) != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("Failed to decode JPEG");
            return false;
        }
        
        if (imageProcess_.Resize(resizedImage_, yuvImage, config_.modelWidth, config_.modelHeight) != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("Failed to resize image");
            return false;
        }
        
        return true;
    }
    
    bool runModelInference(vector<InferenceOutput>& inferOutputs) {
        if (model_.CreateInput(static_cast<void*>(resizedImage_.data.get()), resizedImage_.size) != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("Failed to create model input");
            return false;
        }
        
        if (model_.Execute(inferOutputs) != ACL_SUCCESS) {
            ACLLITE_LOG_ERROR("Failed to execute model");
            return false;
        }
        
        return true;
    }
    
    bool postprocessResults(vector<InferenceOutput>& inferOutputs, const string& imagePath) {
        if (inferOutputs.empty()) {
            ACLLITE_LOG_ERROR("No inference output");
            return false;
        }
        
        cv::Mat srcImage = cv::imread(imagePath);
        if (srcImage.empty()) {
            ACLLITE_LOG_ERROR("Cannot read source image: %s", imagePath.c_str());
            return false;
        }
        
        float* outputData = static_cast<float*>(inferOutputs[0].data.get());
        
        // 解析模型输出
        vector<OBBBoundingBox> boxes = postProcessor_->parseOutput(
            outputData, inferOutputs[0].size,
            srcImage.cols, srcImage.rows,
            config_.modelWidth, config_.modelHeight,
            config_.confidenceThreshold
        );
        
        ACLLITE_LOG_INFO("Filtered %zu OBB boxes by confidence threshold", boxes.size());
        
        // 应用NMS (使用ProbIoU)
        vector<OBBBoundingBox> finalBoxes = OBBNMSProcessor::applyNMS(boxes, config_.nmsThreshold);
        
        ACLLITE_LOG_INFO("Final result: %zu OBB boxes after ProbIoU NMS", finalBoxes.size());
        
        // 保存结果
        OBBResultSaver::saveResults(finalBoxes, imagePath, config_.outputImgDir, config_.outputTxtDir,
                                   srcImage.cols, srcImage.rows);
        
        // 可选：打印前几个 box 的角度用于调试
        for (size_t i = 0; i < std::min<size_t>(finalBoxes.size(), 5); ++i) {
            ACLLITE_LOG_INFO("Box %zu: cx=%f cy=%f w=%f h=%f angle(rad)=%f angle(deg)=%f conf=%f",
                             i, finalBoxes[i].cx, finalBoxes[i].cy, finalBoxes[i].width, finalBoxes[i].height,
                             finalBoxes[i].angle, finalBoxes[i].angle * 180.0f / M_PI, finalBoxes[i].confidence);
        }
        
        return true;
    }
    
    void releaseResources() {
        model_.DestroyResource();
        imageProcess_.DestroyResource();
        aclResource_.Release();
    }
};

// 使用示例
int main() {
    InferenceConfig config;
    
    // 根据模型配置
    config.modelPath = "../model/YOLO11s_base_obb_MVRSD_640.om";  // OBB模型路径
    config.inputDir = "/home/HwHiAiUser/gp/DATASETS/MVRSD/test";
    config.outputImgDir = "../output_obb/images";
    config.outputTxtDir = "../output_obb/labels";
    
    // 模型参数 - 根据导出信息配置
    config.modelWidth = 640;
    config.modelHeight = 640;
    config.modelOutputBoxNum = 8400;  // 从输出shape (1, 20, 21504) 得出
    config.classNum = 5;              // 类别数
    
    // 检测参数
    config.confidenceThreshold = 0.25;
    config.nmsThreshold = 0.45;
    
    YOLOOBBInference inference(config);
    
    if (inference.initialize()) {
        ACLLITE_LOG_INFO("Starting OBB inference with ProbIoU...");
        inference.runInference();
        ACLLITE_LOG_INFO("OBB inference completed");
    } else {
        ACLLITE_LOG_ERROR("Failed to initialize OBB inference engine");
        return -1;
    }
    
    return 0;
}
