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

#include <string>
using namespace std;

const string label[] = {"so"};

using namespace std;
using namespace cv;

// 兼容 C++11
#if __cplusplus < 201402L
namespace std
{
    template <typename T, typename... Args>
    unique_ptr<T> make_unique(Args &&...args)
    {
        return unique_ptr<T>(new T(forward<Args>(args)...));
    }
}
#endif

// 模型输出格式枚举
enum class ModelOutputFormat
{
    FORMAT_1_5_N, // (1, 5, N)
    FORMAT_1_N_6  // (1, N, 6)
};

// 检测框结构
struct BoundingBox
{
    float x, y, width, height;
    float confidence;
    size_t classIndex;
    size_t index;

    BoundingBox() : x(0), y(0), width(0), height(0), confidence(0), classIndex(0), index(0) {}
};

// 推理配置类
class InferenceConfig
{
public:
    string modelPath;
    string inputDir;
    string outputImgDir;
    string outputTxtDir;
    int32_t modelWidth;
    int32_t modelHeight;
    float confidenceThreshold;
    float nmsThreshold;
    ModelOutputFormat outputFormat;
    size_t modelOutputBoxNum;
    size_t classNum;

    InferenceConfig() : modelWidth(512), modelHeight(512),
                        confidenceThreshold(0.001), nmsThreshold(0.45),
                        outputFormat(ModelOutputFormat::FORMAT_1_N_6),
                        modelOutputBoxNum(5376), classNum(1) {}
};

// 工具类
class Utils
{
public:
    static bool sortByConfidence(const BoundingBox &a, const BoundingBox &b)
    {
        return a.confidence > b.confidence;
    }

    static void createDirectory(const string &path)
    {
        struct stat info;
        if (stat(path.c_str(), &info) != 0)
        {
            mkdir(path.c_str(), 0777);
            ACLLITE_LOG_INFO("Created directory: %s", path.c_str());
        }
    }

    static vector<string> getImagePaths(const string &dirPath)
    {
        vector<string> imagePaths;
        DIR *dir = opendir(dirPath.c_str());
        if (!dir)
        {
            ACLLITE_LOG_ERROR("Cannot open directory: %s", dirPath.c_str());
            return imagePaths;
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            string name = entry->d_name;
            if (name != "." && name != ".." && name != ".keep")
            {
                imagePaths.push_back(dirPath + "/" + name);
            }
        }
        closedir(dir);
        return imagePaths;
    }

    static string getFileNameWithoutExt(const string &path)
    {
        size_t lastSlash = path.find_last_of("/");
        size_t lastDot = path.find_last_of(".");
        return path.substr(lastSlash + 1, lastDot - lastSlash - 1);
    }
};

// 后处理器基类
class PostProcessor
{
public:
    virtual ~PostProcessor() {}
    virtual vector<BoundingBox> parseOutput(float *outputData, size_t outputSize,
                                            int srcWidth, int srcHeight,
                                            int modelWidth, int modelHeight,
                                            float confidenceThreshold) = 0;
};

// 格式1处理器：(1, 5, N)
class Format1Processor : public PostProcessor
{
private:
    size_t modelOutputBoxNum_;
    size_t classNum_;

public:
    Format1Processor(size_t boxNum, size_t classNum)
        : modelOutputBoxNum_(boxNum), classNum_(classNum) {}

    vector<BoundingBox> parseOutput(float *outputData, size_t outputSize,
                                    int srcWidth, int srcHeight,
                                    int modelWidth, int modelHeight,
                                    float confidenceThreshold) override
    {
        vector<BoundingBox> boxes;

        for (size_t i = 0; i < modelOutputBoxNum_; ++i)
        {
            float maxValue = 0;
            size_t maxIndex = 0;

            // 找到最大置信度的类别
            for (size_t j = 0; j < classNum_; ++j)
            {
                float value = outputData[(4 + j) * modelOutputBoxNum_ + i];
                if (value > maxValue)
                {
                    maxIndex = j;
                    maxValue = value;
                }
            }

            if (maxValue > confidenceThreshold)
            {
                BoundingBox box;
                box.x = outputData[i] * srcWidth / modelWidth;
                box.y = outputData[modelOutputBoxNum_ + i] * srcHeight / modelHeight;
                box.width = outputData[2 * modelOutputBoxNum_ + i] * srcWidth / modelWidth;
                box.height = outputData[3 * modelOutputBoxNum_ + i] * srcHeight / modelHeight;
                box.confidence = maxValue;
                box.classIndex = maxIndex;
                box.index = i;
                boxes.push_back(box);
            }
        }
        return boxes;
    }
};

// 格式2处理器：(1, N, 6)
class Format2Processor : public PostProcessor
{
private:
    size_t modelOutputBoxNum_;

public:
    Format2Processor(size_t boxNum) : modelOutputBoxNum_(boxNum) {}

    vector<BoundingBox> parseOutput(float *outputData, size_t outputSize,
                                    int srcWidth, int srcHeight,
                                    int modelWidth, int modelHeight,
                                    float confidenceThreshold) override
    {
        vector<BoundingBox> boxes;

        for (size_t i = 0; i < modelOutputBoxNum_; ++i)
        {
            float x = outputData[i * 6 + 0] * srcWidth / modelWidth;
            float y = outputData[i * 6 + 1] * srcHeight / modelHeight;
            float w = outputData[i * 6 + 2] * srcWidth / modelWidth;
            float h = outputData[i * 6 + 3] * srcHeight / modelHeight;
            float conf = outputData[i * 6 + 4];
            float cls = outputData[i * 6 + 5];

            if (conf > confidenceThreshold)
            {
                BoundingBox box;
                box.x = x;
                box.y = y;
                box.width = w;
                box.height = h;
                box.confidence = conf;
                box.classIndex = static_cast<size_t>(cls);
                box.index = i;
                boxes.push_back(box);
            }
        }
        return boxes;
    }
};

// NMS处理器
class NMSProcessor
{
public:
    static vector<BoundingBox> applyNMS(vector<BoundingBox> &boxes,
                                        float nmsThreshold, int maxLength)
    {
        vector<BoundingBox> result;
        sort(boxes.begin(), boxes.end(), Utils::sortByConfidence);

        while (!boxes.empty())
        {
            result.push_back(boxes[0]);

            for (size_t i = 1; i < boxes.size();)
            {
                float iou = calculateIOU(boxes[0], boxes[i], maxLength);
                if (iou > nmsThreshold)
                {
                    boxes.erase(boxes.begin() + i);
                }
                else
                {
                    ++i;
                }
            }
            boxes.erase(boxes.begin());
        }
        return result;
    }

private:
    static float calculateIOU(const BoundingBox &box1, const BoundingBox &box2, int maxLength)
    {
        // 平移坐标避免不同类别框重叠
        float x1 = box1.x + maxLength * box1.classIndex;
        float y1 = box1.y + maxLength * box1.classIndex;
        float x2 = box2.x + maxLength * box2.classIndex;
        float y2 = box2.y + maxLength * box2.classIndex;

        float xLeft = max(x1, x2);
        float yTop = max(y1, y2);
        float xRight = min(x1 + box1.width, x2 + box2.width);
        float yBottom = min(y1 + box1.height, y2 + box2.height);

        float width = max(0.0f, xRight - xLeft);
        float height = max(0.0f, yBottom - yTop);
        float intersection = width * height;
        float union_area = box1.width * box1.height + box2.width * box2.height - intersection;

        return intersection / union_area;
    }
};

// 结果保存器
class ResultSaver
{
public:
    static void saveResults(const vector<BoundingBox> &boxes,
                            const string &imagePath,
                            const string &outputImgDir,
                            const string &outputTxtDir,
                            int srcWidth, int srcHeight)
    {
        string fileName = Utils::getFileNameWithoutExt(imagePath);
        string outputImagePath = outputImgDir + "/" + fileName + ".jpg";
        string outputTxtPath = outputTxtDir + "/" + fileName + ".txt";

        // 保存TXT文件
        saveTxtFile(boxes, outputTxtPath, srcWidth, srcHeight);

        // 保存可视化图像
        saveVisualization(boxes, imagePath, outputImagePath);
    }

private:
    static void saveTxtFile(const vector<BoundingBox> &boxes,
                            const string &txtPath, int srcWidth, int srcHeight)
    {
        ofstream txtFile(txtPath);
        if (!txtFile.is_open())
        {
            ACLLITE_LOG_ERROR("Cannot open output TXT file: %s", txtPath.c_str());
            return;
        }

        for (const auto &box : boxes)
        {
            float x_center_norm = box.x / srcWidth;
            float y_center_norm = box.y / srcHeight;
            float width_norm = box.width / srcWidth;
            float height_norm = box.height / srcHeight;

            txtFile << box.classIndex << " "
                    << x_center_norm << " "
                    << y_center_norm << " "
                    << width_norm << " "
                    << height_norm << " "
                    << box.confidence << endl;
        }
        txtFile.close();
    }

    static void saveVisualization(const vector<BoundingBox> &boxes,
                                  const string &imagePath,
                                  const string &outputPath)
    {
        cv::Mat srcImage = cv::imread(imagePath);
        const vector<cv::Scalar> colors = {
            cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            const auto &box = boxes[i];
            cv::Point leftUp(box.x - box.width / 2, box.y - box.height / 2);
            cv::Point rightBottom(box.x + box.width / 2, box.y + box.height / 2);

            cv::rectangle(srcImage, leftUp, rightBottom, colors[i % colors.size()], 2);

            string className = label[box.classIndex];
            string markString = to_string(box.confidence) + ":" + className;
            cv::putText(srcImage, markString, cv::Point(leftUp.x, leftUp.y + 11),
                        cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255));
        }

        cv::imwrite(outputPath, srcImage);
    }
};

// 主推理类
class YOLOInference
{
private:
    InferenceConfig config_;
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    unique_ptr<PostProcessor> postProcessor_;

public:
    YOLOInference(const InferenceConfig &config) : config_(config)
    {
        setupPostProcessor();
    }

    ~YOLOInference()
    {
        releaseResources();
    }

    bool initialize()
    {
        if (aclResource_.Init() != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("ACL resource initialization failed");
            return false;
        }

        if (aclrtGetRunMode(&runMode_) != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("Get run mode failed");
            return false;
        }

        if (imageProcess_.Init() != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("Image process initialization failed");
            return false;
        }

        if (model_.Init(config_.modelPath.c_str()) != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("Model initialization failed");
            return false;
        }

        return true;
    }

    void runInference()
    {
        // 创建输出目录
        Utils::createDirectory(config_.outputImgDir);
        Utils::createDirectory(config_.outputTxtDir);

        // 获取图像路径
        vector<string> imagePaths = Utils::getImagePaths(config_.inputDir);
        if (imagePaths.empty())
        {
            ACLLITE_LOG_ERROR("No images found in directory: %s", config_.inputDir.c_str());
            return;
        }

        double totalTime = 0.0;
        size_t processedImages = 0;

        for (size_t i = 0; i < imagePaths.size(); ++i)
        {
            auto start = chrono::steady_clock::now();

            if (processImage(imagePaths[i]))
            {
                auto end = chrono::steady_clock::now();
                double elapsed = chrono::duration<double>(end - start).count();

                if (i == 0)
                {
                    ACLLITE_LOG_INFO("Warmup image processed in %f s, fps: %f", elapsed, 1.0 / elapsed);
                }
                else
                {
                    totalTime += elapsed;
                    processedImages++;
                    ACLLITE_LOG_INFO("Image %zu processed in %f s, fps: %f", i, elapsed, 1.0 / elapsed);
                }
            }
        }

        if (processedImages > 0)
        {
            double avgFps = processedImages / totalTime;
            ACLLITE_LOG_INFO("Processed %zu images, average FPS: %f", processedImages, avgFps);
        }
    }

private:
    void setupPostProcessor()
    {
        switch (config_.outputFormat)
        {
        case ModelOutputFormat::FORMAT_1_5_N:
            postProcessor_ = std::make_unique<Format1Processor>(config_.modelOutputBoxNum, config_.classNum);
            break;
        case ModelOutputFormat::FORMAT_1_N_6:
            postProcessor_ = std::make_unique<Format2Processor>(config_.modelOutputBoxNum);
            break;
        }
    }

    bool processImage(const string &imagePath)
    {
        // 图像预处理
        if (!preprocessImage(imagePath))
        {
            return false;
        }

        // 推理
        vector<InferenceOutput> inferOutputs;
        if (!runModelInference(inferOutputs))
        {
            return false;
        }

        // 后处理
        return postprocessResults(inferOutputs, imagePath);
    }

    bool preprocessImage(const string &imagePath)
    {
        ImageData image;
        if (ReadJpeg(image, imagePath) != ACL_SUCCESS)
        {
            ACLLITE_LOG_ERROR("Failed to read image: %s", imagePath.c_str());
            return false;
        }

        ImageData imageDevice;
        if (CopyImageToDevice(imageDevice, image, runMode_, MEMORY_DVPP) != ACL_SUCCESS)
        {
            return false;
        }

        ImageData yuvImage;
        if (imageProcess_.JpegD(yuvImage, imageDevice) != ACL_SUCCESS)
        {
            return false;
        }

        if (imageProcess_.Resize(resizedImage_, yuvImage, config_.modelWidth, config_.modelHeight) != ACL_SUCCESS)
        {
            return false;
        }

        return true;
    }

    bool runModelInference(vector<InferenceOutput> &inferOutputs)
    {
        if (model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size) != ACL_SUCCESS)
        {
            return false;
        }

        return model_.Execute(inferOutputs) == ACL_SUCCESS;
    }

    bool postprocessResults(vector<InferenceOutput> &inferOutputs, const string &imagePath)
    {
        cv::Mat srcImage = cv::imread(imagePath);
        float *outputData = static_cast<float *>(inferOutputs[0].data.get());

        // 解析模型输出
        vector<BoundingBox> boxes = postProcessor_->parseOutput(
            outputData, inferOutputs[0].size,
            srcImage.cols, srcImage.rows,
            config_.modelWidth, config_.modelHeight,
            config_.confidenceThreshold);

        ACLLITE_LOG_INFO("Filtered %zu boxes by confidence threshold", boxes.size());

        // 应用NMS
        int maxLength = max(config_.modelWidth, config_.modelHeight);
        vector<BoundingBox> finalBoxes = NMSProcessor::applyNMS(boxes, config_.nmsThreshold, maxLength);

        ACLLITE_LOG_INFO("Final result: %zu boxes after NMS", finalBoxes.size());

        // 保存结果
        ResultSaver::saveResults(finalBoxes, imagePath, config_.outputImgDir, config_.outputTxtDir,
                                 srcImage.cols, srcImage.rows);

        return true;
    }

    void releaseResources()
    {
        model_.DestroyResource();
        imageProcess_.DestroyResource();
        aclResource_.Release();
    }
};

// 使用示例
int main()
{
    InferenceConfig config;

    config.inputDir = "/home/HwHiAiUser/gp/DATASETS/IRSTD_1K/test";
    config.outputImgDir = "../output_hbb/images";
    config.outputTxtDir = "../output_hbb/labels";
    config.modelWidth = 512;
    config.modelHeight = 512;
    config.confidenceThreshold = 0.25;
    config.nmsThreshold = 0.45;
    config.classNum = 1;

    // config.modelPath = "../model/best_nextvit200.om";
    // config.outputFormat = ModelOutputFormat::FORMAT_1_N_6; // 根据模型选择
    // config.modelOutputBoxNum = 5376;

    // config.modelPath = "../model/best11sp2.om";
    // config.outputFormat = ModelOutputFormat::FORMAT_1_5_N; // 根据模型选择
    // config.modelOutputBoxNum = 21760;

    config.modelPath = "../model/YOLO11n_p2_hbb_IRSTD_1K_512.om";
    config.outputFormat = ModelOutputFormat::FORMAT_1_5_N; // 根据模型选择
    config.modelOutputBoxNum = 34000;
    config.modelWidth = 640;
    config.modelHeight = 640;

    YOLOInference inference(config);

    if (inference.initialize())
    {
        inference.runInference();
    }
    else
    {
        ACLLITE_LOG_ERROR("Failed to initialize inference engine");
        return -1;
    }

    return 0;
}
