#ifndef YOLO_HPP
#define YOLO_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "../tensorRT/common/trt_tensor.hpp"
#include <vector>

namespace ObjectDetector {

    struct Box {
        float left, top, right, bottom, confidence;
        int class_label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int class_label)
            :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
    };

    typedef std::vector<Box> BoxArray;
};


/**
 * @brief 发挥极致的性能体验
 * 支持YoloX和YoloV5
 */
namespace Yolo{

    using namespace std;
    using namespace ObjectDetector;

    enum class Type : int{
        V5 = 0,
        X  = 1,
        V3 = 2,
        V7 = 3
    };

    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch);

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, Type type, int gpuid,
        float confidence_threshold=0.25f, float nms_threshold=0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );
    const char* type_name(Type type);

}; // namespace Yolo

#endif // YOLO_HPP