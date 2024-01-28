/**
 * @file rknn_yolov8_seg.cpp
 * @author zzx
 * @brief v8分割推理核心代码
 * @version 0.1
 * @date 2024-01-27
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "rknn_yolov8_seg.h"

namespace FasterRKNN {

rknnYolov8Seg::rknnYolov8Seg(const std::string &model_path, const float nms_threshold,
                             const float conf_threshold) {
    Init(model_path);
    postprocess_ =
        std::make_shared<yolov8SegPostProcess>(&model_info_, conf_threshold, nms_threshold);
}

rknnYolov8Seg::~rknnYolov8Seg() {
    printf("rknnYolov8Seg release!\n");
}

SegRes rknnYolov8Seg::infer(const cv::Mat &img) {
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    auto img_out = static_resize(img, model_info_.input_w, model_info_.input_h);
    float img_scale =
        std::min(model_info_.input_w / (img.cols * 1.0), model_info_.input_w / (img.rows * 1.0));
    memcpy(input_mems_[0]->virt_addr, img_out.data, model_info_.input_attrs[0].size_with_stride);

    CHECK_RKNN(rknn_run(ctx_, NULL));

    postprocess_->process(output_mems_, results_, img_scale);

    return results_;
}

std::shared_ptr<rknnInferBaseSegRes> create_infer_yolov8seg(const std::string &model_path,
                                                            const float nms_threshold,
                                                            const float conf_threshold) {
    return std::make_shared<rknnYolov8Seg>(model_path, nms_threshold, conf_threshold);
}

};  // namespace FasterRKNN
