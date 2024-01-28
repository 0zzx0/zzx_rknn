/**
 * @file rknn_yolox.cpp
 * @author zzx
 * @brief yolox推理类的实现
 * @version 0.1
 * @date 2023-07-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "rknn_yolox.h"

namespace FasterRKNN {

RknnYolox::~RknnYolox() {
    printf("begin release! \n");
}

RknnYolox::RknnYolox(const std::string &model_path, const float nms_threshold,
                     const float conf_threshold) {
    Init(model_path);
    postprocess_ = std::make_shared<YoloxPostProcess>(&model_info_, conf_threshold, nms_threshold,
                                                      out_zps_, out_scales_);
}

std::vector<ObjBox> RknnYolox::infer(const cv::Mat &img) {
    auto img_out = static_resize(img, model_info_.input_w, model_info_.input_h);
    float img_scale =
        std::min(model_info_.input_w / (img.cols * 1.0), model_info_.input_h / (img.rows * 1.0));
    memcpy(input_mems_[0]->virt_addr, img_out.data, model_info_.input_attrs[0].size_with_stride);
    CHECK_RKNN(rknn_run(ctx_, NULL));
    postprocess_->process((int8_t *)output_mems_[0]->virt_addr, results_, img_scale);

    return results_;
}

std::shared_ptr<RknnInferBaseObjBox> create_infer_yolox(const std::string &model_path,
                                                        const float nms_threshold,
                                                        const float conf_threshold) {
    return std::make_shared<RknnYolox>(model_path, nms_threshold, conf_threshold);
}

};  // namespace FasterRKNN