#pragma once

/**
 * @file yolox_tools.hpp
 * @author zzx
 * @brief yolov8 seg
 * @version 0.1
 * @date 2024-1-27 15:34:46
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <memory>

#include "../../base/rknn_infer_base.hpp"
#include "../../postprocess/rknn_postprocess.h"
#include "../../preprocess/rknn_preprocess.h"
#include "yolov8_seg_postprocess.h"

namespace FasterRKNN {

using rknnInferBaseSegRes = RknnInferBase<SegRes>;
/**
 * @brief Yolox类
 *
 */
class rknnYolov8Seg : public rknnInferBaseSegRes {
public:
    /**
     * @brief 构造函数 创建推理对象
     *
     * @param model_path rknn模型路径
     * @param nms_threshold nms阈值
     * @param conf_threshold 置信度阈值
     */
    rknnYolov8Seg(const std::string &model_path, const float nms_threshold,
                  const float conf_threshold);

    /**
     * @brief 析构
     *
     */
    virtual ~rknnYolov8Seg();

    /**
     * @brief 推理函数
     *
     * @param img 输入图像
     * @return std::vector<SegRes> 返回结果
     */
    virtual SegRes infer(const cv::Mat &img) override;  // yolox推理

private:
    std::shared_ptr<yolov8SegPostProcess> postprocess_;  // 后处理对象指针
    SegRes results_;                                     // 结果
};

/**
 * @brief 创建一个推理模型对象 返回一个父类指针
 *
 * @param model_path rknn模型路径
 * @param nms_threshold nms阈值
 * @param conf_threshold 置信度阈值
 * @return std::shared_ptr<rknnInferBaseSegRes> 返回父类指针
 */
std::shared_ptr<rknnInferBaseSegRes> create_infer_yolov8seg(const std::string &model_path,
                                                            const float nms_threshold,
                                                            const float conf_threshold);

};  // namespace FasterRKNN
