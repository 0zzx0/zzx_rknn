#pragma once
/**
 * @file rknn_yolox.h
 * @author zzx
 * @brief yolox 推理类
 * @version 0.1
 * @date 2023-07-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <memory>

#include "../base/rknn_infer_base.hpp"
#include "../postprocess/rknn_postprocess.h"
#include "yolox_tools.hpp"

using RknnInferBaseObjBox = RknnInferBase<std::vector<ObjBox>>;
/**
 * @brief Yolox类
 *
 */
class RknnYolox : public RknnInferBaseObjBox {
public:
    /**
     * @brief 构造函数 创建yolox推理对象
     *
     * @param model_path rknn模型路径
     * @param nms_threshold nms阈值
     * @param conf_threshold 置信度阈值
     */
    RknnYolox(const std::string &model_path, const float nms_threshold, const float conf_threshold);

    /**
     * @brief 析构
     *
     */
    virtual ~RknnYolox();

    /**
     * @brief 推理函数
     *
     * @param img 输入图像
     * @return std::vector<ObjBox> 返回结果
     */
    virtual std::vector<ObjBox> infer(const cv::Mat &img) override;  // yolox推理

private:
    std::shared_ptr<YoloxPostProcess> postprocess_;  // 后处理对象指针
    std::vector<ObjBox> results_;                    // 结果
};

/**
 * @brief 创建一个yolox对象 返回一个父类指针
 *
 * @param model_path rknn模型路径
 * @param nms_threshold nms阈值
 * @param conf_threshold 置信度阈值
 * @return std::shared_ptr<RknnInferBaseObjBox> 返回父类指针
 */
std::shared_ptr<RknnInferBaseObjBox> create_infer_yolox(const std::string &model_path,
                                                        const float nms_threshold,
                                                        const float conf_threshold);
