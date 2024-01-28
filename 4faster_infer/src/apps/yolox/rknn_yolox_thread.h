#pragma once

/**
 * @file rknn_yolox_thread.h
 * @author zzx
 * @brief yolox多线程
 * @version 0.1
 * @date 2023-07-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <memory>

#include "../../base/rknn_infer_thread_base.hpp"  // 注意这个地方，需要包含实现文件。因为是模板类，仅包含头文件会链接错误。
#include "../../postprocess/rknn_postprocess.h"
#include "../../preprocess/rknn_preprocess.h"
#include "yolox_postprocess.h"

namespace FasterRKNN {

using RknnInferThreadBaseYolo = RknnInferThreadBase<std::vector<ObjBox>, float>;
/**
 * @brief yolox多线程推理类
 *
 */
class RknnYoloxThread : public RknnInferThreadBaseYolo {
public:
    /**
     * @brief yolox 推理基类对象
     *
     * @param model_path rknn模型路径
     * @param nms_threshold nms阈值
     * @param conf_threshold 置信度阈值
     */
    RknnYoloxThread(const std::string &model_path, const float nms_threshold,
                    const float conf_threshold);

    /**
     * @brief 析构
     *
     */
    virtual ~RknnYoloxThread();

    /**
     * @brief 工作线程函数 不断从工作队列中获取图片推理
     *
     */
    void worker();

    /**
     * @brief 预处理过程
     *
     * @param job 工作抽象数据结构
     * @param img 输入图片
     * @return true 成功
     * @return false 失败
     */
    bool preprocess(Job &job, const cv::Mat &img);

private:
    std::shared_ptr<YoloxPostProcess> postprocess_;  // 后处理类指针
    std::vector<ObjBox> results;                     // 结果
    float nms_threshold_;                            // nms阈值
    float conf_threshold_;                           // 置信度阈值
};

/**
 * @brief 创建一个RknnYoloxThread对象 返回一个父类RknnInferThreadBaseYolo指针
 *
 * @param model_path rknn模型路径
 * @param nms_threshold nms阈值
 * @param conf_threshold 置信度阈值
 *
 * @return std::shared_ptr<RknnInferThreadBaseYolo> 返回父类指针
 */
std::shared_ptr<RknnInferThreadBaseYolo> create_infer_yolox_thread(const std::string &model_path,
                                                                   const float nms_threshold,
                                                                   const float conf_threshold);

};  // namespace FasterRKNN