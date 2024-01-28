#pragma once

/**
 * @file yolox_postprocess.h
 * @author zzx
 * @brief yolox后处理
 * @version 0.1
 * @date 2024-01-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "../../postprocess/rknn_postprocess.h"

namespace FasterRKNN {

/**
 * @brief Yolox 后处理类
 *
 */
class YoloxPostProcess {
public:
    /**
     * @brief 构造函数。创建yolox后处理对象
     *
     * @param input_size 模型输入尺寸
     * @param prob_threshold 分类阈值
     * @param nms_threshold nms阈值
     * @param output_attrs rknn输出张量信息
     * @param zps 量化参数1
     * @param scales 量化参数2
     */
    YoloxPostProcess(rknn_model_info *model_info, float prob_threshold, float nms_threshold,
                     std::vector<int32_t> &zps, std::vector<float> &scales);
    ~YoloxPostProcess() = default;

    /**
     * @brief 后处理
     *
     * @param src 输出数据指针
     * @param results 结果保存
     * @param img_scale 原始图片resize scale
     */
    void process(int8_t *src, std::vector<ObjBox> &results, float img_scale);

private:
    std::vector<int> strides_{8, 16, 32};      // stride
    std::vector<GridAndStride> grid_strides_;  // grid_strides
    std::vector<ObjBox> out_boxes;
    // std::vector<ObjBox> nms_boxes;

    int input_size_;   // 输入大小
    int num_grid_;     // grid数量
    int num_class_;    // 模型类别数
    int num_anchors_;  // anchors数量1

    int each_grid_long_;  // 步长

    float prob_threshold_;  // socre 阈值
    float nms_threshold_;   // nms 阈值

    int32_t zp_;   // 量化参数
    float scale_;  // 量化参数
};

};  // namespace FasterRKNN