#pragma once

/**
 * @file yolox_tools.hpp
 * @author zzx
 * @brief yolox需要的一些处理功能
 * @version 0.1
 * @date 2023-07-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "opencv2/opencv.hpp"

namespace FasterRKNN {

/**
 * @brief 保持长宽比resize
 *
 * @param img 输入图像
 * @param input_w 目标宽度
 * @param input_h 目标高度
 * @return cv::Mat 返回resize后的结果
 */
cv::Mat static_resize(const cv::Mat& img, int input_w, int input_h);

};  // namespace FasterRKNN