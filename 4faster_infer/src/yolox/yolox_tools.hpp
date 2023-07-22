#pragma once

/**
 * @file yolox_tools.hpp
 * @author your name (you@domain.com)
 * @brief yolox需要的一些处理功能
 * @version 0.1
 * @date 2023-07-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "opencv2/opencv.hpp"

/**
 * @brief 保持长宽比resize
 * 
 * @param img 输入图像
 * @param input_w 目标宽度
 * @param input_h 目标高度
 * @return cv::Mat 返回resize后的结果
 */
static cv::Mat static_resize(const cv::Mat& img, int input_w, int input_h) {
    float r = std::min(input_w / (img.cols*1.0), input_h / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}


