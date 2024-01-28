/**
 * @file rknn_preprocess.cpp
 * @author zzx
 * @brief
 * @version 0.1
 * @date 2024-01-28
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "rknn_preprocess.h"

namespace FasterRKNN {

cv::Mat static_resize(const cv::Mat& img, int input_w, int input_h) {
    float r = std::min(input_w / (img.cols * 1.0), input_h / (img.rows * 1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

};  // namespace FasterRKNN