#pragma once
/**
 * @file rknn_postprocess.h
 * @author zzx
 * @brief 后处理
 * @version 0.1
 * @date 2023-07-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "../base/rknn_infer_base.hpp"

namespace FasterRKNN {

/**
 * @brief 目标框结构
 *
 */
struct ObjBox {
    float GetWidth() { return (x2 - x1); };
    float GetHeight() { return (y2 - y1); };
    float area() { return GetWidth() * GetHeight(); };
    ObjBox() {}
    ObjBox(int x1_, int y1_, int x2_, int y2_, int category_, float score_)
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_), category(category_), score(score_) {}

    int x1;
    int y1;
    int x2;
    int y2;

    int category;
    float score;
};

/**
 * @brief 分割数据结构
 *
 */
struct SegRes {
    std::vector<ObjBox> objs;
    std::vector<uint8_t> segs;
};

/**
 * @brief grid stride
 *
 */
struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

/**
 * @brief 限幅
 *
 * @param val 待处理值
 * @param min 最小
 * @param max 最大
 * @return int32_t 返回
 */
inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

/**
 * @brief 量化f32 -> int8
 *
 * @param f32 float数据
 * @param zp 量化参数
 * @param scale 量化参数
 * @return int8_t 量化数据
 */
int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);

/**
 * @brief 反量化 int8 -> fp32
 *
 * @param qnt 量化数据
 * @param zp 量化参数
 * @param scale 量化参数
 * @return float float值
 */
float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);

/**
 * @brief 计算IOU中的重合部分
 *
 * @param a ObjBox对象1
 * @param b ObjBox对象2
 * @return float 面积
 */
static float InterSectionArea(const ObjBox &a, const ObjBox &b);

/**
 * @brief 排序
 *
 * @param a ObjBox对象1
 * @param b ObjBox对象1
 * @return true a.score > b.score
 * @return false a.score <= b.score
 */
static bool ScoreSort(ObjBox a, ObjBox b);

/**
 * @brief NMS
 *
 * @param src_boxes 原始数据
 * @param dst_boxes 结果数据
 * @param threshold NMS阈值
 */
void nms(std::vector<ObjBox> &src_boxes, std::vector<ObjBox> &dst_boxes, float threshold);

// 特征图映射原图
void generate_grids_and_stride(std::vector<int> &strides, std::vector<GridAndStride> &grid_strides,
                               int input_w_);

};  // namespace FasterRKNN