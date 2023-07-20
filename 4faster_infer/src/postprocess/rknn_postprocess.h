#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "rknn_api.h"

// 目标框
struct ObjBox{
    float GetWidth() { return (x2 - x1); };
    float GetHeight() { return (y2 - y1); };
    float area() { return GetWidth() * GetHeight(); };

    int x1;
    int y1;
    int x2;
    int y2;

    int category;
    float score;
};

// grid stride
struct GridAndStride{
    int grid0;
    int grid1;
    int stride;
};



// 限幅
inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

// f32 -> int8
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);
// int8 -> fp32
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);


// 计算IOU中的重合部分
static float InterSectionArea(const ObjBox &a, const ObjBox &b);

// 排序
static bool ScoreSort(ObjBox a, ObjBox b);

// nms
static void nms(std::vector<ObjBox> &src_boxes, std::vector<ObjBox> &dst_boxes, float threshold);

// 特征图映射原图
static void generate_grids_and_stride(std::vector<int> &strides,std::vector<GridAndStride> &grid_strides, int input_w_);


class YoloxPostProcess {

public:
    YoloxPostProcess(int input_size, float prob_threshold, float nms_threshold, std::vector<rknn_tensor_attr> &output_attrs);
    ~YoloxPostProcess() { }
    std::vector<ObjBox> process(int8_t *src, std::vector<int32_t> &zps, std::vector<float> &scales);

private:
    std::vector<int> strides_{8, 16, 32};
    std::vector<GridAndStride> grid_strides_;
    std::vector<ObjBox> out_boxes;
    std::vector<ObjBox> nms_boxes;

    int input_size_;
    int num_grid_;
    int num_class_;
    int num_anchors_;

    int each_grid_long_;

    float prob_threshold_;
    float nms_threshold_;

};

