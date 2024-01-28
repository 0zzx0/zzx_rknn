/**
 * @file rknn_postprocess.cpp
 * @author zzx
 * @brief 后处理类的实现
 * @version 0.1
 * @date 2023-07-22
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "rknn_postprocess.h"

namespace FasterRKNN {

int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

static float InterSectionArea(const ObjBox &a, const ObjBox &b) {
    if(a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1) {
        return 0.0f;
    }
    float inter_w = std::min(a.x2, b.x2) - std::min(a.x1, b.x1);
    float inter_h = std::min(a.y2, b.y2) - std::min(a.y1, b.y1);

    return inter_w * inter_h;
}

static bool ScoreSort(ObjBox a, ObjBox b) {
    return (a.score > b.score);
}

void nms(std::vector<ObjBox> &src_boxes, std::vector<ObjBox> &dst_boxes, float threshold) {
    std::vector<int> picked;
    std::sort(src_boxes.begin(), src_boxes.end(), ScoreSort);

    for(int i = 0; i < src_boxes.size(); i++) {
        int keep = 1;
        for(int j = 0; j < picked.size(); j++) {
            float inter_area = InterSectionArea(src_boxes[i], src_boxes[picked[j]]);
            float union_area = src_boxes[i].area() + src_boxes[picked[j]].area() - inter_area;
            float iou = inter_area / union_area;
            if((iou > threshold) && (src_boxes[i].category == src_boxes[picked[j]].category)) {
                keep = 0;
                break;
            }
        }
        if(keep) {
            picked.push_back(i);
        }
    }
    for(int i = 0; i < picked.size(); i++) {
        dst_boxes.push_back(src_boxes[picked[i]]);
    }
    return;
}

void generate_grids_and_stride(std::vector<int> &strides, std::vector<GridAndStride> &grid_strides,
                               int input_w_) {
    for(auto stride : strides) {
        int num_grid = input_w_ / stride;
        for(int g1 = 0; g1 < num_grid; g1++) {
            for(int g0 = 0; g0 < num_grid; g0++) {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

};  // namespace FasterRKNN
