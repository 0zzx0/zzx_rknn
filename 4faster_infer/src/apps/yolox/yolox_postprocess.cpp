/**
 * @file yolox_postprocess.cpp
 * @author your name (you@domain.com)
 * @brief yolox后处理实现
 * @version 0.1
 * @date 2024-01-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "yolox_postprocess.h"

namespace FasterRKNN {

YoloxPostProcess::YoloxPostProcess(rknn_model_info *model_info, float prob_threshold,
                                   float nms_threshold, std::vector<int32_t> &zps,
                                   std::vector<float> &scales) {
    input_size_ = model_info->input_h;
    prob_threshold_ = prob_threshold;
    nms_threshold_ = nms_threshold;
    generate_grids_and_stride(strides_, grid_strides_, input_size_);

    num_grid_ = model_info->output_attrs[0].dims[1];
    num_class_ = model_info->output_attrs[0].dims[2] - 5;
    num_anchors_ = grid_strides_.size();

    each_grid_long_ = model_info->output_attrs[0].dims[2];

    zp_ = zps[0];
    scale_ = scales[0];
}

void YoloxPostProcess::process(int8_t *src, std::vector<ObjBox> &results, float img_scale) {
    // std::vector<ObjBox> out_boxes;  // nms前
    out_boxes.clear();
    // nms_boxes.clear();
    results.clear();

    const int8_t *feat_ptr = src;

    for(int anchor_idx = 0; anchor_idx < num_anchors_; anchor_idx++) {
        float box_objectness = deqnt_affine_to_f32(feat_ptr[4], zp_, scale_);
        for(int class_idx = 0; class_idx < num_class_; class_idx++) {
            float box_cls_score = deqnt_affine_to_f32(feat_ptr[5 + class_idx], zp_, scale_);
            float box_prob = box_objectness * box_cls_score;
            if(box_prob > prob_threshold_) {
                const int grid0 = grid_strides_[anchor_idx].grid0;
                const int grid1 = grid_strides_[anchor_idx].grid1;
                const int stride = grid_strides_[anchor_idx].stride;

                float x_center = (deqnt_affine_to_f32(feat_ptr[0], zp_, scale_) + grid0) * stride;
                float y_center = (deqnt_affine_to_f32(feat_ptr[1], zp_, scale_) + grid1) * stride;
                float w = exp(deqnt_affine_to_f32(feat_ptr[2], zp_, scale_)) * stride;
                float h = exp(deqnt_affine_to_f32(feat_ptr[3], zp_, scale_)) * stride;

                ObjBox obj;
                obj.x1 = (x_center - w * 0.5f) / img_scale;
                obj.y1 = (y_center - h * 0.5f) / img_scale;
                obj.x2 = (x_center + w * 0.5f) / img_scale;
                obj.y2 = (y_center + h * 0.5f) / img_scale;

                obj.category = class_idx;
                obj.score = box_prob;

                out_boxes.push_back(obj);
            }

        }  // class loop

        feat_ptr += each_grid_long_;

    }  // point anchor loop

    nms(out_boxes, results, nms_threshold_);
    return;
}

};  // namespace FasterRKNN
