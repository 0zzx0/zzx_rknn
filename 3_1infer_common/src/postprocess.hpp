#ifndef POSTPROCESS_HPP
#define POSTPROCESS_HPP

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "rknn_api.h"

// 限幅
inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

// f32 -> int8
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {

    float  dst_val = (f32 / scale) + zp;
    int8_t res     = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

// int8 -> fp32
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { 

    return ((float)qnt - (float)zp) * scale; 
}



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

// 计算IOU中的重合部分
static float InterSectionArea(const ObjBox &a, const ObjBox &b){
    if(a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1){
        return 0.0f;
    }
    float inter_w = std::min(a.x2, b.x2) - std::min(a.x1, b.x1);
    float inter_h = std::min(a.y2, b.y2) - std::min(a.y1, b.y1);

    return inter_w * inter_h; 
}

// 排序
static bool ScoreSort(ObjBox a, ObjBox b){
    return (a.score > b.score);
}

// nms
static void nms(std::vector<ObjBox> &src_boxes, std::vector<ObjBox> &dst_boxes, float threshold){
    std::vector<int> picked;
    std::sort(src_boxes.begin(), src_boxes.end(), ScoreSort);

    for (int i = 0; i < src_boxes.size(); i++){
        int keep = 1;
        for(int j=0; j < picked.size(); j++){
            float inter_area = InterSectionArea(src_boxes[i], src_boxes[picked[j]]);
            float union_area = src_boxes[i].area() + src_boxes[picked[j]].area() - inter_area;
            float iou = inter_area / union_area;
            if((iou > threshold) && (src_boxes[i].category == src_boxes[picked[j]].category)){
                keep = 0;
                break;
            }
        }
        if(keep){
            picked.push_back(i);
        }
    }
    for(int i=0;i<picked.size();i++){
        dst_boxes.push_back(src_boxes[picked[i]]);
    }
    return ;

}

// 特征图映射原图
static void generate_grids_and_stride(std::vector<int> &strides,std::vector<GridAndStride> &grid_strides, int input_w_){
    
    for( auto stride: strides){
        int num_grid = input_w_ / stride;
        for(int g1=0; g1<num_grid; g1++){
            for (int g0 = 0; g0<num_grid; g0++){
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}


class YoloxPostProcess{

public:
    YoloxPostProcess(int input_size, float prob_threshold, float nms_threshold, std::vector<rknn_tensor_attr> &output_attrs){
        input_size_ = input_size;
        prob_threshold_ = prob_threshold;
        nms_threshold_ = nms_threshold;
        generate_grids_and_stride(strides_, grid_strides_, input_size_);

        num_grid_ = output_attrs[0].dims[1];
        num_class_ = output_attrs[0].dims[2] - 5;
        num_anchors_ = grid_strides_.size();

        each_grid_long_ = output_attrs[0].dims[2];

    }
    ~YoloxPostProcess() { }

    std::vector<ObjBox> process(int8_t *src, std::vector<int32_t> &zps, std::vector<float> &scales) {

        out_boxes.clear();
        nms_boxes.clear();

        const int8_t *feat_ptr = src;
        
        int32_t zp = zps[0];
        float scale = scales[0];
       
        for (int anchor_idx = 0; anchor_idx < num_anchors_; anchor_idx++) {

            const int grid0 = grid_strides_[anchor_idx].grid0;
            const int grid1 = grid_strides_[anchor_idx].grid1;
            const int stride = grid_strides_[anchor_idx].stride;

            // yolox/models/yolo_head.py decode logic
            //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
            //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
            float x_center = (deqnt_affine_to_f32(feat_ptr[0], zp, scale) + grid0) * stride;
            float y_center = (deqnt_affine_to_f32(feat_ptr[1], zp, scale) + grid1) * stride;
            float w = exp(deqnt_affine_to_f32(feat_ptr[2], zp, scale)) * stride;
            float h = exp(deqnt_affine_to_f32(feat_ptr[3], zp, scale)) * stride;
            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;
            float x1 = x_center + w * 0.5f;
            float y1 = y_center + h * 0.5f;

            
            float box_objectness = deqnt_affine_to_f32(feat_ptr[4], zp, scale);    
            for (int class_idx = 0; class_idx < num_class_; class_idx++){
                float box_cls_score = deqnt_affine_to_f32(feat_ptr[5 + class_idx], zp, scale) ;
                float box_prob = box_objectness * box_cls_score;
                if (box_prob > prob_threshold_){

                    ObjBox obj;
                    obj.x1 = x0;
                    obj.y1 = y0;
                    obj.x2 = x1;
                    obj.y2 = y1;

                    obj.category = class_idx;
                    obj.score = box_prob;

                    out_boxes.push_back(obj);
                }

            } // class loop
           
            feat_ptr += each_grid_long_;

        } // point anchor loop

        nms(out_boxes, nms_boxes, nms_threshold_);
        return nms_boxes;
    }


    std::vector<ObjBox> f16process(float *src) {


        out_boxes.clear();
        nms_boxes.clear();

        const float *feat_ptr = src;
        
        for (int anchor_idx = 0; anchor_idx < num_anchors_; anchor_idx++) {

            const int grid0 = grid_strides_[anchor_idx].grid0;
            const int grid1 = grid_strides_[anchor_idx].grid1;
            const int stride = grid_strides_[anchor_idx].stride;

            // yolox/models/yolo_head.py decode logic
            //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
            //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
            float x_center = (feat_ptr[0] + grid0) * stride;
            float y_center = (feat_ptr[1] + grid1) * stride;
            float w = exp(feat_ptr[2]) * stride;
            float h = exp(feat_ptr[3]) * stride;
            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;
            float x1 = x_center + w * 0.5f;
            float y1 = y_center + h * 0.5f;

            // printf("%f, %f, %f, %f\n", x0, x1, y0, y1);
            

            float box_objectness = feat_ptr[4];    
            for (int class_idx = 0; class_idx < num_class_; class_idx++){

                float box_cls_score = feat_ptr[5 + class_idx];
                float box_prob = box_objectness * box_cls_score;

                if (box_prob > prob_threshold_){

                    ObjBox obj;
                    obj.x1 = x0;
                    obj.y1 = y0;
                    obj.x2 = x1;
                    obj.y2 = y1;

                    obj.category = class_idx;
                    obj.score = box_prob;

                    out_boxes.push_back(obj);
                }

            } // class loop
           
            feat_ptr += each_grid_long_;

        } // point anchor loop
        printf("nms before: %ld\n", out_boxes.size());
        
        nms(out_boxes, nms_boxes, nms_threshold_);
        return nms_boxes;
    }

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



#endif