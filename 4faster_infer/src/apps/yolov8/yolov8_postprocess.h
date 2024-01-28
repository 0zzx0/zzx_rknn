#pragma once

#include "rknn_matmul_api.h"
#include "Float16.h"
#include "../../postprocess/rknn_postprocess.h"

namespace FasterRKNN {

#define OBJ_CLASS_NUM 1

#define OBJ_NUMB_MAX_SIZE 32

#define PROTO_CHANNEL 32
#define PROTO_HEIGHT 160
#define PROTO_WEIGHT 160

static int process_i8(std::vector<rknn_tensor_mem *> &all_input, int input_id, int grid_h,
                      int grid_w, int height, int width, int stride, int dfl_len,
                      std::vector<float> &boxes, std::vector<float> &segments, float *proto,
                      std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                      rknn_model_info *model_info);

static int process_fp32(std::vector<rknn_tensor_mem *> &all_input, int input_id, int grid_h,
                        int grid_w, int height, int width, int stride, int dfl_len,
                        std::vector<float> &boxes, std::vector<float> &segments, float *proto,
                        std::vector<float> &objProbs, std::vector<int> &classId, float threshold);

static void compute_dfl(float *tensor, int dfl_len, float *box);

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right,
                                     std::vector<int> &indices);

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1,
                              float ymin1, float xmax1, float ymax1);
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds,
               std::vector<int> &order, int filterId, float threshold);

static void matmul_by_npu_i8(std::vector<float> &A_input, float *B_input, uint8_t *C_input,
                             int ROWS_A, int COLS_A, int COLS_B, rknn_model_info *model_info);

static void matmul_by_npu_fp16(std::vector<float> &A_input, float *B_input, uint8_t *C_input,
                               int ROWS_A, int COLS_A, int COLS_B, rknn_model_info *model_info);

static void crop_mask(uint8_t *seg_mask, uint8_t *all_mask_in_one, float *boxes, int boxes_num,
                      int *cls_id, int height, int width);

static void resize_by_opencv(uint8_t *input_image, int input_width, int input_height,
                             uint8_t *output_image, int target_width, int target_height);

class yolov8SegPostProcess {
public:
    yolov8SegPostProcess(rknn_model_info *model_info, float prob_threshold, float nms_threshold);

    ~yolov8SegPostProcess() = default;

    void process(std::vector<rknn_tensor_mem *> &outputs, SegRes &res, float scale);

private:
    // SegRes out_seg_;

    rknn_model_info *model_info_;

    int model_in_w_;
    int model_in_h_;
    int dfl_len_;
    int output_per_branch_;

    float conf_threshold_;
    float nms_threshold_;
};

};  // namespace FasterRKNN