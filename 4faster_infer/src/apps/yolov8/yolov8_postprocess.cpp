/**
 * @file yolov8_seg_postprocess.cpp
 * @author your name (you@domain.com)
 * @brief yolov8 分割后处理
 * @version 0.1
 * @date 2024-01-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "yolov8_postprocess.h"

namespace FasterRKNN {

static void compute_dfl(float *tensor, int dfl_len, float *box) {
    for(int b = 0; b < 4; b++) {
        float exp_t[dfl_len];
        float exp_sum = 0;
        float acc_sum = 0;
        for(int i = 0; i < dfl_len; i++) {
            exp_t[i] = exp(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }

        for(int i = 0; i < dfl_len; i++) {
            acc_sum += exp_t[i] / exp_sum * i;
        }
        box[b] = acc_sum;
    }
}

static int process_i8(std::vector<rknn_tensor_mem *> &all_input, int input_id, int grid_h,
                      int grid_w, int height, int width, int stride, int dfl_len,
                      std::vector<float> &boxes, std::vector<float> &segments, float *proto,
                      std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                      rknn_model_info *model_info) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    // Skip if input_id is not 0, 4, 8, or 12
    if(input_id % 4 != 0) {
        return validCount;
    }

    if(input_id == 12) {
        int8_t *input_proto = (int8_t *)all_input[input_id]->virt_addr;
        int32_t zp_proto = model_info->output_attrs[input_id].zp;
        for(int i = 0; i < PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT; i++) {
            proto[i] = input_proto[i] - zp_proto;
        }
        return validCount;
    }

    int8_t *box_tensor = (int8_t *)all_input[input_id]->virt_addr;
    int32_t box_zp = model_info->output_attrs[input_id].zp;
    float box_scale = model_info->output_attrs[input_id].scale;

    int8_t *score_tensor = (int8_t *)all_input[input_id + 1]->virt_addr;
    int32_t score_zp = model_info->output_attrs[input_id + 1].zp;
    float score_scale = model_info->output_attrs[input_id + 1].scale;

    int8_t *score_sum_tensor = nullptr;
    int32_t score_sum_zp = 0;
    float score_sum_scale = 1.0;
    score_sum_tensor = (int8_t *)all_input[input_id + 2]->virt_addr;
    score_sum_zp = model_info->output_attrs[input_id + 2].zp;
    score_sum_scale = model_info->output_attrs[input_id + 2].scale;

    int8_t *seg_tensor = (int8_t *)all_input[input_id + 3]->virt_addr;
    int32_t seg_zp = model_info->output_attrs[input_id + 3].zp;
    float seg_scale = model_info->output_attrs[input_id + 3].scale;

    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for(size_t i = 0; i < grid_h; i++) {
        for(size_t j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            int offset_seg = i * grid_w + j;
            int8_t *in_ptr_seg = seg_tensor + offset_seg;

            // for quick filtering through "score sum"
            if(score_sum_tensor != nullptr) {
                if(score_sum_tensor[offset] < score_sum_thres_i8) {
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for(size_t c = 0; c < OBJ_CLASS_NUM; c++) {
                if((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score)) {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if(max_score > score_thres_i8) {
                for(size_t k = 0; k < PROTO_CHANNEL; k++) {
                    int8_t seg_element_i8 = in_ptr_seg[(k)*grid_len] - seg_zp;
                    segments.push_back(seg_element_i8);
                }

                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len * 4];
                for(int k = 0; k < dfl_len * 4; k++) {
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

static int process_fp32(std::vector<rknn_tensor_mem *> &all_input, int input_id, int grid_h,
                        int grid_w, int height, int width, int stride, int dfl_len,
                        std::vector<float> &boxes, std::vector<float> &segments, float *proto,
                        std::vector<float> &objProbs, std::vector<int> &classId, float threshold) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    // Skip if input_id is not 0, 4, 8, or 12
    if(input_id % 4 != 0) {
        return validCount;
    }

    if(input_id == 12) {
        float *input_proto = (float *)all_input[input_id]->virt_addr;
        for(int i = 0; i < PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT; i++) {
            proto[i] = input_proto[i];
        }
        return validCount;
    }

    float *box_tensor = (float *)all_input[input_id]->virt_addr;
    float *score_tensor = (float *)all_input[input_id + 1]->virt_addr;
    float *score_sum_tensor = (float *)all_input[input_id + 2]->virt_addr;
    float *seg_tensor = (float *)all_input[input_id + 3]->virt_addr;

    for(int i = 0; i < grid_h; i++) {
        for(int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            int offset_seg = i * grid_w + j;
            float *in_ptr_seg = seg_tensor + offset_seg;

            // for quick filtering through "score sum"
            if(score_sum_tensor != nullptr) {
                if(score_sum_tensor[offset] < threshold) {
                    continue;
                }
            }

            float max_score = 0;
            for(int c = 0; c < OBJ_CLASS_NUM; c++) {
                if((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score)) {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if(max_score > threshold) {
                for(int k = 0; k < PROTO_CHANNEL; k++) {
                    float seg_element_f32 = in_ptr_seg[(k)*grid_len];
                    segments.push_back(seg_element_f32);
                }

                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len * 4];
                for(int k = 0; k < dfl_len * 4; k++) {
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right,
                                     std::vector<int> &indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if(left < right) {
        key_index = indices[left];
        key = input[left];
        while(low < high) {
            while(low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while(low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1,
                              float ymin1, float xmax1, float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
              (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds,
               std::vector<int> &order, int filterId, float threshold) {
    for(int i = 0; i < validCount; ++i) {
        if(order[i] == -1 || classIds[i] != filterId) {
            continue;
        }
        int n = order[i];
        for(int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if(m == -1 || classIds[i] != filterId) {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if(iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static void matmul_by_npu_i8(std::vector<float> &A_input, float *B_input, uint8_t *C_input,
                             int ROWS_A, int COLS_A, int COLS_B, rknn_model_info *model_info) {
    int B_layout = 0;
    int AC_layout = 0;
    int32_t M = 1;
    int32_t K = COLS_A;
    int32_t N = COLS_B;

    rknn_matmul_ctx ctx;
    rknn_matmul_info info;
    memset(&info, 0, sizeof(rknn_matmul_info));
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = RKNN_INT8_MM_INT8_TO_INT32;
    info.B_layout = B_layout;
    info.AC_layout = AC_layout;

    rknn_matmul_io_attr io_attr;
    memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));

    int8_t int8Vector_A[ROWS_A * COLS_A];
    for(int i = 0; i < ROWS_A * COLS_A; ++i) {
        int8Vector_A[i] = (int8_t)A_input[i];
    }

    int8_t int8Vector_B[COLS_A * COLS_B];
    for(int i = 0; i < COLS_A * COLS_B; ++i) {
        int8Vector_B[i] = (int8_t)B_input[i];
    }

    int ret = rknn_matmul_create(&ctx, &info, &io_attr);
    // Create A
    rknn_tensor_mem *A = rknn_create_mem(ctx, io_attr.A.size);
    // Create B
    rknn_tensor_mem *B = rknn_create_mem(ctx, io_attr.B.size);
    // Create C
    rknn_tensor_mem *C = rknn_create_mem(ctx, io_attr.C.size);

    memcpy(B->virt_addr, int8Vector_B, B->size);
    // Set A
    ret = rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
    // Set B
    ret = rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
    // Set C
    ret = rknn_matmul_set_io_mem(ctx, C, &io_attr.C);

    for(int i = 0; i < ROWS_A; ++i) {
        memcpy(A->virt_addr, int8Vector_A + i * A->size, A->size);

        // Run
        ret = rknn_matmul_run(ctx);

        for(int j = 0; j < COLS_B; ++j) {
            if(((int32_t *)C->virt_addr)[j] > 0) {
                C_input[i * COLS_B + j] = 1;
            } else {
                C_input[i * COLS_B + j] = 0;
            }
        }
    }

    // destroy
    rknn_destroy_mem(ctx, A);
    rknn_destroy_mem(ctx, B);
    rknn_destroy_mem(ctx, C);
    rknn_matmul_destroy(ctx);
}

static void matmul_by_npu_fp16(std::vector<float> &A_input, float *B_input, uint8_t *C_input,
                               int ROWS_A, int COLS_A, int COLS_B, rknn_model_info *model_info) {
    int B_layout = 0;
    int AC_layout = 0;
    int32_t M = ROWS_A;
    int32_t K = COLS_A;
    int32_t N = COLS_B;

    rknn_matmul_ctx ctx;
    rknn_matmul_info info;
    memset(&info, 0, sizeof(rknn_matmul_info));
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout = B_layout;
    info.AC_layout = AC_layout;

    rknn_matmul_io_attr io_attr;
    memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));

    rknpu2::float16 int8Vector_A[ROWS_A * COLS_A];
    for(int i = 0; i < ROWS_A * COLS_A; ++i) {
        int8Vector_A[i] = (rknpu2::float16)A_input[i];
    }

    rknpu2::float16 int8Vector_B[COLS_A * COLS_B];
    for(int i = 0; i < COLS_A * COLS_B; ++i) {
        int8Vector_B[i] = (rknpu2::float16)B_input[i];
    }

    int ret = rknn_matmul_create(&ctx, &info, &io_attr);
    // Create A
    rknn_tensor_mem *A = rknn_create_mem(ctx, io_attr.A.size);
    // Create B
    rknn_tensor_mem *B = rknn_create_mem(ctx, io_attr.B.size);
    // Create C
    rknn_tensor_mem *C = rknn_create_mem(ctx, io_attr.C.size);

    memcpy(A->virt_addr, int8Vector_A, A->size);
    memcpy(B->virt_addr, int8Vector_B, B->size);

    // Set A
    ret = rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
    // Set B
    ret = rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
    // Set C
    ret = rknn_matmul_set_io_mem(ctx, C, &io_attr.C);

    // Run
    ret = rknn_matmul_run(ctx);
    for(int i = 0; i < ROWS_A * COLS_B; ++i) {
        if(((float *)C->virt_addr)[i] > 0) {
            C_input[i] = 1;
        } else {
            C_input[i] = 0;
        }
    }

    // destroy
    rknn_destroy_mem(ctx, A);
    rknn_destroy_mem(ctx, B);
    rknn_destroy_mem(ctx, C);
    rknn_matmul_destroy(ctx);
}

static void crop_mask(uint8_t *seg_mask, uint8_t *all_mask_in_one, float *boxes, int boxes_num,
                      int *cls_id, int height, int width) {
    for(int b = 0; b < boxes_num; b++) {
        float x1 = boxes[b * 4 + 0];
        float y1 = boxes[b * 4 + 1];
        float x2 = boxes[b * 4 + 2];
        float y2 = boxes[b * 4 + 3];

        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                if(j >= x1 && j < x2 && i >= y1 && i < y2) {
                    if(all_mask_in_one[i * width + j] == 0) {
                        all_mask_in_one[i * width + j] =
                            seg_mask[b * width * height + i * width + j] * (cls_id[b] + 1);
                    }
                }
            }
        }
    }
}

static void resize_by_opencv(uint8_t *input_image, int input_width, int input_height,
                             uint8_t *output_image, int target_width, int target_height) {
    cv::Mat src_image(input_height, input_width, CV_8U, input_image);
    cv::Mat dst_image;
    cv::resize(src_image, dst_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
    memcpy(output_image, dst_image.data, target_width * target_height);
}

yolov8SegPostProcess::yolov8SegPostProcess(rknn_model_info *model_info, float prob_threshold,
                                           float nms_threshold) {
    model_info_ = model_info;

    model_in_w_ = model_info->input_w;
    model_in_h_ = model_info->input_h;

    dfl_len_ = model_info->output_attrs[0].dims[1] / 4;
    output_per_branch_ = model_info->io_num.n_output / 3;  // default 3 branch

    conf_threshold_ = prob_threshold;
    nms_threshold_ = nms_threshold;
}

void yolov8SegPostProcess::process(std::vector<rknn_tensor_mem *> &outputs, SegRes &out_seg_,
                                   float scale) {
    out_seg_.objs.clear();

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    std::vector<float> filterSegments;
    float proto[PROTO_CHANNEL * PROTO_HEIGHT * PROTO_WEIGHT];
    std::vector<float> filterSegments_by_nms;

    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;

    // process the outputs of rknn
    for(int i = 0; i < 13; i++) {
        grid_h = model_info_->output_attrs[i].dims[2];
        grid_w = model_info_->output_attrs[i].dims[3];
        stride = model_info_->input_h / grid_h;

        if(model_info_->is_quant) {
            validCount += process_i8(outputs, i, grid_h, grid_w, model_in_h_, model_in_w_, stride,
                                     dfl_len_, filterBoxes, filterSegments, proto, objProbs,
                                     classId, conf_threshold_, model_info_);
        } else {
            validCount += process_fp32(outputs, i, grid_h, grid_w, model_in_h_, model_in_w_, stride,
                                       dfl_len_, filterBoxes, filterSegments, proto, objProbs,
                                       classId, conf_threshold_);
        }
    }

    // nms
    if(validCount <= 0) {
        return;
    }
    std::vector<int> indexArray;
    for(int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for(auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold_);
    }

    int last_count = 0;

    for(int i = 0; i < validCount; ++i) {
        if(indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] / scale;
        float y1 = filterBoxes[n * 4 + 1] / scale;
        float x2 = x1 + filterBoxes[n * 4 + 2] / scale;
        float y2 = y1 + filterBoxes[n * 4 + 3] / scale;
        int id = classId[n];
        float obj_conf = objProbs[i];

        for(int k = 0; k < PROTO_CHANNEL; k++) {
            filterSegments_by_nms.push_back(filterSegments[n * PROTO_CHANNEL + k]);
        }
        out_seg_.objs.emplace_back(x1, y1, x2, y2, id, obj_conf);

        last_count++;
    }

    int boxes_num = out_seg_.objs.size();

    // compute the mask (binary matrix) through Matmul
    int ROWS_A = boxes_num;
    int COLS_A = PROTO_CHANNEL;
    int COLS_B = PROTO_HEIGHT * PROTO_WEIGHT;
    uint8_t matmul_out[boxes_num * PROTO_HEIGHT * PROTO_WEIGHT];
    if(model_info_->is_quant) {
        matmul_by_npu_i8(filterSegments_by_nms, proto, matmul_out, ROWS_A, COLS_A, COLS_B,
                         model_info_);
    } else {
        matmul_by_npu_fp16(filterSegments_by_nms, proto, matmul_out, ROWS_A, COLS_A, COLS_B,
                           model_info_);
    }

    float filterBoxes_by_nms[boxes_num * 4];
    int cls_id[boxes_num];
    for(int i = 0; i < boxes_num; i++) {
        // for crop_mask
        // 640 / 160 = 4.0
        filterBoxes_by_nms[i * 4 + 0] = out_seg_.objs[i].x1 / 4.0 * scale;  // x1;
        filterBoxes_by_nms[i * 4 + 1] = out_seg_.objs[i].y1 / 4.0 * scale;  // y1;
        filterBoxes_by_nms[i * 4 + 2] = out_seg_.objs[i].x2 / 4.0 * scale;  // x2;
        filterBoxes_by_nms[i * 4 + 3] = out_seg_.objs[i].y2 / 4.0 * scale;  // y2;
        cls_id[i] = out_seg_.objs[i].category;
    }

    // crop seg outside box
    uint8_t all_mask_in_one[PROTO_HEIGHT * PROTO_WEIGHT] = {0};
    crop_mask(matmul_out, all_mask_in_one, filterBoxes_by_nms, boxes_num, cls_id, PROTO_HEIGHT,
              PROTO_WEIGHT);

    // // get real mask
    int cropped_height = PROTO_HEIGHT;
    int cropped_width = PROTO_WEIGHT;

    int ori_in_height = model_in_h_ / scale;
    int ori_in_width = model_in_w_ / scale;

    out_seg_.segs.resize(ori_in_height * ori_in_width);
    resize_by_opencv(all_mask_in_one, cropped_width, cropped_height, out_seg_.segs.data(),
                     ori_in_width, ori_in_height);
}

};  // namespace FasterRKNN