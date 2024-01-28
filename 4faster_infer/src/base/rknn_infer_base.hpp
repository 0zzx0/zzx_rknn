#pragma once
/**
 * @file rknn_infer_base.hpp
 * @author zzx
 * @brief rknn推理基类
 * @version 0.1
 * @date 2023-07-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cstdio>
#include <string>
#include <chrono>
#include <vector>
#include "opencv2/opencv.hpp"

#include "rknn_api.h"
#include "tools.hpp"

namespace FasterRKNN {
struct rknn_model_info {
    rknn_input_output_num io_num;    // 输入输出数量
    rknn_tensor_attr *input_attrs;   // 输入信息
    rknn_tensor_attr *output_attrs;  // 输出信息
    int input_c;                     // 输入通道
    int input_w;                     // 输入宽度
    int input_h;                     // 输入高度
    bool is_quant = true;
};

/**
 * @brief 打印输入输出相关信息
 *
 * @param attr rknn_tensor_attr* 类型, 需要解析打印的数据
 */
static void dump_tensor_attr(rknn_tensor_attr *attr) {
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for(int i = 1; i < attr->n_dims; ++i) {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }

    printf(
        "    index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, "
        "size_with_stride=%d, fmt=%s, "
        "type=%s, qnt_type=%s, "
        "zp=%d, scale=%f\n",
        attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size,
        attr->w_stride, attr->size_with_stride, get_format_string(attr->fmt),
        get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

/**
 * @brief rknn 推理基类 采用零拷贝API
 *
 * @tparam OUTPUT 推理结果输出类型
 */
template <class OUTPUT>
class RknnInferBase {
public:
    /**
     * @brief 析构 销毁相关资源
     *
     */
    virtual ~RknnInferBase();

    /**
     * @brief 初始化rknn模型资源
     *
     * @param model_path rknn文件路径
     */
    virtual void Init(const std::string &model_path);

    /**
     * @brief 设置npu核心
     *
     * @param core_mask rknn_core_mask类型 指定npu核心, 默认 RKNN_NPU_CORE_AUTO
     */
    void set_npu_core(rknn_core_mask &core_mask);

    /**
     * @brief 推理图片, 基类必须重写
     *
     * @param img 输入图片 cv::Mat类型
     * @return OUTPUT
     */
    virtual OUTPUT infer(const cv::Mat &img) = 0;  // 纯虚函数

protected:
    std::string model_path_;  // 模型路径
    rknn_context ctx_;        // 上下文

    rknn_model_info model_info_;

    std::vector<rknn_tensor_mem *> input_mems_;   // 输入张量
    std::vector<rknn_tensor_mem *> output_mems_;  // 输出张量

    unsigned char *model_data_ = nullptr;  // 模型数据

    std::vector<float> out_scales_;  // 反量化参数scales
    std::vector<int32_t> out_zps_;   // 反量化参数zps

private:
    /**
     * @brief 打印sdk和驱动的版本信息
     *
     */
    void print_version_info();
    /**
     * @brief 设置输入形状
     *
     */
    virtual void set_input_hwc();  // 看情况是否重写，默认是单输入

    /**
     * @brief 申请输入输出内存
     *
     */
    virtual void init_io_tensor_mem();  // 一般也要重写，指定输入输出类型和大小
};

template <class OUTPUT>
RknnInferBase<OUTPUT>::~RknnInferBase() {
    if(model_data_ != nullptr) {
        free(model_data_);
    }

    if(input_mems_.size() == model_info_.io_num.n_input) {
        for(size_t i = 0; i < model_info_.io_num.n_input; i++) {
            CHECK_RKNN(rknn_destroy_mem(ctx_, input_mems_[i]));
        }
    }
    if(output_mems_.size() == model_info_.io_num.n_output) {
        for(size_t i = 0; i < model_info_.io_num.n_output; i++) {
            CHECK_RKNN(rknn_destroy_mem(ctx_, output_mems_[i]));
        }
    }

    CHECK_RKNN(rknn_destroy(ctx_));
    printf("RknnInferBase release! \n");
}

template <class OUTPUT>
void RknnInferBase<OUTPUT>::Init(const std::string &model_path) {
    this->model_path_ = model_path;

    // 加载文件
    int model_data_size = 0;
    model_data_ = load_model(model_path.c_str(), &model_data_size);

    // 加载模型
    CHECK_RKNN(rknn_init(&ctx_, model_data_, model_data_size, 0, NULL));

    // 打印sdk信息
    print_version_info();

    // 获取输入输出数量
    CHECK_RKNN(
        rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &model_info_.io_num, sizeof(model_info_.io_num)));
    printf("io number info: input num: %d, output num: %d\n", model_info_.io_num.n_input,
           model_info_.io_num.n_output);

    // 获取io attrs
    model_info_.input_attrs =
        (rknn_tensor_attr *)malloc(model_info_.io_num.n_input * sizeof(rknn_tensor_attr));
    model_info_.output_attrs =
        (rknn_tensor_attr *)malloc(model_info_.io_num.n_output * sizeof(rknn_tensor_attr));
    // input_attrs_.resize(model_info_.io_num_.n_input);
    // output_attrs_.resize(model_info_.io_num_.n_output);

    for(size_t i = 0; i < model_info_.io_num.n_input; i++) {
        model_info_.input_attrs[i].index = i;
        CHECK_RKNN(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(model_info_.input_attrs[i]),
                              sizeof(rknn_tensor_attr)));
        printf("input[%ld]:\n", i);
        dump_tensor_attr(&(model_info_.input_attrs[i]));
    }

    for(size_t i = 0; i < model_info_.io_num.n_output; i++) {
        model_info_.output_attrs[i].index = i;
        CHECK_RKNN(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(model_info_.output_attrs[i]),
                              sizeof(rknn_tensor_attr)));
        printf("output[%ld]:\n", i);
        dump_tensor_attr(&(model_info_.output_attrs[i]));
    }

    // 获取反量化参数
    for(size_t i = 0; i < model_info_.io_num.n_output; ++i) {
        out_scales_.push_back(model_info_.output_attrs[i].scale);
        out_zps_.push_back(model_info_.output_attrs[i].zp);
    }

    // set_npu_core();

    // 获取输入形状
    set_input_hwc();
    // 初始化输入输出
    init_io_tensor_mem();
}

template <class OUTPUT>
void RknnInferBase<OUTPUT>::print_version_info() {
    rknn_sdk_version version;
    CHECK_RKNN(rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version)));
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
}

template <class OUTPUT>
void RknnInferBase<OUTPUT>::set_input_hwc() {
    if(model_info_.input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        model_info_.input_c = model_info_.input_attrs[0].dims[1];
        model_info_.input_h = model_info_.input_attrs[0].dims[2];
        model_info_.input_w = model_info_.input_attrs[0].dims[3];
    } else {
        model_info_.input_h = model_info_.input_attrs[0].dims[1];
        model_info_.input_w = model_info_.input_attrs[0].dims[2];
        model_info_.input_c = model_info_.input_attrs[0].dims[3];
    }
}

template <class OUTPUT>
void RknnInferBase<OUTPUT>::set_npu_core(rknn_core_mask &core_mask) {
    CHECK_RKNN(rknn_set_core_mask(ctx_, core_mask));
}

template <class OUTPUT>
void RknnInferBase<OUTPUT>::init_io_tensor_mem() {
    input_mems_.resize(model_info_.io_num.n_input);
    output_mems_.resize(model_info_.io_num.n_output);

    for(size_t i = 0; i < model_info_.io_num.n_input; i++) {
        model_info_.input_attrs[i].type = RKNN_TENSOR_UINT8;
        input_mems_[i] = rknn_create_mem(ctx_, model_info_.input_attrs[i].size_with_stride);
        rknn_set_io_mem(ctx_, input_mems_[i], &model_info_.input_attrs[i]);
    }

    for(size_t i = 0; i < model_info_.io_num.n_output; i++) {
        model_info_.output_attrs[i].type = RKNN_TENSOR_INT8;
        output_mems_[i] =
            rknn_create_mem(ctx_, model_info_.output_attrs[i].n_elems * sizeof(int8_t));
        rknn_set_io_mem(ctx_, output_mems_[i], &model_info_.output_attrs[i]);
    }
}

};  // namespace FasterRKNN