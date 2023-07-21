#pragma once

#include <cstdio>
#include <string>
#include <chrono>
#include <vector>
#include "opencv2/opencv.hpp"


#include "rknn_api.h"
#include "tools.h"

/*
    @author: zzx
    @date: 2023-7-20 13:45:31
    
    把rknn 推理基类 零拷贝API
*/
template<class OUTPUT>
class RknnInferBase {

public:
    // 析构
    virtual ~RknnInferBase();

    // 初始化
    virtual void Init(const std::string &model_path);
    // 打印sdk和驱动的版本信息
    void print_version_info();

    // 设置npu核心
    void set_npu_core(rknn_core_mask &core_mask);

     // 获得输入形状
    virtual void set_input_hwc();           // 看情况是否重写，默认是单输入
    // 申请输入输出内存
    virtual void init_io_tensor_mem();      // 一般也要重写，指定输入输出类型和大小
    // 推理图片
    virtual OUTPUT infer(const cv::Mat &img) = 0; // 纯虚函数


protected:
    std::string model_path_;         // 模型路径
	rknn_context ctx_;               // 上下文
    rknn_input_output_num io_num_;   // 输入输出数量

    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;

    std::vector<rknn_tensor_mem*> input_mems_;
    std::vector<rknn_tensor_mem*> output_mems_;

    unsigned char* model_data_ = nullptr;   // 模型

    std::vector<float>    out_scales_;  // 反量化
	std::vector<int32_t>  out_zps_;     // 反量化
	
    int input_h_;
    int input_w_;
    int input_channel_;

};

// 打印信息
static void dump_tensor_attr(rknn_tensor_attr* attr);


