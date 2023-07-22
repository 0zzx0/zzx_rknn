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

#include "rknn_api.h"

/**
 * @brief 目标框结构
 * 
 */
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

/**
 * @brief grid stride
 * 
 */
struct GridAndStride{
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
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);

/**
 * @brief 反量化 int8 -> fp32
 * 
 * @param qnt 量化数据
 * @param zp 量化参数
 * @param scale 量化参数
 * @return float float值
 */
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);


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
static void nms(std::vector<ObjBox> &src_boxes, std::vector<ObjBox> &dst_boxes, float threshold);

// 特征图映射原图
static void generate_grids_and_stride(std::vector<int> &strides, std::vector<GridAndStride> &grid_strides, int input_w_);

/**
 * @brief Yolox 后处理类
 * 
 */
class YoloxPostProcess {

public:
    /**
     * @brief 构造函数。创建yolox后处理对象
     * 
     * @param input_size 模型输入尺寸   
     * @param prob_threshold 分类阈值
     * @param nms_threshold nms阈值
     * @param output_attrs rknn输出张量信息
     * @param zps 量化参数1
     * @param scales 量化参数2
     */
    YoloxPostProcess(int input_size, 
                        float prob_threshold, 
                        float nms_threshold, 
                        std::vector<rknn_tensor_attr> &output_attrs,
                        std::vector<int32_t> &zps, 
                        std::vector<float> &scales);
    ~YoloxPostProcess() = default;

    /**
     * @brief 后处理
     * 
     * @param src 输出数据指针
     * @param results 结果保存
     * @param img_scale 原始图片resize scale
     */
    void process(int8_t *src, std::vector<ObjBox> &results, float img_scale);

private:
    std::vector<int> strides_{8, 16, 32};       // stride
    std::vector<GridAndStride> grid_strides_;   // grid_strides
    std::vector<ObjBox> out_boxes;  
    // std::vector<ObjBox> nms_boxes;

    int input_size_;    // 输入大小
    int num_grid_;      // grid数量
    int num_class_;     // 模型类别数
    int num_anchors_;   // anchors数量1

    int each_grid_long_;    // 步长

    float prob_threshold_;  // socre 阈值
    float nms_threshold_;   // nms 阈值

    int32_t zp_;    // 量化参数
    float scale_;   // 量化参数

};

