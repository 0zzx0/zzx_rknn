#pragma once
/**
 * @file tools.hpp
 * @author zzx
 * @brief 一些必备基础功能函数
 * @version 0.1
 * @date 2023-07-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "rga.h"

/**
 * @brief 通过检查RKNN部分api返回值，判断是否正常运行
 * 
 */
#define CHECK_RKNN(call)                    \
do                                          \
{                                           \
    const int error_code = call;            \
    if (error_code < 0)                     \
    {                                                 \
        printf("RKNN Error: %d\n", error_code);       \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        exit(1);                                      \
    }                                                 \
} while (0)


/**
 * @brief 通过检查RGA部分api返回值，判断是否正常运行
 * 
 */
#define CHECK_RKNN_RGA(call)                \
do                                          \
{                                           \
    const int error_code = call;            \
    if (error_code != IM_STATUS_NOERROR &&  error_code != IM_STATUS_SUCCESS)    \
    {                                       \
        printf("RKNN RGA Error: %s\n", imStrError((IM_STATUS)error_code));  \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        exit(1);                                      \
    }                                                 \
} while (0)



/**
 * @brief 读取文件信息
 * 
 * @param fp 文件头地址
 * @param ofst 偏移量
 * @param sz 数据大小
 * @return unsigned char* 返回数据
 */
static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz){
    unsigned char* data;
    int ret;

    data = NULL;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char*)malloc(sz);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}


/**
 * @brief 
 * 
 * @param filename 文件路径
 * @param model_size 模型大小
 * @return unsigned char* 模型数据指针
 */
static unsigned char* load_model(const char* filename, int* model_size) {

    FILE*  fp;
    unsigned char* data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}



