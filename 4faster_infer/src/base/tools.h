#ifndef TOOLS_H
#define TOOLS_H

#include "rga.h"


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




#endif