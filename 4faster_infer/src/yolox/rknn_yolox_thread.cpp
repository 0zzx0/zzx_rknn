/**
 * @file rknn_yolox_thread.cpp
 * @author zzx
 * @brief yolox多线程的实现
 * @version 0.1
 * @date 2023-07-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include "rknn_yolox_thread.h"


RknnYoloxThread::RknnYoloxThread(const std::string &model_path, const float nms_threshold, const float conf_threshold) {

    Init(model_path);
    postprocess_ = std::make_shared<YoloxPostProcess>(input_h_, conf_threshold, nms_threshold, output_attrs_, out_zps_, out_scales_);

    startup();

}
    
RknnYoloxThread::~RknnYoloxThread() { 

    printf("子类析构\n");
}


void RknnYoloxThread::worker() {

    while(get_job_and_wait()) {

        memcpy(input_mems_[0]->virt_addr, fetch_job_.input.data, input_attrs_[0].size_with_stride);
        CHECK_RKNN(rknn_run(ctx_, NULL));
        postprocess_->process((int8_t *)output_mems_[0]->virt_addr, results, fetch_job_.additional);
        fetch_job_.pro->set_value(results);
    }
    printf("worker over! \n");

}


bool RknnYoloxThread::preprocess(Job& job, const cv::Mat &img) {

    job.input = static_resize(img, input_w_, input_h_);
    job.additional = std::min(input_w_ / (img.cols*1.0), input_h_ / (img.rows*1.0));

    return true;
}




std::shared_ptr<RknnInferThreadBaseYolo> create_infer_yolox_thread(const std::string &model_path, const float nms_threshold, const float conf_threshold) {

    return std::make_shared<RknnYoloxThread>(model_path, nms_threshold, conf_threshold);
}

