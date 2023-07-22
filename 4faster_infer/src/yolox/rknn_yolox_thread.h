#pragma once

#include <memory>

#include "../base/rknn_infer_thread_base.h"  // 注意这个地方，需要包含实现文件。因为是模板类，仅包含头文件会链接错误。
#include "../postprocess/rknn_postprocess.h"


// resize
cv::Mat static_resize_for_thread(const cv::Mat& img, int input_w, int input_h) {
    float r = std::min(input_w / (img.cols*1.0), input_h / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

using RknnInferThreadBaseYolo = RknnInferThreadBase<std::vector<ObjBox>, float>;
class RknnYoloxThread :public RknnInferThreadBaseYolo {

public:
    RknnYoloxThread(const std::string &model_path, const float nms_threshold, const float conf_threshold) {
        Init(model_path);
        postprocess_ = std::make_shared<YoloxPostProcess>(input_h_, conf_threshold, nms_threshold, output_attrs_, out_zps_, out_scales_);

        startup();

    }
    // RknnYoloxThread(const float nms_threshold, const float conf_threshold) {
    //     nms_threshold_ = nms_threshold;
    //     conf_threshold_ = conf_threshold;
    // }
    virtual ~RknnYoloxThread() { printf("子类析构\n");};

    // void init_task_info() {
    //     postprocess_ = std::make_shared<YoloxPostProcess>(input_h_, conf_threshold_, nms_threshold_, output_attrs_, out_zps_, out_scales_);
    // }

    void worker() {

        while(get_job_and_wait()) {

            memcpy(input_mems_[0]->virt_addr, fetch_job_.input.data, input_attrs_[0].size_with_stride);
            CHECK_RKNN(rknn_run(ctx_, NULL));
            postprocess_->process((int8_t *)output_mems_[0]->virt_addr, results, fetch_job_.additional);
            fetch_job_.pro->set_value(results);
        }
        printf("worker over! \n");

    }

    bool preprocess(Job& job, const cv::Mat &img) {
        job.input = static_resize_for_thread(img, input_w_, input_h_);
        job.additional = std::min(input_w_ / (img.cols*1.0), input_h_ / (img.rows*1.0));

        return true;
    }



private:
    std::shared_ptr<YoloxPostProcess> postprocess_;
    std::vector<ObjBox> results;
    float nms_threshold_;
    float conf_threshold_;
};



std::shared_ptr<RknnInferThreadBaseYolo> create_infer_yolox_thread(const std::string &model_path, const float nms_threshold, const float conf_threshold) {

    return std::make_shared<RknnYoloxThread>(model_path, nms_threshold, conf_threshold);
}




