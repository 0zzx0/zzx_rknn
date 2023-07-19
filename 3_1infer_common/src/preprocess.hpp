#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#include <opencv2/opencv.hpp>
#include <string.h>
#include "rga.h"
#include "im2d.h"

#include "tools.h"

typedef int(* FUNC_RGA_INIT)();
typedef void(* FUNC_RGA_DEINIT)();
typedef int(* FUNC_RGA_BLIT)(rga_info_t *, rga_info_t *, rga_info_t *);

class RGAPreProcess{

public:
    RGAPreProcess(int w, int h){
        want_h = h;
        want_w = w;

        memset(&src_rect, 0, sizeof(src_rect));
        memset(&dst_rect, 0, sizeof(dst_rect));
        memset(&src, 0, sizeof(src));
        memset(&dst, 0, sizeof(dst));
    }
    ~RGAPreProcess() {};


    void letter_img(cv::Mat &img, void *resize_buf, int src_w, int src_h, int channel, bool debug = false) {

        memset(resize_buf, 0x00, want_h * want_w * channel);

        src = wrapbuffer_virtualaddr((void*)img.data, src_w, src_h, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void*)resize_buf, want_w, want_h, RK_FORMAT_RGB_888);
        CHECK_RKNN_RGA(imcheck(src, dst, src_rect, dst_rect));
        CHECK_RKNN_RGA(imresize(src, dst));

        // for debug
        if (debug) {
            cv::Mat resize_img(cv::Size(want_w, want_h), CV_8UC3, resize_buf);
            cv::imwrite("debug_rga_resize.jpg", resize_img);
        }
    }


private:
    // 初始化rga
	rga_buffer_t src;
	rga_buffer_t dst;
	im_rect      src_rect;
	im_rect      dst_rect;

    int want_w;
    int want_h;
	
};









#endif