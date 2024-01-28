#include "apps/yolox/rknn_yolox.h"
#include "apps/yolox/rknn_yolox_thread.h"
#include "apps/yolov8-seg/rknn_yolov8_seg.h"

using namespace std;
using namespace FasterRKNN;

// https://github.com/PaddlePaddle/FastDeploy/blob/develop/fastdeploy/runtime/backends/rknpu2/rknpu2_backend.cc

void testv1() {
    string model_name = "../../1convert/yolox_relu_nodecode.rknn";
    string image_name = "../img/1.jpg";
    const float nms_threshold = 0.65;
    const float box_conf_threshold = 0.45;

    cv::Mat img = cv::imread(image_name);

    // init model
    auto model = create_infer_yolox(model_name, nms_threshold, box_conf_threshold);
    // infer once
    auto res = model->infer(img);
    for(auto a : res) {
        std::cout << "ans: " << a.x1 << " " << a.y1 << " " << a.x2 << " " << a.y2 << " " << a.score
                  << " " << a.category << std::endl;
        cv::rectangle(img, cv::Point(a.x1, a.y1), cv::Point(a.x2, a.y2), cv::Scalar(255, 0, 0, 255),
                      3);
        cv::putText(img, std::to_string(a.category), cv::Point(a.x1, a.y1 + 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::Mat img1 = cv::imread("../img/1.jpg");
    // infer once
    auto res1 = model->infer(img1);
    for(auto a : res1) {
        std::cout << "ans2: " << a.x1 << " " << a.y1 << " " << a.x2 << " " << a.y2 << " " << a.score
                  << " " << a.category << std::endl;
        cv::rectangle(img1, cv::Point(a.x1, a.y1), cv::Point(a.x2, a.y2),
                      cv::Scalar(255, 0, 0, 255), 3);
        cv::putText(img1, std::to_string(a.category), cv::Point(a.x1, a.y1 + 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        cv::imwrite("2.jpg", img1);
    }

    // warmup
    for(int i = 0; i < 50; ++i) {
        model->infer(img);
    }
    // benchmark
    int test_count = 1000;
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < test_count; ++i) {
        auto res = model->infer(img);
    }
    auto end = std::chrono::system_clock::now();
    float infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("运行 %d 次，平均耗时 %f ms\n", test_count, infer_time / (float)test_count);
}

void test_thread() {
    string model_name = "../../1convert/yolox_relu_nodecode.rknn";
    string image_name = "../img/1.jpg";
    const float nms_threshold = 0.65;
    const float box_conf_threshold = 0.45;

    cv::Mat img = cv::imread(image_name);

    // init model
    auto model = create_infer_yolox_thread(model_name, nms_threshold, box_conf_threshold);
    // infer once
    auto res = model->commit(img);
    auto ans = res.get();
    for(auto a : ans) {
        std::cout << "ans: " << a.x1 << " " << a.y1 << " " << a.x2 << " " << a.y2 << " " << a.score
                  << " " << a.category << std::endl;
        cv::rectangle(img, cv::Point(a.x1, a.y1), cv::Point(a.x2, a.y2), cv::Scalar(255, 0, 0, 255),
                      3);
        cv::putText(img, std::to_string(a.category), cv::Point(a.x1, a.y1 + 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    queue<shared_future<vector<ObjBox>>> res_q;  // 结果保存队列
    int max_queue = 2;  // 结果队列中最大数量。（可以理解成预加载图片数量）

    int test_count = 1000;  // 测试轮数
    auto start = std::chrono::system_clock::now();

    for(int i = 0; i < test_count; i++) {
        if(res_q.size() < (max_queue - 1)) {
            res_q.push(model->commit(img));
            continue;
        } else {
            res_q.push(model->commit(img));
        }
        auto res = res_q.front().get();
        res_q.pop();
    }
    while(!res_q.empty()) {
        auto res = res_q.front().get();
        res_q.pop();
    }

    auto end = std::chrono::system_clock::now();
    float infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("运行 %d 次，平均耗时 %f ms\n", test_count, infer_time / (float)test_count);
}

void test_v8() {
    string model_name = "/home/orangepi/zzx/zzx_rknn/4faster_infer/src/seg.rknn";
    const float nms_threshold = 0.3;
    const float box_conf_threshold = 0.3;
    // init model
    auto model = create_infer_yolov8seg(model_name, nms_threshold, box_conf_threshold);

    string image_name = "/home/orangepi/zzx/zzx_rknn/4faster_infer/img/10.jpg";
    cv::Mat img = cv::imread(image_name);
    auto res = model->infer(img);

    for(auto a : res.objs) {
        std::cout << "ans2: " << a.x1 << " " << a.y1 << " " << a.x2 << " " << a.y2 << " " << a.score
                  << " " << a.category << std::endl;
        cv::rectangle(img, cv::Point(a.x1, a.y1), cv::Point(a.x2, a.y2), cv::Scalar(255, 0, 0, 255),
                      3);
        cv::putText(img, std::to_string(a.category), cv::Point(a.x1, a.y1 + 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        
    }
    cv::imwrite("2.jpg", img);

    for (int j = 0; j < 720; j++)
        {
            for (int k = 0; k < 1280; k++)
            {
                int pixel_offset = 3 * (j * 1280 + k);
                if (res.segs[j * 1280 + k] != 0)
                {
                    img.data[pixel_offset + 0] = (unsigned char)255; // r
                    img.data[pixel_offset + 1] = (unsigned char)255; // g
                    img.data[pixel_offset + 2] = (unsigned char)255; // b
                }
            }
        }
        cv::imwrite("3.jpg", img);

}

int main() {
    testv1();
    // test_thread();

    test_v8();
    return 0;
}
