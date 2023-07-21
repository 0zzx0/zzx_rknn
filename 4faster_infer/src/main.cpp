#include "yolox/rknn_yolox.h"

using namespace std;

// https://github.com/PaddlePaddle/FastDeploy/blob/develop/fastdeploy/runtime/backends/rknpu2/rknpu2_backend.cc

int main() {
    string model_name = "../../1convert/yolox_relu_nodecode.rknn";
	string image_name = "../img/1.jpg";
	const float nms_threshold      = 0.65;
	const float box_conf_threshold = 0.45;

    cv::Mat img = cv::imread(image_name);
    auto model = create_infer_yolox(model_name, nms_threshold, box_conf_threshold);


    auto res = model->infer(img);
    for (auto a : res) {
		std::cout<<"ans: "<<a.x1<<" "<<a.y1<<" "<<a.x2<<" "<<a.y2 <<" "<<a.score<<" "<<a.category<< std::endl; 
		cv::rectangle(img, cv::Point(a.x1, a.y1), cv::Point(a.x2, a.y2), cv::Scalar(255, 0, 0, 255), 3);
		cv::putText(img, std::to_string(a.category), cv::Point(a.x1, a.y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}


	for (int i = 0; i < 50; ++i) {
		model->infer(img);
	}

    int test_count = 1000;
    auto start = std::chrono::system_clock::now();
	for (int i = 0; i < test_count; ++i) {
		auto res = model->infer(img);
	}
	auto end = std::chrono::system_clock::now();
	float infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() ;
	printf("运行 %d 次，平均耗时 %f ms\n", test_count, infer_time / (float)test_count);

    return 0;
}