#include "yolox/rknn_yolox.h"

using namespace std;


int main() {
    string model_name = "../../1convert/yolox_relu_nodecode.rknn";
	string image_name = "../img/1.jpg";
	const float nms_threshold      = 0.65;
	const float box_conf_threshold = 0.45;

    auto img = cv::imread(image_name);

    auto model = create_infer(model_name, nms_threshold, box_conf_threshold);
    model->infer(img);


}