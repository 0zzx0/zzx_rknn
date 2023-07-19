#include <string>
#include <vector>
#include <chrono>

#include "rga.h"
#include "im2d.h"
#include "rknn_api.h"
#include "opencv2/opencv.hpp"

#include "tools.hpp"
#include "postprocess.hpp"

// 打印信息
static void dump_tensor_attr(rknn_tensor_attr* attr){

	std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
	for (int i = 1; i < attr->n_dims; ++i) {
		shape_str += ", " + std::to_string(attr->dims[i]);
	}

	printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
			"type=%s, qnt_type=%s, "
			"zp=%d, scale=%f\n",
			attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
			attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
			get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


cv::Mat static_resize(cv::Mat& img, int input_w, int input_h) {
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


int main(){

	std::string model_name = "../../1convert/yolox_relu_nodecode.rknn";
	std::string image_name = "../img/1.jpg";
	const float nms_threshold      = 0.65;
	const float box_conf_threshold = 0.45;

	rknn_context   ctx;
	std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;

	// 反量化参数
	std::vector<float>    out_scales;
	std::vector<int32_t>  out_zps;

	// 加载文件
	int model_data_size = 0;
	unsigned char* model_data = load_model(model_name.c_str(), &model_data_size);

	// 初始化
	CHECK_RKNN(rknn_init(&ctx, model_data, model_data_size, 0, NULL));
	
	// 获取&打印版本信息
	rknn_sdk_version version;
	CHECK_RKNN(rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version)));
	printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

	// 获取&打印输入输出数量
	rknn_input_output_num io_num;
	CHECK_RKNN(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)));
	printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

	// 获取&打印input信息
	input_attrs.resize(io_num.n_input);
	// memset(input_attrs, 0, sizeof(input_attrs));
	for (int i = 0; i < io_num.n_input; i++) {
		input_attrs[i].index = i;
		CHECK_RKNN(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr)));
		printf("input information: ");
		dump_tensor_attr(&(input_attrs[i]));
	}
	// 获取&打印output信息
	output_attrs.resize(io_num.n_output);
	// memset(output_attrs, 0, sizeof(output_attrs));
	for (int i = 0; i < io_num.n_output; i++) {
		output_attrs[i].index = i;
		CHECK_RKNN(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr)));
		printf("output information:");
		dump_tensor_attr(&(output_attrs[i]));
	}

	int channel = 3;
	int width   = 0;
	int height  = 0;
	if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
		printf("model is NCHW input fmt\n");
		channel = input_attrs[0].dims[1];
		height  = input_attrs[0].dims[2];
		width   = input_attrs[0].dims[3];
	} else {
		printf("model is NHWC input fmt\n");
		height  = input_attrs[0].dims[1];
		width   = input_attrs[0].dims[2];
		channel = input_attrs[0].dims[3];
	}

	
	// 初始化后处理类
	for (int i = 0; i < io_num.n_output; ++i) {
		out_scales.push_back(output_attrs[i].scale);
		out_zps.push_back(output_attrs[i].zp);
	}
	std::shared_ptr<YoloxPostProcess> post_process = std::make_shared<YoloxPostProcess>(height, box_conf_threshold, nms_threshold, output_attrs);

	// 读取图片
	cv::Mat img = cv::imread(image_name, 1);
	// cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);

	// 预处理
	float scale = std::min(width / (img.cols*1.0), height / (img.rows*1.0));
	auto img_out = static_resize(img, width, height);

	// 输出内存
	rknn_tensor_mem* input_mems[1];
	rknn_tensor_mem* output_mems[1];
	input_mems[0]   = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    output_mems[0]  = rknn_create_mem(ctx, output_attrs[0].n_elems * sizeof(int8_t));

	memcpy(input_mems[0]->virt_addr, img_out.data, input_attrs[0].size_with_stride);

	input_attrs[0].type = RKNN_TENSOR_UINT8;
	output_attrs[0].type = RKNN_TENSOR_INT8;


	CHECK_RKNN(rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]));
	CHECK_RKNN(rknn_set_io_mem(ctx, output_mems[0], &output_attrs[0]));

	// 推理
	CHECK_RKNN(rknn_run(ctx, NULL));


	// 后处理
	auto res = post_process->process((int8_t *)output_mems[0]->virt_addr, out_zps, out_scales);
	
	// 打印结果
	printf("res size: %ld\n", res.size());
	for (auto a : res) {
		std::cout << scale << std::endl;
		a.x1 /= scale;
		a.y1 /= scale;
		a.x2 /= scale;
		a.y2 /= scale;

		std::cout<<a.x1<<" "<<a.y1<<" "<<a.x2<<" "<<a.y2 <<" "<<a.score<<" "<<a.category<< std::endl; 
		cv::rectangle(img, cv::Point(a.x1, a.y1), cv::Point(a.x2, a.y2), cv::Scalar(255, 0, 0, 255), 3);
		cv::putText(img, std::to_string(a.category), cv::Point(a.x1, a.y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
	cv::imwrite("./out.jpg", img);

	// 测速
	int test_count = 1000;
	// warmup
	for (int i = 0; i < 50; ++i) {
		auto img_out = static_resize(img, width, height);
		memcpy(input_mems[0]->virt_addr, img_out.data, input_attrs[0].size_with_stride);
		CHECK_RKNN(rknn_run(ctx, NULL));
		auto res = post_process->process((int8_t *)output_mems[0]->virt_addr, out_zps, out_scales);

	}
	
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < test_count; ++i) {
		auto img_out = static_resize(img, width, height);
		memcpy(input_mems[0]->virt_addr, img_out.data, input_attrs[0].size_with_stride);
		CHECK_RKNN(rknn_run(ctx, NULL));
		auto res = post_process->process((int8_t *)output_mems[0]->virt_addr, out_zps, out_scales);

	}

	auto end = std::chrono::system_clock::now();
	float infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() ;
	printf("运行 %d 次，平均耗时 %f ms\n", test_count, infer_time / (float)test_count);


	// release
	
	CHECK_RKNN(rknn_destroy_mem(ctx, input_mems[0]));
	CHECK_RKNN(rknn_destroy_mem(ctx, output_mems[0]));
	CHECK_RKNN(rknn_destroy(ctx));

	if (model_data) {
		free(model_data);
	}
	return 0;
}
