#pragma once
#include <iostream>
#include <opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>

#include <torch/torch.h>
#include <torch/script.h>
#include <memory>

using namespace cv;
using namespace std;

int main()
{
	Mat image;
	image = imread("img.jpg", IMREAD_GRAYSCALE);//读取灰度图

	torch::jit::script::Module module;
	try {
		module = torch::jit::load("model.pt");  //加载模型
	}
	catch (const c10::Error & e) {
		std::cerr << "无法加载model.pt模型\n";
		return -1;
	}

	std::vector<int64_t> sizes = { 1, 1, image.rows, image.cols };  //依次为batchsize、通道数、图像高度、图像宽度
	at::TensorOptions options(at::ScalarType::Byte);
	at::Tensor tensor_image = torch::from_blob(image.data, at::IntList(sizes), options);//将opencv的图像数据转为Tensor张量数据
	tensor_image = tensor_image.toType(at::kFloat);//转为浮点型张量数据
	at::Tensor result = module.forward({ tensor_image }).toTensor();//推理

	auto max_result = result.max(1, true);
	auto max_index = std::get<1>(max_result).item<float>();
	std::cerr << "检测结果为：";
	std::cout << max_index << std::endl;

	waitKey(6000);
}