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
	image = imread("img.jpg", IMREAD_GRAYSCALE);//��ȡ�Ҷ�ͼ

	torch::jit::script::Module module;
	try {
		module = torch::jit::load("model.pt");  //����ģ��
	}
	catch (const c10::Error & e) {
		std::cerr << "�޷�����model.ptģ��\n";
		return -1;
	}

	std::vector<int64_t> sizes = { 1, 1, image.rows, image.cols };  //����Ϊbatchsize��ͨ������ͼ��߶ȡ�ͼ����
	at::TensorOptions options(at::ScalarType::Byte);
	at::Tensor tensor_image = torch::from_blob(image.data, at::IntList(sizes), options);//��opencv��ͼ������תΪTensor��������
	tensor_image = tensor_image.toType(at::kFloat);//תΪ��������������
	at::Tensor result = module.forward({ tensor_image }).toTensor();//����

	auto max_result = result.max(1, true);
	auto max_index = std::get<1>(max_result).item<float>();
	std::cerr << "�����Ϊ��";
	std::cout << max_index << std::endl;

	waitKey(6000);
}