#include <iostream>
#include <memory>
#include <chrono>

#include "detector.h"
#include "utils.h"

using namespace std;
using namespace cv;

std::vector<std::string> LoadNames(const std::string& path) {
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open()) {
        std::string line;
        while (getline (infile,line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

    return class_names;
}


void Demo(cv::Mat& img,
        const std::vector<std::tuple<cv::Rect, float, int>>& data_vec,
        const std::vector<std::string>& class_names,
        bool label = true) {
    for (const auto & data : data_vec) {
        cv::Rect box;
        float score;
        int class_idx;
        std::tie(box, score, class_idx) = data;

        cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

        if (label) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << score;
            std::string s = class_names[class_idx] + " " + ss.str();

            auto font_face = cv::FONT_HERSHEY_DUPLEX;
            auto font_scale = 1.0;
            int thickness = 1;
            int baseline=0;
            auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
            cv::rectangle(img,
                    cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                    cv::Point(box.tl().x + s_size.width, box.tl().y),
                    cv::Scalar(0, 0, 255), -1);
            cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                        font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
        }
    }

}


int main(int argc, const char* argv[]) {
    /*if (argc != 3 && argc != 4) {
        std::cerr << "usage: app <path-to-exported-script-module> <path-to-image> <-gpu>\n";
        std::cerr << "Example: app xxx.pt xxx.jpg -gpu\n";
        return -1;
    }*/
  
    string sModePath = "./models/best002.torchscript.pt";
    // check if gpu flag is set
    bool is_gpu = false;
    if(is_gpu==true) {
        is_gpu = "-gpu";
    }
 
    // set device type - CPU/GPU
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && is_gpu) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    // load class names from dataset for visualization
    std::vector<std::string> class_names = LoadNames("./weights/coco.names");
    if (class_names.empty()) {
        return -1;
    }
    cv::String imagefolder = "./images/";
    std::vector<string>imagenames;
    cv::glob(imagefolder, imagenames, false);

    for (int idx = 0; idx < imagenames.size(); idx++)
    {
        string sFileName = imagenames[idx];
        // load input image
        cv::Mat img = cv::imread(sFileName);
        if (img.empty()) {
            std::cerr << "Error loading the image!\n";
            return -1;
        }

        // load network
        auto detector = Detector(sModePath, device_type);
        // inference
        auto result = detector.Run(img, kConfThreshold, kIouThreshold);

        // visualize detections
        Demo(img, result, class_names);
        string soutFile = "./Result/" + to_string(idx) + ".jpg";

        imwrite(soutFile, img);
    }
    double start = static_cast<double>(getTickCount());
    double time = ((double)getTickCount() - start) / getTickFrequency();
    cout << "所用时间为：" << time << "秒" << endl;
    
  
}