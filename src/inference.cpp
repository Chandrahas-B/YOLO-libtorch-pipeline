#include <iostream>
#include <fstream>
#include <sstream>
#include<vector>
#include <string>
#include <filesystem>
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<ctime>
#include<chrono>
#include "model.h"
#define IMG_SZ 416
using namespace std;

torch::Tensor CVtoTensor(cv::Mat img) {
        cv::resize(img, img, cv::Size{IMG_SZ, IMG_SZ}, 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        auto img_tensor = torch::from_blob(img.data, {IMG_SZ, IMG_SZ, 3}, torch::kByte);
        img_tensor = img_tensor.permute({2, 0, 1}).toType(torch::kFloat).div_(255);
        return img_tensor;
    }
std::vector<float> TorchtoVec(torch::Tensor grid)
{
    auto sz = grid.sizes()[0];
    std::vector<float> vec(sz); 
    for(auto i = 0; i < sz; i++)
        vec[i] = grid[i].item<float>();
    
    return vec;
}

int main() {
    std::string pieces[13];
    pieces[0] = "bishop";
    pieces[1] = "b-bishop";
    pieces[2] = "b-king";
    pieces[3] = "b-knight";
    pieces[4] = "b-pawn";
    pieces[5] = "b-queen";
    pieces[6] = "b-rook";
    pieces[7] = "w-bishop";
    pieces[8] = "w-king";
    pieces[9] = "w-knight";
    pieces[10] = "w-pawn";
    pieces[11] = "w-queen";
    pieces[12] = "w-rook";

	int box_sz = 50;
	std::string model_path;  // Path for the model which has to be used.
    std::cout << "Enter the model path: " << std::endl;

	auto cuda = torch::cuda::is_available();
    torch::Device device = cuda ? torch::kCUDA : torch::kCPU;
	std::cout << "Device type: " << device << std::endl <<std::endl;
	// std::cin >> model_path;
    model_path = "MainModel.pt";

    while(true)
    {

    std::string image_path;  // Path for the image location for predicting. Ex: /home/chandrahas/SandLogic/CustomYOLO/YOLOv3format/train/47e842dd95735a11cf92c0ddf1161193_jpg.rf.60c38d132c7d19cb8454be79650d53a5.jpg
    std::cout << std::endl << std::endl << "Enter the image path: " << std::endl;
    std::cin >> image_path;
    if(image_path == "0")
        break;
    // image_path = "/home/chandrahas/SandLogic/CustomYOLO/YOLOv3format/test/IMG_0159_JPG.rf.f0d34122f8817d538e396b04f2b70d33.jpg";
    // image_path = "/home/chandrahas/SandLogic/CustomYOLO/YOLOv3format/train/47e842dd95735a11cf92c0ddf1161193_jpg.rf.60c38d132c7d19cb8454be79650d53a5.jpg";
    // image_path = "/home/chandrahas/SandLogic/CustomYOLO/YOLOv3format/test/cfc306bf86176b92ffc1afbb98d7896f_jpg.rf.effd71a5dcd98ec0f24072af5f7c0a31.jpg";
    // image_path = "/home/chandrahas/SandLogic/CustomYOLO/YOLOv3format/test/2f6fb003bb89cd401322a535acb42f65_jpg.rf.66c0a46773a9cd583fb96c3df41a9e0c.jpg";
    std::cout << std::endl;
    size_t pos = image_path.find_last_of("/\\");
    std::string name = (pos == std::string::npos) ? image_path : image_path.substr(pos+1);
    auto cuda = torch::cuda::is_available();
   cv::Mat img;
    torch::Tensor img_tensor;
    Model model;

    try {
    img = cv::imread(image_path);
    img_tensor = CVtoTensor(img);
    cv::resize(img, img, cv::Size{IMG_SZ, IMG_SZ}, 0, 0, cv::INTER_LINEAR);
    }
    catch (...) {
        std::cout << "\nError in reading the image. Please enter the correct path.\n\n";
        return 0;
    };
    img_tensor = img_tensor.unsqueeze(0);

    try {
    model = Model();
    torch::load(model, model_path);
    }
    catch (...) {
        std::cout<< "\nError in loading the model. Please enter the correct path.\n\n";
        return 0;
    };

    img_tensor = img_tensor.to(device);
    model->to(device);
    // time_t t1 = time(nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();

    // std::cout << model << std::endl;
    auto output = model->forward(img_tensor);
    output = output.reshape({8,8,18});
    // cout<< output << endl;
    // std::cout << "max: " << torch::max(output) << std::endl;
    int i = 1;
    int x = 0, y = 0;
    for(int r = 0; r < 8; r++)
    {
        for(int c = 0; c < 8; c++)
        {
            auto grid = output[r][c];
            auto bbox = grid.slice(0,0,5);
            auto bbx = TorchtoVec(bbox);
            bbx[0] = (bbx[0]+r)*52;
            bbx[1] = (bbx[1]+c)*52;
            bbx[2] = bbx[2]*IMG_SZ;
            bbx[3] = bbx[3]*IMG_SZ;
            
            // 5.000000000000000000e-01
            // 9.519230769230770939e-01
            // 6.730769230769230449e-02
            // 1.995192307692307820e-01
            bbx[2] = (bbx[2]<box_sz)?bbx[2]:box_sz;
            bbx[3] = (bbx[3]<box_sz+15)?bbx[3]:box_sz+15;
            bbx[4] = std::round(bbx[4]);
            // std::cout << "Bbox: " << bbox << std::endl;
            if (bbx[4] < 0.5)
                continue;
            // std::cout << int(0.5*52 + x) << std::endl;
            // std::cout << IMG_SZ/2 - round(9.519230769230770939e-01*52 +y) << std::endl;
            // std::cout << int(6.730769230769230449e-02 * IMG_SZ) << std::endl;
            // std::cout << round(1.995192307692307820e-01 * IMG_SZ) << std::endl;
            auto cls = grid.slice(0, 5, 18);
            auto probs = torch::nn::functional::softmax(cls, 0);
            auto class_id = torch::argmax(cls).item<int>();
            auto prob = probs[class_id].item<double>();
            // std::cout << class_id << std::endl;
            // std::cout << bbx << std::endl;
            
            int xmin = round(bbx[0] - bbx[2]/2);
            int ymin = round(bbx[1] - bbx[3]/2);
            int xmax = round(bbx[0] + bbx[2]/2);
            int ymax = round(bbx[1] + bbx[3]/2);
            // std::cout << grid.slice(0, 5, 18) << std::endl;
            // std::cout << "Row: " << r << "\tCol: " << c << std::endl;
            std::cout << i << ". " << pieces[class_id] << std::endl;
            cv::Scalar color(0, 0, 255);
            if (class_id > 6)
                color = cv::Scalar(255, 0, 0);

            
            cv::putText(img, std::to_string(i), cv::Point(xmin, ymin - 4), cv::FONT_HERSHEY_TRIPLEX, 1, color, 2, cv::LINE_AA);
            cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), color, 2, cv::LINE_AA);

            y += 52;
            i += 1;
        }
        x += 52;
        y = 0;
        
    }
    // time_t t2 = time(nullptr);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    std::cout <<"Time taken: " << diff.count() << "ms" << std::endl;
    // std::cout << "No bbox: " << i << std::endl;
    cv::namedWindow("ObjectDetection", cv::WINDOW_NORMAL);
    cv::resizeWindow("ObjectDetection", cv::Size{IMG_SZ, IMG_SZ});
    // cv::setWindowProperty("ObjectDetection", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    cv::imshow("ObjectDetection", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    }
    
    return 0;
}
