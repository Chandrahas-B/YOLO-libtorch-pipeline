#include <torch/torch.h>
#include <torch/data/datasets.h>
#include <torch/data/example.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#define IMG_SZ 416

class CSVImageDataset : public torch::data::Dataset<CSVImageDataset>
{
public:
    CSVImageDataset(std::string csv_file, std::string root_dir)
        : csv_file_(csv_file), root_dir_(root_dir)
    {
        std::ifstream file(csv_file);
        if (!file.is_open())
            throw std::runtime_error("Could not open CSV file: " + csv_file);

        std::string line;
        std::getline(file, line);

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string filename, arrname;
            std::getline(ss, filename, ',');
            std::getline(ss, arrname, ',');

            std::string image_path = root_dir_ + "/" + filename;
            std::string arr_path = root_dir_ + "/" + arrname;

            data_paths_.push_back(image_path);
            arr_paths_.push_back(arr_path);
            
        }

        file.close();
    }

    torch::data::Example<> get(size_t index) override {
        auto data_path = data_paths_.at(index);
        auto arr_path  = arr_paths_.at(index);

        cv::Mat image = cv::imread(data_path);
        // cv::Mat np_arr = cv::imread(arr_path);

        std::ifstream infile(arr_path);
        std::string line;
        std::vector<float> data;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                data.push_back(value);
            }
        }
        float* arr = data.data();
        // for (int i = 0; i < data.size(); ++i) {
        //    std::cout << arr[i] << " ";
        // }
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor tensor = torch::from_blob(arr, {1152}, options).clone();
        // std::array<int64_t, 4> shape = {8, 8, 1, 18};
        // std::vector<float> array_data(shape[0] * shape[1] * shape[2] * shape[3]);
        // std::copy(data.begin(), data.end(), array_data.begin());
        // // auto max = std::max_element(array_data.begin(), array_data.end());
        // std::cout << *data.data() << std::endl;
        // auto tensor = torch::from_blob(data.data(), shape, options);
        infile.close();

    // return tensor;

        torch::Tensor img_data = torch::from_blob(image.data, { IMG_SZ, IMG_SZ, 3 }, torch::kByte).permute({ 2, 0, 1 }) / 255;
        // tensor = tensor.view({8,8, 18});

        return {img_data, tensor};
    }

    torch::optional<size_t> size() const override
    {
        return data_paths_.size();
    }

private:
    std::vector<std::string> data_paths_;
    std::vector<std::string> arr_paths_;
    std::string csv_file_;
    std::string root_dir_;
};
