#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "model.h"
#include "dataloader.h"

using namespace torch;

namespace {
    const int trainSamples = 1;
}

int main() {
    
    std::string root = "/home/chandrahas/SandLogic/CustomYOLO/YOLOv3format/train";
    std::string csv_path = "/home/chandrahas/SandLogic/CustomYOLO/Train.csv";
    int batch_size = 2;
    size_t epochs = 40;
    auto learning_rate = 0.0000000000001;
    std::string model_path = "/home/chandrahas/SandLogic/CustomYOLO/build/MainModel.pt";
    auto cuda = torch::cuda::is_available();
    torch::Device device = cuda ? torch::kCUDA : torch::kCPU;

    std::cout << "Device type: \n" << device << std::endl;

    auto model = Model();
    try {
    torch::load(model, model_path);
    }
    catch (...) {
        std::cout<< "\nError in loading the model. Please enter the correct path.\n\n";
        return 0;
    };

    model->to(device);
    // std::cout << "\n\t\tModel architecture:" << std::endl;
    // std::cout << model << std::endl << std::endl;

    auto train_dataset = CSVImageDataset(csv_path, root).map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions({learning_rate}));
    
    std::cout << std::fixed << std::setprecision(4);

    for (int i = 0; i < epochs; i++)
    {
        double epoch_train_loss = 0.0;
        size_t correct = 0;
        std::cout << "------------------------------------------------------------------------------------------------------------" << std::endl << std::endl;
        std::cout << "Epoch " << (i + 1) << "/" << epochs << std::endl;

        for (auto& batch : *train_loader) {
            auto imgs = batch.data.to(device);
            auto labels = batch.target.to(device);
            // std::cout << torch::max(labels) << std::endl;
            // std::cout << "batch: " << std::endl;
            // std::cout << labels << std::endl;
            // auto bbox_labels = labels.slice(1, 0, 4);
            // torch::Tensor obj_score_labels = labels.slice(1, 4, 5);
            auto output = model->forward(imgs);

            torch::nn::MSELoss mse_loss;
            torch::Tensor loss = mse_loss(output, labels);

            epoch_train_loss = epoch_train_loss + loss.item<double>() * imgs.size(0);
            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();
        }
        epoch_train_loss = epoch_train_loss / trainSamples;
        //auto train_accuracy = static_cast<double>(correct) / trainSamples;
        std::cout << "\t" << "\nTrain Loss: " << epoch_train_loss* 1000 << std::endl;
        
    }
    // define the path and file name where you want to save the model
    std::string save_model_path = "MainModel.pt";

    // serialize and save the model
    torch::save(model, save_model_path);


    return 0;
}
