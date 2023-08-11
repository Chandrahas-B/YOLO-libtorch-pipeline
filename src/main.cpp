#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "model.h"
#include "dataloader.h"
// #include "NumCpp.hpp"
// #include "../cnpy.h"
using namespace torch;

namespace {
    const int trainSamples = 1;
}

int main() {
    
    std::string root = "/home/chandrahas/SandLogic/CustomYOLO/YOLOv3format/train";
    std::string csv_path = "/home/chandrahas/SandLogic/CustomYOLO/Train.csv";
    int batch_size = 4;
    size_t epochs = 50;
    double learning_rate = 0.001;

    auto cuda = torch::cuda::is_available();
    torch::Device device = cuda ? torch::kCUDA : torch::kCPU;

    std::cout << "Device type: \n" << device << std::endl;

    auto model = Model();
    model->to(device);
    std::cout << "\n\t\tModel architecture:" << std::endl;
    std::cout << model << std::endl << std::endl;

    auto train_dataset = CSVImageDataset(csv_path, root).map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);

    torch::optim::RMSprop optimizer(model->parameters(), torch::optim::RMSpropOptions({learning_rate}));
    
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
            auto bbox_labels = labels.slice(1, 0, 4);
            torch::Tensor obj_class_labels = labels.slice(1, 4, 18);
            auto output = model->forward(imgs);

            // auto loss = torch::nn::functional::mse_loss(output, labels);
            // torch::nn::MSELoss mse_loss;
            // torch::Tensor loss = mse_loss(output, labels);


            torch::nn::MSELoss mse_loss;
            torch::nn::CrossEntropyLoss ce_loss;
            torch::Tensor bbox_output = output.slice(1, 0, 4);
            torch::Tensor bbox_loss = mse_loss(bbox_output, bbox_labels);
            torch::Tensor obj_class_output = output.slice(1, 4, 18);
            torch::Tensor obj_sig =  torch::sigmoid(obj_class_output);
            // torch::Tensor obj_sig = obj_class_output;
            // std::cout << "Obj pred: " << std::endl << obj_sig << std::endl;
            // std::cout << "Obj bbox: " << std::endl << bbox_output << std::endl;
            torch::Tensor obj_score_loss = ce_loss(obj_sig, obj_class_labels);
            //auto loss_bound = torch::nn::functional::mse_loss(output[1], );
            //
            auto alpha = 0.5;
            auto loss = alpha * bbox_loss + (1 - alpha) * obj_score_loss;
            epoch_train_loss = epoch_train_loss + loss.item<double>() * imgs.size(0);
            // torch::nn::BCELoss bce_loss;
            // torch::Tensor bbox_output = output.slice(1, 0, 4);
            // torch::Tensor bbox_loss = mse_loss(bbox_output, bbox_labels);
            // torch::Tensor obj_score_output = output.slice(1, 4, 5);
            // torch::Tensor obj_sig =  torch::sigmoid(obj_score_output) ;
            // std::cout << "Obj pred: " << std::endl << obj_sig << std::endl;
            // std::cout << "Obj bbox: " << std::endl << bbox_output << std::endl;
            // torch::Tensor obj_score_loss = bce_loss(obj_sig, obj_score_labels);
            //auto loss_bound = torch::nn::functional::mse_loss(output[1], );
            //
            // auto alpha = 0.75;
            // auto loss = alpha * bbox_loss + (1 - alpha) * obj_score_loss;
            // std::cout << output << std::endl;
            epoch_train_loss = epoch_train_loss + loss.item<double>() * imgs.size(0);
            optimizer.zero_grad();
            loss.backward();
            // torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();
        }
        epoch_train_loss = epoch_train_loss / trainSamples;
        //auto train_accuracy = static_cast<double>(correct) / trainSamples;
        std::cout << "\t" << "\nTrain Loss: " << epoch_train_loss << std::endl;
        
    }
    // define the path and file name where you want to save the model
    std::string model_path = "model.pt";

    // serialize and save the model
    torch::save(model, model_path);


    return 0;
}
