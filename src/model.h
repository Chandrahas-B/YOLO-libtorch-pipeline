#pragma once

#include <torch/torch.h>
#include <iostream>


struct ModelImpl : torch::nn::Module {

    ModelImpl()
    {
        register_module("cb1", cb1);
        register_module("cb2", cb2);
        register_module("fc1", fc1);
    }

    torch::nn::Sequential cb1 {
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1)), // (414, 414, 64)
        torch::nn::LeakyReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1)), // (412, 412, 128)
        torch::nn::LeakyReLU(),
        torch::nn::BatchNorm2d(128),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))   // (206, 206, 128)
    };

    torch::nn::Sequential cb2 {
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 64, 5).stride(1)), // (202, 202, 64)
        torch::nn::LeakyReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 3).stride(1)), // (200, 200, 32)
        torch::nn::LeakyReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),  // (100, 100, 32)
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 16, 5).stride(1)), // (96, 96, 16)
        torch::nn::LeakyReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)), // (96, 96, 16)
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 8, 3).stride(1).padding(1)), // (96, 96, 8)
        torch::nn::LeakyReLU(),
        torch::nn::BatchNorm2d(8),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)), // (48, 48, 8)
    };

    torch::nn::Sequential fc1 {
        torch::nn::Linear(24*24*8, 1152),
        torch::nn::Sigmoid()
        // torch::nn::LayerNorm(torch::nn::LayerNormOptions({4096})),
        // torch::nn::Linear(4096, 1152)
        // torch::nn::LayerNorm(torch::nn::LayerNormOptions({})),
        // torch::nn::LeakyReLU(),
        // torch::nn::Linear(256, 5)
    };

    

    torch::Tensor forward(torch::Tensor x) {
        // torch::autograd::DetectAnomalyGuard(true);
        x = cb1 -> forward(x);
        x = cb2 -> forward(x);
        x = x.view({-1, 24*24*8});
        x = fc1 -> forward(x);
        // x = x.view({-1, 8, 8, 18});
        return x;
    }
    
};

TORCH_MODULE(Model);
