//
// Created by aherrera on 3/28/20.
//

#ifndef ALPHAZERO_CONNECT4_MODEL_H
#define ALPHAZERO_CONNECT4_MODEL_H

#include <torch/torch.h>
#include "utils.h"

struct ConvBlockImpl : torch::nn::Module {
    ConvBlockImpl(int64_t input_channels, int64_t output_chennels, int64_t kernel_size, bool padding);

    torch::Tensor forward(const torch::Tensor &x);

    torch::nn::Conv2d conv2D{nullptr};
    torch::nn::BatchNorm2d batchNorm2D{nullptr};
};
TORCH_MODULE(ConvBlock);

struct ModelImpl : torch::nn::Module {
    ModelImpl(const std::vector<int64_t> &input_shape, int64_t n_actions, int64_t num_filters);
    TensorPair forward(torch::Tensor x);

    ConvBlock convBlockIn{nullptr}, convBlock1{nullptr}, convBlock2{nullptr}, convBlock3{nullptr},
    convBlock4{nullptr}, convBlock5{nullptr}, convValue{nullptr}, convPolicy{nullptr};
    torch::nn::Linear linearValue1{nullptr}, linearValue2{nullptr}, linearPolicy{nullptr};
};
TORCH_MODULE(Model);

#endif //ALPHAZERO_CONNECT4_MODEL_H
