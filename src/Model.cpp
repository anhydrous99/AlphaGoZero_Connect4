//
// Created by aherrera on 3/28/20.
//

#include "Model.h"
#include <iostream>

ConvBlockImpl::ConvBlockImpl(int64_t input_channels, int64_t output_chennels, int64_t kernel_size, bool padding=true) {
    if (padding)
        conv2D = register_module("conv2D", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input_channels, output_chennels, kernel_size).padding(1)));
    else
        conv2D = register_module("conv2D", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input_channels, output_chennels, kernel_size)));
    batchNorm2D = register_module("batchNorm2D", torch::nn::BatchNorm2d(output_chennels));
}

torch::Tensor ConvBlockImpl::forward(const torch::Tensor &x) {
    return torch::leaky_relu(batchNorm2D(conv2D(x)));
}

ModelImpl::ModelImpl(const std::vector<int64_t> &input_shape, int64_t n_actions, int64_t num_filters) {
    convBlockIn = register_module("convBlockIn", ConvBlock(input_shape[0], num_filters, 3));
    convBlock1 = register_module("convBlock1", ConvBlock(num_filters, num_filters, 3));
    convBlock2 = register_module("convBlock2", ConvBlock(num_filters, num_filters, 3));
    convBlock3 = register_module("convBlock3", ConvBlock(num_filters, num_filters, 3));
    convBlock4 = register_module("convBlock4", ConvBlock(num_filters, num_filters, 3));
    convBlock5 = register_module("convBlock5", ConvBlock(num_filters, num_filters, 3));
    convValue = register_module("convValue", ConvBlock(num_filters, 1, 1, false));
    convPolicy = register_module("convPolicy", ConvBlock(num_filters, 2, 1, false));
    linearValue1 = register_module("linearValue1",
            torch::nn::Linear(torch::nn::LinearOptions(42, 20)));
    linearValue2 = register_module("linearValue2",
            torch::nn::Linear(torch::nn::LinearOptions(20, 1)));
    linearPolicy = register_module("linearPolicy",
            torch::nn::Linear(torch::nn::LinearOptions(84, n_actions)));
}

TensorPair ModelImpl::forward(torch::Tensor x) {
    x = convBlockIn(x);
    x = x + convBlock1(x);
    x = x + convBlock2(x);
    x = x + convBlock3(x);
    x = x + convBlock4(x);
    x = x + convBlock5(x);

    torch::Tensor value = convValue(x);
    value = value.view({value.size(0), -1});
    value = torch::leaky_relu(linearValue1(value));
    value = torch::tanh(linearValue2(value));
    torch::Tensor policy = convPolicy(x);
    policy = policy.view({value.size(0), -1});
    policy = linearPolicy(policy);
    return {policy, value};
}

void sync_weights(Model &output_model, const Model &input_model) {
    torch::autograd::GradMode::set_enabled(false);
    auto input_params = input_model->named_parameters(true);
    auto input_buffers = input_model->named_buffers(true);
    auto output_params = output_model->named_parameters(true);
    auto output_buffers = output_model->named_buffers(true);
    for (const auto& val : input_params) {
        const auto& name = val.key();
        auto* t = output_params.find(name);
        if (t != nullptr)
            t->copy_(val.value());
        else {
            t = output_buffers.find(name);
            if (t != nullptr)
                t->copy_(val.value());
        }
    }
    torch::autograd::GradMode::set_enabled(true);
}
