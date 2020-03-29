#include <iostream>
#include "Model.h"

int main() {
    std::vector<int64_t> size{2, 6, 7};
    torch::Tensor tensor = torch::rand(size);
    tensor = torch::unsqueeze(tensor, 0);
    Model model(size, 7, 64);
    TensorPair pair = model->forward(tensor);
    std::cout << pair.tensor1 << std::endl;
    std::cout << pair.tensor2 << std::endl;
    return 0;
}
