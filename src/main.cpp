#include <iostream>
#include "GameBoard.h"
#include "Model.h"
#include "MCTS.h"

int main() {
    std::vector<int64_t> input_shape{2, 6, 7};
    torch::Tensor input_tensor = torch::rand(input_shape);
    input_tensor = torch::unsqueeze(input_tensor, 0);
    Model model1(input_shape, 7, 64);
    Model model2(input_shape, 7, 64);

    TensorPair pair1 = model1(input_tensor);
    TensorPair pair2 = model2(input_tensor);

    std::cout << "Tensor 1: \n" << pair1.tensor1 << std::endl;
    std::cout << "Tensor 2: \n" << pair2.tensor1 << std::endl;

    sync_weights(model1, model2);

    pair1 = model1(input_tensor);
    pair2 = model2(input_tensor);

    std::cout << "Tensor 1: \n" << pair1.tensor1 << std::endl;
    std::cout << "Tensor 2: \n" << pair2.tensor1 << std::endl;
    return 0;
}
