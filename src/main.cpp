#include <iostream>
#include "utils.h"

int main() {
    Eigen::Matrix2f matrix2F = Eigen::Matrix2f::Random();
    std::cout << "Eigen: \n" << matrix2F << std::endl;
    auto tensor = toTensor(matrix2F);
    std::cout << "Torch::tensor: \n" << tensor << std::endl;
    Eigen::Matrix2f ret_mat = toEigenMatrix<float, 2, 2>(tensor);
    std::cout << "Return Eigen Matrix: \n" << ret_mat << std::endl;
    Eigen::Array22f ret_arr = toEigenArray<float, 2, 2>(tensor);
    std::cout << "Return Eigen Array: \n" << ret_arr << std::endl;

    Eigen::Array2f array2F = Eigen::Array2f::Random();
    std::cout << "Eigen: \n" << array2F << std::endl;
    auto tensor1 = toTensor(array2F);
    std::cout << "Torch::tensor: \n" << tensor1 << std::endl;
    Eigen::Matrix2Xf ret_mat1 = toEigenMatrix<float, 2, 1>(tensor1);
    std::cout << "Return Eigen Matrix: \n" << ret_mat1 << std::endl;
    return 0;
}
