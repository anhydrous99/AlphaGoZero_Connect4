#include <iostream>
#include "utils.h"

int main() {
    Eigen::Matrix2f matrix2F = Eigen::Matrix2f::Random();
    std::cout << "Eigen: \n" << matrix2F << std::endl;
    auto tensor = toTensor(matrix2F);
    std::cout << "Torch::tensor: \n" << tensor << std::endl;
    Eigen::Matrix2f ret_mat = toEigen<Eigen::Matrix2f>(tensor);
    std::cout << "Returned Eigen matrix: \n" << ret_mat << std::endl;
    Eigen::Array22f ret_arr = toEigen<Eigen::Array22f>(tensor);
    std::cout << "Returned Eigen array: \n" << ret_arr << std::endl;
    return 0;
}
