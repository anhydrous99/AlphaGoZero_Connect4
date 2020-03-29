//
// Created by aherrera on 3/29/20.
//

#ifndef ALPHAZERO_CONNECT4_UTILS_H
#define ALPHAZERO_CONNECT4_UTILS_H

#include <torch/torch.h>
#include <Eigen/Core>

struct TensorPair {
    torch::Tensor tensor1;
    torch::Tensor tensor2;

    TensorPair() = default;
    TensorPair(const torch::Tensor& t1, const torch::Tensor& t2);
};

enum Spot {
    None,
    Player1,
    Player2
};

// -1 for white, 0 for none, 1 for black
typedef Eigen::Matrix<Spot, 6, 7> State;

struct MoveResult {
    State state;
    bool won;

    MoveResult(State st, bool wn);
};

#endif //ALPHAZERO_CONNECT4_UTILS_H
