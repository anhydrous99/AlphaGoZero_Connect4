//
// Created by aherrera on 3/29/20.
//

#ifndef ALPHAZERO_CONNECT4_UTILS_H
#define ALPHAZERO_CONNECT4_UTILS_H

#include <torch/torch.h>
#include <Eigen/Core>
#include <limits>

constexpr float inf_float = std::numeric_limits<float>::infinity();

enum Player {
    None,
    Player1,
    Player2
};

// -1 for white, 0 for none, 1 for black
typedef Eigen::Matrix<Player, 6, 7> State;
typedef Eigen::Array<int64_t, 7, 1> Vector7i;
typedef Eigen::Array<float, 7, 1> Vector7f;
typedef Eigen::Array<double, 7, 1> Vector7d;

bool operator<(const State &a, const State &b);

struct TensorPair {
    torch::Tensor tensor1;
    torch::Tensor tensor2;

    TensorPair() = default;

    TensorPair(const torch::Tensor &t1, const torch::Tensor &t2);
};

struct MoveResult {
    State state;
    bool done;

    MoveResult(State st, bool wn);
};

struct LeafResult {
    float value;
    State current_state;
    Player current_player;
    std::vector<State> states;
    std::vector<int64_t> actions;

    LeafResult(float val, State cur_state, Player pl, std::vector<State> st,
               std::vector<int64_t> acts);
};

Vector7f generate_dirichlet(const Vector7d &alpha);

#endif //ALPHAZERO_CONNECT4_UTILS_H
