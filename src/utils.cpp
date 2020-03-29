//
// Created by aherrera on 3/29/20.
//

#include "utils.h"

#include <utility>

TensorPair::TensorPair(const torch::Tensor &t1, const torch::Tensor &t2) : tensor1(t1), tensor2(t1) {}

MoveResult::MoveResult(State st, bool wn) : state(std::move(st)), won(wn) {}