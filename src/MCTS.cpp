//
// Created by aherrera on 3/27/20.
//

#include "MCTS.h"

MCTS::MCTS(float c_puct) : _c_puct(c_puct) {}

void MCTS::clear() {
    _visit_count.clear();
    _value.clear();
    _value_avg.clear();
    _probs.clear();
}

uint64_t MCTS::size() {
    return _value.size();
}