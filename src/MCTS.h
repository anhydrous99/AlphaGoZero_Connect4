//
// Created by Armando Herrera on 3/27/20.
//

#ifndef ALPHAZERO_CONNECT4_MCTS_H
#define ALPHAZERO_CONNECT4_MCTS_H

#include <map>
#include <cstdint>
#include "Utilities.h"
#include "Model.h"
#include "GameBoard.h"

class MCTS {
    GameBoard _game;
    Model *_net;
    float _c_puct;
    std::map<State, Vector7i> _visit_count;
    std::map<State, Vector7f> _value;
    std::map<State, Vector7f> _value_avg;
    std::map<State, Vector7f> _probs;

public:
    explicit MCTS(Model *net, float c_punct = 1.0f);
    void clear();
    uint64_t size();
    LeafResult find_leaf(const State &state, Player player);
    bool is_leaf(const State &state);
    void search_batch(int64_t count, int64_t batch_size, const State &state, Player player);
    void search_minibatch(int64_t count, const State &state, Player player);
    VP get_policy_value(const State &state, int64_t tau = 1);
};


#endif //ALPHAZERO_CONNECT4_MCTS_H
