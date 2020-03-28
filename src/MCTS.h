//
// Created by aherrera on 3/27/20.
//

#ifndef ALPHAZERO_CONNECT4_MCTS_H
#define ALPHAZERO_CONNECT4_MCTS_H

#include <vector>
#include <cstdint>

struct Dummy {};

class MCTS {
    float _c_puct;
    std::vector<Dummy> _visit_count;
    std::vector<Dummy> _value;
    std::vector<Dummy> _value_avg;
    std::vector<Dummy> _probs;

public:
    MCTS(float c_punct);
    void clear();
    uint64_t size();
};


#endif //ALPHAZERO_CONNECT4_MCTS_H
