//
// Created by aherrera on 3/26/20.
//

#ifndef ALPHAZERO_CONNECT4_GAME_H
#define ALPHAZERO_CONNECT4_GAME_H

#include <Eigen/Core>
#include <ostream>
#include <cstdint>
#include <utility>
#include "utils.h"

class GameBoard {
    State board;

public:
    GameBoard();
    bool check_won(Spot player);
    bool check_won();
    MoveResult move(long col, Spot player);
    friend std::ostream &operator<<(std::ostream &out, const GameBoard &board);
};

std::ostream &operator<<(std::ostream &out, const GameBoard &board);

#endif //ALPHAZERO_CONNECT4_GAME_H
