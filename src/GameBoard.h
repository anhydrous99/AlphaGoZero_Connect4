//
// Created by aherrera on 3/26/20.
//

#ifndef ALPHAZERO_CONNECT4_GAMEBOARD_H
#define ALPHAZERO_CONNECT4_GAMEBOARD_H

#include <Eigen/Core>
#include <ostream>
#include "utils.h"

class GameBoard {
    State _board;
    Player _turn = Player::Player1;
    bool done = false;

    void change_turn();
public:
    GameBoard();
    explicit GameBoard(Player turn);
    bool check_won(Player player);
    bool check_done();
    Player winner();
    MoveResult move(int64_t col);
    MoveResult move(int64_t col, Player player);
    friend std::ostream &operator<<(std::ostream &out, const GameBoard &board);
    std::vector<int64_t> invalid_actions();
    int64_t invalid_actions_count();
    std::vector<int64_t> possible_moves();
    int64_t possible_moves_count();
};

std::ostream &operator<<(std::ostream &out, const GameBoard &board);

#endif //ALPHAZERO_CONNECT4_GAMEBOARD_H
