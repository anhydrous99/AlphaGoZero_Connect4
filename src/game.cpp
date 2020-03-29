//
// Created by aherrera on 3/26/20.
//

#include "game.h"
#include <cassert>

GameBoard::GameBoard() : board(State::Zero()) {
}

bool GameBoard::check_won(Spot player) {
    assert(player == Spot::Player1 || player == Spot::Player2);
    const long rows = board.rows();
    const long cols = board.cols();

    // horizontalCheck
    for (long j = 0; j < cols - 3; j++) {
        for (long i = 0; i < rows; i++) {
            if (board(i, j) == player && board(i, j + 1) == player &&
                board(i, j + 2) && board(i, j + 3) == player)
                return true;
        }
    }
    // verticalCheck
    for (long i = 0; i < rows - 3; i++) {
        for (long j = 0; j < cols; j++) {
            if (board(i, j) == player && board(i + 1, j) == player &&
                board(i + 2, j) == player && board(i + 3, j) == player)
                return true;
        }
    }
    // ascendingDiagonalCheck
    for (long i = 3; i < rows; i++) {
        for (long j = 0; j < cols - 3; j++) {
            if (board(i, j) == player && board(i - 1, j + 1) == player &&
                board(i - 2, j + 3) == player && board(i - 3, j + 3))
                return true;
        }
    }
    // descendingDiagnonalCheck
    for (long i = 3; i < rows; i++) {
        for (long j = 3; j < cols; j++) {
            if (board(i, j) == player && board(i - 1, j - 1) == player &&
                board(i - 2, j - 2) == player && board(i - 3, j - 3) == player)
                return true;
        }
    }
    return false;
}

bool GameBoard::check_won() {
    return check_won(Spot::Player1) && check_won(Spot::Player2);
}

MoveResult GameBoard::move(long col, Spot player) {
    assert(player == Spot::Player1 || player == Spot::Player2);
    assert(col < board.cols());
    for (long i = board.rows() - 1; i >= 0; i--) {
        if (board(i, col) != Spot::None) {
            board(i, col) = player;
            return {board, check_won(player)};
        }
    }
    return {board, false}; // No spot available
}

std::ostream &operator<<(std::ostream &out, const GameBoard &board) {
    const State &gbm = board.board;
    const int rows = gbm.rows();
    const int cols = gbm.cols();
    for (int i = 0; i < cols + 1; i++)
        out << '_';
    out << "_\n";
    for (int i = 0; i < rows; i++) {
        out << '|';
        for (int j = 0; j < cols; j++) {
            Spot value = gbm(i, j);
            if (value == Spot::Player1)
                out << '+';
            if (value == Spot::Player2)
                out << '-';
            if (value == Spot::None)
                out << ' ';
        }
        out << "|\n";
    }
    for (int i = 0; i < cols + 1; i++)
        out << '-';
    out << "-\n";
    return out;
}
