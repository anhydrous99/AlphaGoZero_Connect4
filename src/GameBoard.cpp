//
// Created by aherrera on 3/26/20.
//

#include "GameBoard.h"
#include <cassert>

GameBoard::GameBoard() : _board(State::Constant(Player::None)) {
}

GameBoard::GameBoard(Player turn) : _board(State::Constant(Player::None)) {
    _turn = turn;
}

void GameBoard::change_turn() {
    assert(_turn == Player::Player1 || _turn == Player::Player2);
    if (_turn == Player::Player1)
        _turn = Player::Player2;
    else if (_turn == Player::Player2)
        _turn = Player::Player1;
}

bool GameBoard::check_won(Player player) {
    assert(player == Player::Player1 || player == Player::Player2);
    const long rows = _board.rows();
    const long cols = _board.cols();

    // horizontalCheck
    for (long j = 0; j < cols - 3; j++) {
        for (long i = 0; i < rows; i++) {
            if (_board(i, j) == player && _board(i, j + 1) == player &&
                _board(i, j + 2) && _board(i, j + 3) == player)
                return true;
        }
    }
    // verticalCheck
    for (long i = 0; i < rows - 3; i++) {
        for (long j = 0; j < cols; j++) {
            if (_board(i, j) == player && _board(i + 1, j) == player &&
                _board(i + 2, j) == player && _board(i + 3, j) == player)
                return true;
        }
    }
    // ascendingDiagonalCheck
    for (long i = 3; i < rows; i++) {
        for (long j = 0; j < cols - 3; j++) {
            if (_board(i, j) == player && _board(i - 1, j + 1) == player &&
                _board(i - 2, j + 3) == player && _board(i - 3, j + 3))
                return true;
        }
    }
    // descendingDiagnonalCheck
    for (long i = 3; i < rows; i++) {
        for (long j = 3; j < cols; j++) {
            if (_board(i, j) == player && _board(i - 1, j - 1) == player &&
                _board(i - 2, j - 2) == player && _board(i - 3, j - 3) == player)
                return true;
        }
    }
    return false;
}

bool GameBoard::check_done() {
    return check_won(Player::Player1) && check_won(Player::Player2);
}

Player GameBoard::winner() {
    if (check_won(Player::Player1))
        return Player::Player1;
    else if (check_won(Player::Player2))
        return Player::Player2;
    else
        return Player::None;
}

MoveResult GameBoard::move(int64_t col) {
    return move(col, _turn);
}

MoveResult GameBoard::move(int64_t col, Player player) {
    assert(player == Player::Player1 || player == Player::Player2);
    assert(col < _board.cols());
    _turn = player;
    change_turn();
    for (long i = _board.rows() - 1; i >= 0; i--) {
        if (_board(i, col) == Player::None) {
            _board(i, col) = player;
            return {_board, check_won(player)};
        }
    }
    return {_board, false}; // No spot available
}

std::vector<int64_t> GameBoard::invalid_actions() {
    std::vector<int64_t> invalid;
    for (int64_t i = 0; i < _board.cols(); i++) {
        if (_board(0, i) != Player::None)
            invalid.push_back(i);
    }
    return invalid;
}

int64_t GameBoard::invalid_actions_count() {
    int64_t invalid_count = 0;
    for (int64_t i = 0; i < _board.cols(); i++) {
        if (_board(0, i) != Player::None)
            invalid_count++;
    }
    return invalid_count;
}

std::vector<int64_t> GameBoard::possible_moves() {
    std::vector<int64_t> valid;
    for (int64_t i = 0; i < _board.cols(); i++) {
        if (_board(0, i) == Player::None)
            valid.push_back(i);
    }
    return valid;
}

int64_t GameBoard::possible_moves_count() {
    int64_t count = 0;
    for (int64_t i = 0; i < _board.cols(); i++) {
        if (_board(0, i) == Player::None)
            count++;
    }
    return count;
}

std::ostream &operator<<(std::ostream &out, const GameBoard &board) {
    const State &gbm = board._board;
    const int rows = gbm.rows();
    const int cols = gbm.cols();
    for (int i = 0; i < cols + 1; i++)
        out << '_';
    out << "_\n";
    for (int i = 0; i < rows; i++) {
        out << '|';
        for (int j = 0; j < cols; j++) {
            Player value = gbm(i, j);
            if (value == Player::Player1)
                out << '+';
            if (value == Player::Player2)
                out << '-';
            if (value == Player::None)
                out << ' ';
        }
        out << "|\n";
    }
    for (int i = 0; i < cols + 1; i++)
        out << '-';
    out << "-\n";
    return out;
}
