#include <iostream>
#include "GameBoard.h"

int main() {
    GameBoard board;
    std::cout << board << std::endl;
    board.move(2);
    board.move(2);
    board.move(2);
    board.move(2);
    board.move(2);
    board.move(2);
    std::cout << board << std::endl;
    auto invalid_actions = board.invalid_actions();
    for (const auto& action : invalid_actions)
        std::cout << action << " ";
    return 0;
}
