#include <iostream>
#include "GameBoard.h"
#include "Model.h"
#include "MCTS.h"

int main() {
    auto board = GameBoard();
    board.move(2);
    board.move(2);
    board.move(4);
    board.move(5);
    std::cout << board << std::endl;
    torch::Tensor ten = board.get_state_tensor();
    std::cout << ten << std::endl;
    return 0;
}
