#include <iostream>
#include <random>
#include "GameBoard.h"
#include "FixedQueue.h"
#include "MCTS.h"
#include "Model.h"
#include "utils.h"

#define REPLAY_BUFFER 5000

GameResult
play_game(const std::vector<MCTS> &mcts_store, FixedQueue<BufferEntry, REPLAY_BUFFER> &replay_buffer, Model &net1,
          Model &net2, int64_t steps_before_tau_0, int64_t mcts_searches, int64_t mcts_batch_size);

int main() {
    std::vector<int64_t> input_shape{2, 6, 7};
    torch::Tensor input_tensor = torch::rand(input_shape);
    input_tensor = torch::unsqueeze(input_tensor, 0);
    Model model1(input_shape, 7, 64);
    Model model2(input_shape, 7, 64);

    TensorPair pair1 = model1(input_tensor);
    TensorPair pair2 = model2(input_tensor);

    std::cout << "Tensor 1: \n" << pair1.tensor1 << std::endl;
    std::cout << "Tensor 2: \n" << pair2.tensor1 << std::endl;

    sync_weights(model1, model2);

    pair1 = model1(input_tensor);
    pair2 = model2(input_tensor);

    std::cout << "Tensor 1: \n" << pair1.tensor1 << std::endl;
    std::cout << "Tensor 2: \n" << pair2.tensor1 << std::endl;
    return 0;
}

GameResult
play_game(const std::vector<MCTS> &mcts_store, FixedQueue<BufferEntry, REPLAY_BUFFER> &replay_buffer, Model &net1,
          Model &net2, int64_t steps_before_tau_0, int64_t mcts_searches, int64_t mcts_batch_size) {
    std::mt19937 gen((std::random_device())()); // Init randomness generator
    GameBoard board(randint_range(Player::Player1, Player::Player2, gen()));
    std::vector<MCTS> mcts_stores;
    if (mcts_store.empty()) {
        mcts_stores.emplace_back(MCTS(&board, &net1));
        mcts_stores.emplace_back(MCTS(&board, &net2));
    } else if (mcts_store.size() == 1) {
        mcts_stores.push_back(mcts_stores[0]);
        mcts_stores.push_back(mcts_stores[0]);
    } else if (mcts_stores.size() == 2) {
        mcts_stores = mcts_store;
    }

    auto state = board.get_state();
    std::vector<Model> nets{net1, net2};
}
