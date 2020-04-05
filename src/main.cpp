#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "FixedQueue.h"
#include "Utilities.h"
#include "GamePlay.h"
#include "Defines.h"
#include "Model.h"
#include "MCTS.h"

int main() {
    std::vector<int64_t> obs_shape{2, 6, 7};
    int64_t n_actions = N_ACTIONS, n_filters = N_FILTERS;

    Model net(obs_shape, n_actions, n_filters);
    Model best_net(obs_shape, n_actions, n_filters);
    sync_weights(best_net, net);
    std::cout << net;

    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(LEARNING_RATE));

    FixedQueue<BufferEntry, REPLAY_BUFFER> replay_buffer;
    GameBoard board;
    MCTS mcts(&board, &net);
    uint64_t step_idx = 0, best_idx = 0;

    while (true) {
        auto t = std::chrono::time_point::now();
        uint64_t prev_nodes = mcts.size(), game_steps = 0;
        GameResult res;
        for (int i = 0; i < PLAY_EPISODES; i++)
            res = play_game(std::vector<MCTS>{mcts}, &replay_buffer, )
    }

    return 0;
}
