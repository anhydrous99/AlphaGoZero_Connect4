#include <iostream>
#include <random>
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
    MCTS mcts(&net);
    std::vector<MCTS*> mcts_list{&mcts};
    uint64_t step_idx = 0, best_idx = 0;

    std::mt19937 random_generator((std::random_device()()));

    while (true) {
        auto t = std::chrono::high_resolution_clock::now();
        uint64_t prev_nodes = mcts.size(), game_steps = 0;
        GameResult res;
        for (int i = 0; i < PLAY_EPISODES; i++) {
            res = play_game(mcts_list, &replay_buffer, best_net, best_net, STEPS_BEFORE_TAU_0,
                            MCTS_SEARCHES, MCTS_BATCH_SUZE, true);
            game_steps += res.step;
        }
        auto game_nodes = mcts.size() - prev_nodes;
        uint64_t dt = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - t).count();
        uint64_t speed_steps = game_steps / dt, speed_nodes = game_nodes / dt;
        std::cout << "Step " << step_idx << ", steps " << game_steps << ", leaves " << game_nodes << ", steps/s "
                  << speed_steps << ", leaves/s " << speed_nodes << ", best_idx " << best_idx << ", replay "
                  << replay_buffer.size() << std::endl;
        step_idx++;

        if (replay_buffer.size() < MIN_REPLAY_TO_TRAIN)
            continue;

        // Train
        float sum_loss = 0.0f, sum_value_loss = 0.0f, sum_policy_loss = 0.0f;
        for (int i = 0; i < TRAIN_ROUNDS; i++) {
            auto batch = sample(replay_buffer.begin(), replay_buffer.end(), BATCH_SIZE, random_generator);
        }
    }

    return 0;
}
