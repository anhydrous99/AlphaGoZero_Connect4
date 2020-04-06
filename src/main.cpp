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
    Model target_net(obs_shape, n_actions, n_filters);
    sync_weights(target_net, net);
    std::cout << net;

    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(LEARNING_RATE));

    FixedQueue<BufferEntry, REPLAY_BUFFER> replay_buffer;
    GameBoard board;
    MCTS mcts(&net);
    std::vector<MCTS *> mcts_list{&mcts};
    uint64_t step_idx = 0, best_idx = 0;

    std::mt19937 random_generator((std::random_device()()));

    for (int64_t idx = 0; idx < MAX_STEPS; idx++) {
        auto t = std::chrono::high_resolution_clock::now();
        uint64_t prev_nodes = mcts.size(), game_steps = 0;
        GameResult res;
        for (int i = 0; i < PLAY_EPISODES; i++) {
            res = play_game(mcts_list, &replay_buffer, target_net, target_net, STEPS_BEFORE_TAU_0,
                            MCTS_SEARCHES, MCTS_BATCH_SUZE, true);
            game_steps += res.step;
        }
        auto game_nodes = mcts.size() - prev_nodes;
        std::chrono::duration<float> dt = std::chrono::high_resolution_clock::now() - t;
        float speed_steps = game_steps / dt.count(), speed_nodes = game_nodes / dt.count();
        std::cout << "Step " << step_idx << ", steps " << game_steps << ", leaves " << game_nodes << ", steps/s "
                  << speed_steps << ", leaves/s " << speed_nodes << ", best_idx " << best_idx << ", replay "
                  << replay_buffer.size() << std::endl;
        step_idx++;

        if (replay_buffer.size() < MIN_REPLAY_TO_TRAIN)
            continue;

        // Train
        float sum_loss = 0.0f, sum_value_loss = 0.0f, sum_policy_loss = 0.0f;
        for (int i = 0; i < TRAIN_ROUNDS; i++) {
            std::vector<BufferEntry> batch = sample(replay_buffer.begin(), replay_buffer.end(), BATCH_SIZE,
                                                    random_generator);
            BufferTensor bufferTensor(batch);
            optimizer.zero_grad();
            TensorPair pair = net(bufferTensor.states);

            auto loss_value_tensor = torch::mse_loss(pair.tensor2.squeeze(-1), bufferTensor.values);
            auto loss_policy_tensor = -torch::log_softmax(pair.tensor1, 1) * bufferTensor.probabilities;
            loss_policy_tensor = loss_policy_tensor.sum(1).mean();
            auto loss_tensor = loss_policy_tensor + loss_value_tensor;
            loss_tensor.backward();
            optimizer.step();
            sum_loss += loss_tensor.item<float>();
            sum_value_loss += loss_value_tensor.item<float>();
            sum_policy_loss += loss_policy_tensor.item<float>();
        }

        if (step_idx % EVALUATE_EVERY_STEP == 0) {
            float win_ratio = evaluate(net, target_net, EVALUATION_ROUNDS);
            std::cout << "Net evaluated, win ratio = " << win_ratio << std::endl;
            if (win_ratio > BEST_NET_WIN_RATIO) {
                std::cout << "Net is better than current best, syncing\n";
                sync_weights(target_net, net);
                best_idx++;
                mcts.clear();
            }
        }
    }

    return 0;
}
