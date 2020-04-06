//
// Created by Armando Herrera on 4/4/2020.
//

#include "GamePlay.h"

GameResult
play_game(std::vector<MCTS*> &input_mcts_store, FixedQueue<BufferEntry, REPLAY_BUFFER> *replay_buffer, Model &net1,
          Model &net2, int64_t steps_before_tau_0, int64_t mcts_searches, int64_t mcts_batch_size, bool use_replay) {
    std::mt19937 gen((std::random_device()) ()); // Init randomness generator
    GameBoard board(randint_range(Player::Player1, Player::Player2, gen));
    bool do_del = false;
    std::vector<MCTS*> mcts_stores;
    if (input_mcts_store.empty()) {
        mcts_stores.emplace_back(new MCTS(&net1));
        mcts_stores.emplace_back(new MCTS(&net2));
        do_del = true;
    } else if (input_mcts_store.size() == 1) {
        mcts_stores.push_back(input_mcts_store[0]);
        mcts_stores.push_back(input_mcts_store[0]);
    } else if (input_mcts_store.size() == 2) {
        mcts_stores.push_back(input_mcts_store[0]);
        mcts_stores.push_back(input_mcts_store[1]);
    }

    auto state = board.get_state();
    std::vector<Model> nets{net1, net2};
    int64_t step = 0;
    int64_t tau = (steps_before_tau_0 > 0) ? 1 : 0;
    std::vector<HistoryEntry> game_history;

    GameResult output;

    uint8_t res;
    while (true) {
        Player player = board.get_current_turn();
        auto idx = player_to_index<int64_t>(player);
        mcts_stores[idx]->search_batch(mcts_searches, mcts_batch_size, state, player);
        VP store = mcts_stores[idx]->get_policy_value(state, tau);
        game_history.emplace_back(state, player, store.probabilities);
        Vector7d probabilities = store.probabilities.cast<double>();
        int64_t action = generate_choices(probabilities);
        auto possible_moves = board.possible_moves();
        if (std::find(possible_moves.begin(), possible_moves.end(), action) == possible_moves.end())
            std::cerr << "Impossible action selected\n";
        MoveResult result = board.move(action);
        state = result.state;
        if (result.done) {
            output.result = (board.get_current_turn() == Player::Player1) ? 1 : -1;
            res = 1;
            break;
        }
        if (possible_moves.empty()) {
            output.result = 0;
            res = 0;
            break;
        }
        step++;
        if (step >= steps_before_tau_0)
            tau = 0;
    }

    if (use_replay) {
        for (auto it = game_history.rbegin(); it != game_history.rend(); ++it) {
            replay_buffer->emplace(it->state, it->player, it->probabilities, res);
            res = -res;
        }
    }
    output.step = step;

    if (do_del) {
        delete mcts_stores[0];
        delete mcts_stores[1];
    }

    return output;
}

float evaluate(Model &net1, Model &net2, int64_t rounds) {
    int64_t n1_win = 0, n2_win = 0;
    GameBoard board;
    std::vector<MCTS*> mcts{new MCTS(&net1), new MCTS(&net2)};
    GameResult result;

    for (int i = 0; i < rounds; i++) {
        result = play_game(mcts, nullptr, net1, net2, 0, 20, 16, false);
        if (result.result < -0.5)
            n2_win++;
        else if (result.result > 0.5)
            n1_win++;
    }
    delete mcts[0];
    delete mcts[1];
    return static_cast<float>(n1_win) / static_cast<float>(n1_win + n2_win);
}
