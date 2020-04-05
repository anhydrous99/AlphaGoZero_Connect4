//
// Created by Armando Herrera on 4/4/2020.
//

#include "GamePlay.h"

GameResult
play_game(const std::vector<MCTS> &mcts_store, FixedQueue<BufferEntry, REPLAY_BUFFER> &replay_buffer, Model &net1,
          Model &net2, int64_t steps_before_tau_0, int64_t mcts_searches, int64_t mcts_batch_size) {
    std::mt19937 gen((std::random_device()) ()); // Init randomness generator
    GameBoard board(randint_range(Player::Player1, Player::Player2, gen));
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
    int64_t step = 0;
    int64_t tau = (steps_before_tau_0 > 0) ? 1 : 0;
    std::vector<HistoryEntry> game_history;

    GameResult output;

    uint8_t res;
    while (true) {
        Player player = board.get_current_turn();
        auto idx = player_to_index<int64_t>(player);
        mcts_stores[idx].search_batch(mcts_searches, mcts_batch_size);
        VP store = mcts_stores[idx].get_policy_value(state, tau);
        game_history.emplace_back(state, player, store.probabilities);
        int64_t action = generate_choices(store.probabilities.cast<double>());
        auto possible_moves = board.possible_moves();
        if (std::find(possible_moves.begin(), possible_moves.end(), action) == possible_moves.end())
            std::cerr << "Impossible action selected\n";
        MoveResult result = board.move(action);
        if (result.done) {
            output.result = (board.get_current_turn() == Player::Player1) ? 0 : -1;
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

    for (auto it = game_history.rbegin(); it != game_history.rend(); ++it) {
        replay_buffer.emplace(it->state, it->player, it->probabilities, res);
        res = -res;
    }
    output.step = step;
    return output;
}
