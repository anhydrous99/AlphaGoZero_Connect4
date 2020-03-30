//
// Created by aherrera on 3/27/20.
//

#include "MCTS.h"
#include <vector>

MCTS::MCTS(GameBoard *game, float c_puct) :  _c_puct(c_puct) {
    _game = game;
}

void MCTS::clear() {
    _visit_count.clear();
    _value.clear();
    _value_avg.clear();
    _probs.clear();
}

uint64_t MCTS::size() {
    return _value.size();
}

LeafResult MCTS::find_leaf(const State &state, Player player) {
    std::vector<State> states;
    std::vector<int64_t> actions;
    State current_state = state;
    Player current_player = player;
    float value = inf_float;
    Vector7f noises;

    while (!is_leaf(current_state)) {
        states.push_back(current_state);

        Vector7f counts = _visit_count[current_state].cast<float>();
        float total_sqrt = std::sqrt(counts.sum());
        Vector7f probs = _probs[current_state];
        Vector7f values_avg = _value_avg[current_state];

        if (current_state == state) {
            noises = generate_dirichlet(Vector7d::Constant(0.03));
            probs = 0.75 * probs + 0.25 * noises;
        }

        Vector7f score = values_avg + (_c_puct * probs * total_sqrt) / (counts + 1.0f);
        std::vector<int64_t> invalid_acts = _game->invalid_actions();
        for (int64_t act : invalid_acts)
            score[act] = -inf_float;
        int64_t action;
        score.maxCoeff(&action);
        actions.push_back(action);
        MoveResult move = _game->move(action, current_player);
        current_state = move.state;
        if (move.done)
            value = -1.0; // if somebody won the game, the value of the final state is -1
        current_player = (current_player == Player::Player2) ? Player::Player1 : Player::Player2;
        // Check for draw
        if (value == inf_float && _game->possible_moves_count() == 0)
            value = 0.0;
    }
    return {value, current_state, current_player, states, actions};
}

bool MCTS::is_leaf(const State &state) {
    return _probs.find(state) != _probs.end();
}
