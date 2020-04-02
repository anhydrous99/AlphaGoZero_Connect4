//
// Created by aherrera on 3/27/20.
//

#include "MCTS.h"
#include <vector>
#include <set>

MCTS::MCTS(GameBoard *game, Model *net, float c_puct) :  _c_puct(c_puct) {
    _game = game;
    _net = net;
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

void MCTS::search_batch(int64_t count, int64_t batch_size) {
    for (int64_t i = 0; i < count; i++) {
        search_minibatch(batch_size);
    }
}

void MCTS::search_minibatch(int64_t count) {
    std::vector<VSAT> backup_queue;
    std::vector<State> expand_states;
    std::vector<Player> expand_players;
    std::vector<SAT> expand_queue;
    std::set<State> planned;

    for (int i = 0; i < count; i++) {
        LeafResult result = find_leaf(_game->get_state(), _game->get_current_turn());
        if (result.value == inf_float)
            backup_queue.emplace_back(result.value, result.states, result.actions);
        else {
            if (planned.find(result.current_state) == planned.end()) {
                planned.insert(result.current_state);
                expand_states.push_back(result.current_state);
                expand_players.push_back(result.current_player);
                expand_queue.emplace_back(result.current_state, result.states, result.actions);
            }
        }
    }

    // Do expansion of nodes
    if (!expand_queue.empty()) {
        torch::Tensor batch_tensor = get_states_tensors(expand_states);
        TensorPair pair = (*_net)->forward(batch_tensor);
        torch::Tensor prob_tensor = torch::squeeze(torch::softmax(pair.tensor1, 1));
        torch::Tensor value_tensor = torch::squeeze(pair.tensor2);

        for (uint64_t i = 0; i < expand_queue.size(); i++) {
            SAT sat = expand_queue[i];
            _visit_count[sat.state] = Vector7i::Zero();
            _value[sat.state] = Vector7f::Zero();
            _value_avg[sat.state] = Vector7f::Zero();
            _probs[sat.state] = toEigen<Vector7f>(prob_tensor);
            backup_queue.emplace_back(value_tensor[i].item<float>(), sat.states, sat.actions);
        }
    }

    // Perform backup of the searches
    for (const auto& queue : backup_queue) {
        float current_value = -queue.value;
        for (int64_t i = queue.states.size() - 1; i >= 0; i--) {
            const State &state = queue.states[i];
            const int64_t action = queue.actions[i];
            _visit_count[state][action] += 1;
            _value[state][action] += current_value;
            _value_avg[state][action] = _value[state][action] / _visit_count[state][action];
            current_value = -current_value;
        }
    }
}

VP MCTS::get_policy_value(const State &state, int64_t tau) {
    auto counts = _visit_count[state];
    Vector7f probs = Vector7f::Zero();
    if (tau == 0) {
        int64_t argmax;
        counts.maxCoeff(&argmax);
        probs[argmax] = 1.0f;
    } else {
        counts = Eigen::pow(counts, 1.0f / tau);
        int64_t total = counts.sum();
        probs = counts.cast<float>() / total;
    }
    auto values = _value_avg[state];
    return {values, probs};
}
