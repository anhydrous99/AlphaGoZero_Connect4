//
// Created by Armando Herrera on 4/4/2020.
//

#ifndef ALPHAZERO_CONNECT4_GAMEPLAY_H
#define ALPHAZERO_CONNECT4_GAMEPLAY_H

#include "MCTS.h"
#include "Model.h"
#include "Defines.h"
#include "Utilities.h"
#include "FixedQueue.h"

GameResult
play_game(std::vector<MCTS*> &input_mcts_store, FixedQueue<BufferEntry> *replay_buffer, Model &net1,
          Model &net2, int64_t steps_before_tau_0, int64_t mcts_searches, int64_t mcts_batch_size, bool use_replay);

float evaluate(Model &net1, Model &net2, int64_t rounds);

#endif //ALPHAZERO_CONNECT4_GAMEPLAY_H
