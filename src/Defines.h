//
// Created by Armando Herrera on 4/4/2020.
//

#ifndef ALPHAZERO_CONNECT4_DEFINES_H
#define ALPHAZERO_CONNECT4_DEFINES_H

#define PLAY_EPISODES 1
#define MCTS_SEARCHES 10
#define MCTS_BATCH_SIZE 8
#define REPLAY_BUFFER 5000
#define LEARNING_RATE 0.1
#define BATCH_SIZE 256
#define TRAIN_ROUNDS 10
#define MIN_REPLAY_TO_TRAIN 2000
#define BEST_NET_WIN_RATIO 0.60
#define EVALUATE_EVERY_STEP 100
#define EVALUATION_ROUNDS 20
#define STEPS_BEFORE_TAU_0 10
#define N_ACTIONS 7
#define N_FILTERS 64
#define MAX_STEPS 100000

#endif //ALPHAZERO_CONNECT4_DEFINES_H
