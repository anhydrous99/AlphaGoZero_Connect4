cmake_minimum_required(VERSION 3.15)
project(AlphaZero_Connect4)

set(CMAKE_CXX_STANDARD 14)

find_package(GSL REQUIRED)
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(AlphaZero_Connect4
        src/main.cpp
        src/GameBoard.cpp
        src/GameBoard.h
        src/MCTS.cpp
        src/MCTS.h
        src/Model.cpp
        src/Model.h
        src/Utilities.cpp
        src/Utilities.h
        src/FixedQueue.h
        src/GamePlay.cpp
        src/GamePlay.h
        src/Defines.h)
target_link_libraries(AlphaZero_Connect4 ${TORCH_LIBRARIES} Eigen3::Eigen GSL::gsl)