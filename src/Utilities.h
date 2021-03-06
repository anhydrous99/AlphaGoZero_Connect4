//
// Created by Armando Herrera on 3/29/20.
//

#ifndef ALPHAZERO_CONNECT4_UTILITIES_H
#define ALPHAZERO_CONNECT4_UTILITIES_H

#include <torch/torch.h>
#include <Eigen/Core>
#include <type_traits>
#include <iterator>
#include <numeric>
#include <random>
#include <limits>

constexpr float inf_float = std::numeric_limits<float>::infinity();

enum Player {
    None,
    Player1,
    Player2
};

template<typename T>
T player_to_index(Player player) {
    assert(std::is_integral<T>::value);
    return static_cast<T>(player) - 1; // Player1, Player2 -> 0, 1
}

template<typename T>
Player index_to_player(T index) {
    assert(std::is_integral<T>::value);
    return static_cast<Player>(index + 1); // 0, 1 -> Player1, Player2
}

typedef Eigen::Matrix<Player, 6, 7> State;
typedef Eigen::Array<int64_t, 7, 1> Vector7i;
typedef Eigen::Array<float, 7, 1> Vector7f;
typedef Eigen::Array<double, 7, 1> Vector7d;

bool operator<(const State &a, const State &b);

struct TensorPair {
    torch::Tensor tensor1;
    torch::Tensor tensor2;

    TensorPair() = default;
    TensorPair(torch::Tensor t1, torch::Tensor t2);
};

struct MoveResult {
    State state;
    bool done;

    MoveResult(State st, bool wn);
};

struct LeafResult {
    float value;
    State current_state;
    Player current_player;
    std::vector<State> states;
    std::vector<int64_t> actions;

    LeafResult(float val, State cur_state, Player pl, std::vector<State> st,
               std::vector<int64_t> acts);
};

struct VSAT { // Value State Action Trajectory - value of State Action Trajectory
    float value;
    std::vector<State> states;
    std::vector<int64_t> actions;

    VSAT(float val, std::vector<State> sts, std::vector<int64_t> acts);
};

struct SAT { // State action trajectory
    State state;
    std::vector<State> states;
    std::vector<int64_t> actions;

    SAT(State st, std::vector<State> sts, std::vector<int64_t> acts);
};

struct VP { // Value probability
    Vector7f values;
    Vector7f probabilities;

    VP(Vector7f v, Vector7f p);
};

struct BufferEntry {
    State state;
    Player player;
    Vector7f probability;
    int8_t result;

    BufferEntry();
    BufferEntry(State st, Player pl, Vector7f prob, int8_t res);
};

struct BufferTensor {
    torch::Tensor states;
    torch::Tensor probabilities;
    torch::Tensor values;

    explicit BufferTensor(const std::vector<BufferEntry> &entries);
};

struct HistoryEntry {
    State state;
    Player player;
    Vector7f probabilities;

    HistoryEntry(State st, Player pl, Vector7f prob);
};

struct GameResult {
    int8_t result;
    int64_t step;

    GameResult();
    GameResult(int8_t res, int64_t st);
};

Vector7f generate_dirichlet(const Vector7d &alpha);

template<typename T>
constexpr torch::ScalarType get_torch_type() {
    if (std::is_same<T, int8_t>::value)
        return torch::kInt8;
    else if (std::is_same<T, int16_t>::value)
        return torch::kInt16;
    else if (std::is_same<T, int32_t>::value)
        return torch::kInt32;
    else if (std::is_same<T, int64_t>::value)
        return torch::kInt64;
    else if (std::is_same<T, uint8_t>::value)
        return torch::kUInt8;
    else if (std::is_same<T, torch::Half>::value)
        return torch::kFloat16;
    else if (std::is_same<T, float>::value)
        return torch::kFloat32;
    else if (std::is_same<T, double>::value)
        return torch::kFloat64;
    else {
        std::cerr << "Error: type not supported";
        return torch::ScalarType::Undefined;
    }
}

template<typename Derived>
torch::Tensor toTensor(const Eigen::DenseBase<Derived> &b) {
    typedef typename Eigen::internal::traits<Derived>::Scalar scalar_type;

    std::vector<int64_t> shape = (b.cols() != 1) ? std::vector<int64_t>{b.rows(), b.cols()} : std::vector<int64_t>{
            b.rows()};
    int64_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    torch::ScalarType dtype = get_torch_type<scalar_type>();
    auto options = torch::TensorOptions().dtype(dtype);
    torch::Tensor out_tensor = torch::zeros(shape, options);

    scalar_type *out_data = out_tensor.data_ptr<scalar_type>();
    const scalar_type *in_data = &b(0);
    if (shape.size() == 2)
        std::reverse_copy(in_data, in_data + size, out_data);
    else
        std::copy(in_data, in_data + size, out_data);
    return out_tensor;
}

template<typename Derived>
Derived toEigen(const torch::Tensor &tensor) {
    typedef typename Eigen::internal::traits<Derived>::Scalar scalar_type;
    assert(tensor.scalar_type() == get_torch_type<scalar_type>());
    assert(tensor.dim() <= 2);
    auto shape = tensor.sizes();
    int64_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    Derived out_matrix = (shape.size() == 2) ? Derived::Zero(shape[0], shape[1]) : Derived::Zero(shape[0], 1);

    scalar_type *out_data = &out_matrix(0);
    const scalar_type *in_data = tensor.data_ptr<scalar_type>();
    if (shape.size() == 2)
        std::reverse_copy(in_data, in_data + size, out_data);
    else
        std::copy(in_data, in_data + size, out_data);
    return out_matrix;
}

torch::Tensor get_state_tensor(const State &state);

torch::Tensor get_states_tensors(const std::vector<State> &states);

template<typename T, typename RNG>
T randint_range(T low, T high, RNG& g) {
    std::uniform_int_distribution<int64_t> dist(static_cast<int64_t>(low), static_cast<int64_t>(high));
    return static_cast<T>(dist(g));
}

size_t generate_choices(const Vector7d &P);

template<typename Itr, typename URNG>
std::vector<typename std::iterator_traits<Itr>::value_type>
sample(Itr first, Itr last, typename std::iterator_traits<Itr>::difference_type k, URNG &&g) {
    typedef typename std::iterator_traits<Itr>::difference_type diff_type;
    typedef typename std::iterator_traits<Itr>::value_type val_type;
    diff_type i = 0;
    diff_type n = last - first;
    std::vector<val_type> reservoir(k);
    for (; i < k; i++)
        reservoir[i] = first[i];
    for (; i < n; i++) {
        std::uniform_int_distribution<diff_type> d(0, i);
        diff_type j = d(g);
        if (j < k)
            reservoir[j] = first[i];
    }
    return reservoir;
}

std::string name_generator(int64_t best_idx, int64_t step_idx);

#endif //ALPHAZERO_CONNECT4_UTILITIES_H
