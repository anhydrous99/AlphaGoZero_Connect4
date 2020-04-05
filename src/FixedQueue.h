//
// Created by Armando Herrera on 4/3/20.
//

#ifndef ALPHAZERO_CONNECT4_FIXEDQUEUE_H
#define ALPHAZERO_CONNECT4_FIXEDQUEUE_H

#include <queue>
#include <deque>

template <typename T, int MaxLen, typename Container=std::deque<T>>
class FixedQueue : public std::queue<T, Container> {
public:
    typedef typename Container::iterator iterator;
    typedef typename Container::const_iterator const_iterator;

    void push(const T &value);

    template<class... Ts>
    decltype(auto) emplace(Ts&&... values);

    iterator begin() { return this->c.begin(); }
    iterator end() { return this->c.end(); }
    const_iterator begin() const { return this->c.begin(); }
    const_iterator end() const { return this->c.end(); }
};

template<typename T, int MaxLen, typename Container>
void FixedQueue<T, MaxLen, Container>::push(const T &value) {
    if (this->size() == MaxLen)
        this->c.pop_front();
    std::queue<T, Container>::push(value);
}

template<typename T, int MaxLen, typename Container>
template<class... Ts>
decltype(auto) FixedQueue<T, MaxLen, Container>::emplace(Ts &&... values) {
    if (this->size() == MaxLen)
        this->c.pop_front();
    this->c.emplace_back(std::forward<Ts>(values)...);
}

#endif //ALPHAZERO_CONNECT4_FIXEDQUEUE_H
