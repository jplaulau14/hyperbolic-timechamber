#pragma once

#include "dynamic_array.hpp"
#include <stdexcept>

template <typename T>
class Stack {
public:
    using size_type = std::size_t;

    Stack() = default;

    Stack(const Stack& other) = default;

    Stack(Stack&& other) noexcept = default;

    Stack& operator=(const Stack& other) = default;

    Stack& operator=(Stack&& other) noexcept = default;

    ~Stack() = default;

    void push(const T& value) { data_.push_back(value); }

    void push(T&& value) { data_.push_back(std::move(value)); }

    T pop() {
        if (data_.empty())
            throw std::out_of_range("Stack::pop: empty stack");
        T value = std::move(data_.back());
        data_.pop_back();
        return value;
    }

    T& top() {
        if (data_.empty())
            throw std::out_of_range("Stack::top: empty stack");
        return data_.back();
    }

    const T& top() const {
        if (data_.empty())
            throw std::out_of_range("Stack::top: empty stack");
        return data_.back();
    }

    size_type size() const { return data_.size(); }

    bool empty() const { return data_.empty(); }

    void clear() { data_.clear(); }

private:
    DynamicArray<T> data_;
};
