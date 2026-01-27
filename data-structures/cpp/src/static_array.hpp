#pragma once

#include <cstddef>
#include <stdexcept>

template <typename T, std::size_t N>
class StaticArray {
public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    constexpr T& at(size_type index) {
        if (index >= N)
            throw std::out_of_range("StaticArray::at: index out of range");
        return data_[index];
    }

    constexpr const T& at(size_type index) const {
        if (index >= N)
            throw std::out_of_range("StaticArray::at: index out of range");
        return data_[index];
    }

    constexpr T& operator[](size_type index) { return data_[index]; }
    constexpr const T& operator[](size_type index) const { return data_[index]; }

    constexpr T& front() { return data_[0]; }
    constexpr const T& front() const { return data_[0]; }

    constexpr T& back() { return data_[N - 1]; }
    constexpr const T& back() const { return data_[N - 1]; }

    constexpr T* data() { return data_; }
    constexpr const T* data() const { return data_; }

    constexpr size_type size() const { return N; }
    constexpr bool empty() const { return N == 0; }

    constexpr void fill(const T& value) {
        for (size_type i = 0; i < N; ++i)
            data_[i] = value;
    }

    constexpr iterator begin() { return data_; }
    constexpr const_iterator begin() const { return data_; }
    constexpr iterator end() { return data_ + N; }
    constexpr const_iterator end() const { return data_ + N; }

private:
    T data_[N];
};

template <typename T>
class StaticArray<T, 0> {
public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    constexpr T& at(size_type) {
        throw std::out_of_range("StaticArray::at: index out of range");
    }

    constexpr const T& at(size_type) const {
        throw std::out_of_range("StaticArray::at: index out of range");
    }

    constexpr T* data() { return nullptr; }
    constexpr const T* data() const { return nullptr; }

    constexpr size_type size() const { return 0; }
    constexpr bool empty() const { return true; }

    constexpr void fill(const T&) {}

    constexpr iterator begin() { return nullptr; }
    constexpr const_iterator begin() const { return nullptr; }
    constexpr iterator end() { return nullptr; }
    constexpr const_iterator end() const { return nullptr; }
};
