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

    constexpr T& operator[](size_type index) noexcept { return data_[index]; }
    constexpr const T& operator[](size_type index) const noexcept { return data_[index]; }

    constexpr T& front() { return data_[0]; }
    constexpr const T& front() const { return data_[0]; }

    constexpr T& back() { return data_[N - 1]; }
    constexpr const T& back() const { return data_[N - 1]; }

    constexpr T* data() noexcept { return data_; }
    constexpr const T* data() const noexcept { return data_; }

    constexpr size_type size() const noexcept { return N; }
    constexpr bool empty() const noexcept { return N == 0; }

    constexpr void fill(const T& value) {
        for (size_type i = 0; i < N; ++i)
            data_[i] = value;
    }

    constexpr iterator begin() noexcept { return data_; }
    constexpr const_iterator begin() const noexcept { return data_; }
    constexpr iterator end() noexcept { return data_ + N; }
    constexpr const_iterator end() const noexcept { return data_ + N; }
    constexpr const_iterator cbegin() const noexcept { return data_; }
    constexpr const_iterator cend() const noexcept { return data_ + N; }

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

    constexpr T* data() noexcept { return nullptr; }
    constexpr const T* data() const noexcept { return nullptr; }

    constexpr size_type size() const noexcept { return 0; }
    constexpr bool empty() const noexcept { return true; }

    constexpr void fill(const T&) {}

    constexpr iterator begin() noexcept { return nullptr; }
    constexpr const_iterator begin() const noexcept { return nullptr; }
    constexpr iterator end() noexcept { return nullptr; }
    constexpr const_iterator end() const noexcept { return nullptr; }
    constexpr const_iterator cbegin() const noexcept { return nullptr; }
    constexpr const_iterator cend() const noexcept { return nullptr; }
};
