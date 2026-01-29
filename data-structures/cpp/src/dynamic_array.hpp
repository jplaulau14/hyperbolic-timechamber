#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

template <typename T>
class DynamicArray {
public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    DynamicArray() : data_(nullptr), size_(0), capacity_(0) {}

    explicit DynamicArray(size_type count)
        : data_(count > 0 ? new T[count]() : nullptr), size_(count), capacity_(count) {}

    DynamicArray(const DynamicArray& other)
        : data_(other.capacity_ > 0 ? new T[other.capacity_] : nullptr),
          size_(other.size_),
          capacity_(other.capacity_) {
        for (size_type i = 0; i < size_; ++i)
            data_[i] = other.data_[i];
    }

    DynamicArray(DynamicArray&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    DynamicArray& operator=(const DynamicArray& other) {
        if (this != &other) {
            delete[] data_;
            capacity_ = other.capacity_;
            size_ = other.size_;
            data_ = capacity_ > 0 ? new T[capacity_] : nullptr;
            for (size_type i = 0; i < size_; ++i)
                data_[i] = other.data_[i];
        }
        return *this;
    }

    DynamicArray& operator=(DynamicArray&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    ~DynamicArray() { delete[] data_; }

    T& at(size_type index) {
        if (index >= size_)
            throw std::out_of_range("DynamicArray::at: index out of range");
        return data_[index];
    }

    const T& at(size_type index) const {
        if (index >= size_)
            throw std::out_of_range("DynamicArray::at: index out of range");
        return data_[index];
    }

    T& operator[](size_type index) { return data_[index]; }
    const T& operator[](size_type index) const { return data_[index]; }

    T& front() {
        if (size_ == 0)
            throw std::out_of_range("DynamicArray::front: array is empty");
        return data_[0];
    }
    const T& front() const {
        if (size_ == 0)
            throw std::out_of_range("DynamicArray::front: array is empty");
        return data_[0];
    }

    T& back() {
        if (size_ == 0)
            throw std::out_of_range("DynamicArray::back: array is empty");
        return data_[size_ - 1];
    }
    const T& back() const {
        if (size_ == 0)
            throw std::out_of_range("DynamicArray::back: array is empty");
        return data_[size_ - 1];
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

    size_type size() const { return size_; }
    size_type capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }

    void reserve(size_type new_cap) {
        if (new_cap <= capacity_)
            return;
        T* new_data = new T[new_cap];
        for (size_type i = 0; i < size_; ++i)
            new_data[i] = std::move(data_[i]);
        delete[] data_;
        data_ = new_data;
        capacity_ = new_cap;
    }

    void push_back(const T& value) {
        if (size_ == capacity_)
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        data_[size_++] = value;
    }

    void push_back(T&& value) {
        if (size_ == capacity_)
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        data_[size_++] = std::move(value);
    }

    void pop_back() {
        if (size_ == 0)
            throw std::out_of_range("DynamicArray::pop_back: array is empty");
        --size_;
        data_[size_].~T();
    }

    void clear() {
        for (size_type i = 0; i < size_; ++i)
            data_[i].~T();
        size_ = 0;
    }

    iterator begin() { return data_; }
    const_iterator begin() const { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator end() const { return data_ + size_; }

private:
    T* data_;
    size_type size_;
    size_type capacity_;
};
