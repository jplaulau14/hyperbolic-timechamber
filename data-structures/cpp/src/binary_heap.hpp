#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

template <typename T>
class BinaryHeap {
public:
    using value_type = T;
    using size_type = std::size_t;

    BinaryHeap() : data_(nullptr), size_(0), capacity_(0) {}

    BinaryHeap(const BinaryHeap& other)
        : data_(other.capacity_ > 0 ? new T[other.capacity_] : nullptr),
          size_(other.size_),
          capacity_(other.capacity_) {
        for (size_type i = 0; i < size_; ++i)
            data_[i] = other.data_[i];
    }

    BinaryHeap(BinaryHeap&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    BinaryHeap& operator=(const BinaryHeap& other) {
        BinaryHeap temp(other);
        swap(temp);
        return *this;
    }

    void swap(BinaryHeap& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    BinaryHeap& operator=(BinaryHeap&& other) noexcept {
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

    ~BinaryHeap() { delete[] data_; }

    void push(const T& value) {
        if (size_ == capacity_)
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        data_[size_] = value;
        sift_up(size_);
        ++size_;
    }

    void push(T&& value) {
        if (size_ == capacity_)
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        data_[size_] = std::move(value);
        sift_up(size_);
        ++size_;
    }

    T pop() {
        if (size_ == 0)
            throw std::out_of_range("BinaryHeap::pop: heap is empty");
        T result = std::move(data_[0]);
        --size_;
        if (size_ > 0) {
            data_[0] = std::move(data_[size_]);
            sift_down(0);
        }
        return result;
    }

    T& peek() {
        if (size_ == 0)
            throw std::out_of_range("BinaryHeap::peek: heap is empty");
        return data_[0];
    }

    const T& peek() const {
        if (size_ == 0)
            throw std::out_of_range("BinaryHeap::peek: heap is empty");
        return data_[0];
    }

    size_type size() const { return size_; }
    bool empty() const { return size_ == 0; }
    void clear() { size_ = 0; }

    static BinaryHeap from_array(const T* arr, size_type n) {
        BinaryHeap heap;
        if (n == 0)
            return heap;
        heap.capacity_ = n;
        heap.size_ = n;
        heap.data_ = new T[n];
        for (size_type i = 0; i < n; ++i)
            heap.data_[i] = arr[i];
        // Heapify: start from last non-leaf and sift down
        for (size_type i = n / 2; i > 0; --i)
            heap.sift_down(i - 1);
        return heap;
    }

private:
    T* data_;
    size_type size_;
    size_type capacity_;

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

    void sift_up(size_type index) {
        while (index > 0) {
            size_type parent = (index - 1) / 2;
            if (!(data_[index] < data_[parent]))
                break;
            std::swap(data_[index], data_[parent]);
            index = parent;
        }
    }

    void sift_down(size_type index) {
        while (true) {
            size_type smallest = index;
            size_type left = 2 * index + 1;
            size_type right = 2 * index + 2;

            if (left < size_ && data_[left] < data_[smallest])
                smallest = left;
            if (right < size_ && data_[right] < data_[smallest])
                smallest = right;

            if (smallest == index)
                break;

            std::swap(data_[index], data_[smallest]);
            index = smallest;
        }
    }
};
