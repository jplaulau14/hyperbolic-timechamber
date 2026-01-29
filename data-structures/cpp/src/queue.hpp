#ifndef QUEUE_HPP
#define QUEUE_HPP

#include <cstddef>
#include <stdexcept>
#include <utility>

template <typename T>
class Queue {
public:
    Queue() : data_(nullptr), head_(0), tail_(0), size_(0), capacity_(0) {}

    ~Queue() { delete[] data_; }

    Queue(const Queue& other)
        : data_(nullptr), head_(0), tail_(0), size_(other.size_), capacity_(other.size_) {
        if (size_ > 0) {
            data_ = new T[capacity_];
            for (std::size_t i = 0; i < size_; ++i) {
                data_[i] = other.data_[(other.head_ + i) % other.capacity_];
            }
            tail_ = size_;
        }
    }

    Queue(Queue&& other) noexcept
        : data_(other.data_), head_(other.head_), tail_(other.tail_),
          size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.head_ = 0;
        other.tail_ = 0;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    Queue& operator=(const Queue& other) {
        if (this != &other) {
            Queue tmp(other);
            swap(tmp);
        }
        return *this;
    }

    Queue& operator=(Queue&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            head_ = other.head_;
            tail_ = other.tail_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.head_ = 0;
            other.tail_ = 0;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    void enqueue(const T& value) {
        if (size_ == capacity_) {
            grow();
        }
        data_[tail_] = value;
        tail_ = (tail_ + 1) % capacity_;
        ++size_;
    }

    void enqueue(T&& value) {
        if (size_ == capacity_) {
            grow();
        }
        data_[tail_] = std::move(value);
        tail_ = (tail_ + 1) % capacity_;
        ++size_;
    }

    T dequeue() {
        if (size_ == 0) {
            throw std::out_of_range("dequeue from empty queue");
        }
        T value = std::move(data_[head_]);
        head_ = (head_ + 1) % capacity_;
        --size_;
        return value;
    }

    T& front() {
        if (size_ == 0) {
            throw std::out_of_range("front on empty queue");
        }
        return data_[head_];
    }

    const T& front() const {
        if (size_ == 0) {
            throw std::out_of_range("front on empty queue");
        }
        return data_[head_];
    }

    T& back() {
        if (size_ == 0) {
            throw std::out_of_range("back on empty queue");
        }
        std::size_t back_idx = (tail_ + capacity_ - 1) % capacity_;
        return data_[back_idx];
    }

    const T& back() const {
        if (size_ == 0) {
            throw std::out_of_range("back on empty queue");
        }
        std::size_t back_idx = (tail_ + capacity_ - 1) % capacity_;
        return data_[back_idx];
    }

    std::size_t size() const { return size_; }

    bool empty() const { return size_ == 0; }

    void clear() {
        head_ = 0;
        tail_ = 0;
        size_ = 0;
    }

private:
    T* data_;
    std::size_t head_;
    std::size_t tail_;
    std::size_t size_;
    std::size_t capacity_;

    void swap(Queue& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(head_, other.head_);
        std::swap(tail_, other.tail_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    void grow() {
        std::size_t new_capacity = capacity_ == 0 ? 1 : capacity_ * 2;
        T* new_data = new T[new_capacity];
        for (std::size_t i = 0; i < size_; ++i) {
            new_data[i] = std::move(data_[(head_ + i) % capacity_]);
        }
        delete[] data_;
        data_ = new_data;
        head_ = 0;
        tail_ = size_;
        capacity_ = new_capacity;
    }
};

#endif
