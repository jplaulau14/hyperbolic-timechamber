#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

template <typename T>
class LinkedList {
private:
    struct Node {
        T value;
        Node* next;
        Node(const T& v) : value(v), next(nullptr) {}
        Node(T&& v) : value(std::move(v)), next(nullptr) {}
    };

public:
    using size_type = std::size_t;

    class Iterator {
    public:
        Iterator(Node* node) : current_(node) {}
        T& operator*() { return current_->value; }
        Iterator& operator++() { current_ = current_->next; return *this; }
        bool operator!=(const Iterator& other) const { return current_ != other.current_; }
    private:
        Node* current_;
    };

    class ConstIterator {
    public:
        ConstIterator(const Node* node) : current_(node) {}
        const T& operator*() const { return current_->value; }
        ConstIterator& operator++() { current_ = current_->next; return *this; }
        bool operator!=(const ConstIterator& other) const { return current_ != other.current_; }
    private:
        const Node* current_;
    };

    LinkedList() : head_(nullptr), tail_(nullptr), size_(0) {}

    LinkedList(const LinkedList& other) : head_(nullptr), tail_(nullptr), size_(0) {
        for (Node* curr = other.head_; curr != nullptr; curr = curr->next)
            push_back(curr->value);
    }

    LinkedList(LinkedList&& other) noexcept
        : head_(other.head_), tail_(other.tail_), size_(other.size_) {
        other.head_ = nullptr;
        other.tail_ = nullptr;
        other.size_ = 0;
    }

    LinkedList& operator=(const LinkedList& other) {
        if (this != &other) {
            clear();
            for (Node* curr = other.head_; curr != nullptr; curr = curr->next)
                push_back(curr->value);
        }
        return *this;
    }

    LinkedList& operator=(LinkedList&& other) noexcept {
        if (this != &other) {
            clear();
            head_ = other.head_;
            tail_ = other.tail_;
            size_ = other.size_;
            other.head_ = nullptr;
            other.tail_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~LinkedList() { clear(); }

    void push_front(const T& value) {
        Node* node = new Node(value);
        node->next = head_;
        head_ = node;
        if (tail_ == nullptr)
            tail_ = node;
        ++size_;
    }

    void push_front(T&& value) {
        Node* node = new Node(std::move(value));
        node->next = head_;
        head_ = node;
        if (tail_ == nullptr)
            tail_ = node;
        ++size_;
    }

    void push_back(const T& value) {
        Node* node = new Node(value);
        if (tail_ != nullptr)
            tail_->next = node;
        tail_ = node;
        if (head_ == nullptr)
            head_ = node;
        ++size_;
    }

    void push_back(T&& value) {
        Node* node = new Node(std::move(value));
        if (tail_ != nullptr)
            tail_->next = node;
        tail_ = node;
        if (head_ == nullptr)
            head_ = node;
        ++size_;
    }

    T pop_front() {
        if (head_ == nullptr)
            throw std::out_of_range("LinkedList::pop_front: list is empty");
        Node* node = head_;
        T value = std::move(node->value);
        head_ = head_->next;
        if (head_ == nullptr)
            tail_ = nullptr;
        delete node;
        --size_;
        return value;
    }

    T pop_back() {
        if (head_ == nullptr)
            throw std::out_of_range("LinkedList::pop_back: list is empty");
        if (head_ == tail_) {
            T value = std::move(head_->value);
            delete head_;
            head_ = nullptr;
            tail_ = nullptr;
            size_ = 0;
            return value;
        }
        Node* prev = head_;
        while (prev->next != tail_)
            prev = prev->next;
        T value = std::move(tail_->value);
        delete tail_;
        tail_ = prev;
        tail_->next = nullptr;
        --size_;
        return value;
    }

    T& front() {
        if (head_ == nullptr)
            throw std::out_of_range("LinkedList::front: list is empty");
        return head_->value;
    }

    const T& front() const {
        if (head_ == nullptr)
            throw std::out_of_range("LinkedList::front: list is empty");
        return head_->value;
    }

    T& back() {
        if (tail_ == nullptr)
            throw std::out_of_range("LinkedList::back: list is empty");
        return tail_->value;
    }

    const T& back() const {
        if (tail_ == nullptr)
            throw std::out_of_range("LinkedList::back: list is empty");
        return tail_->value;
    }

    T& at(size_type index) {
        if (index >= size_)
            throw std::out_of_range("LinkedList::at: index out of range");
        Node* curr = head_;
        for (size_type i = 0; i < index; ++i)
            curr = curr->next;
        return curr->value;
    }

    const T& at(size_type index) const {
        if (index >= size_)
            throw std::out_of_range("LinkedList::at: index out of range");
        Node* curr = head_;
        for (size_type i = 0; i < index; ++i)
            curr = curr->next;
        return curr->value;
    }

    void insert_at(size_type index, const T& value) {
        if (index > size_)
            throw std::out_of_range("LinkedList::insert_at: index out of range");
        if (index == 0) {
            push_front(value);
            return;
        }
        if (index == size_) {
            push_back(value);
            return;
        }
        Node* prev = head_;
        for (size_type i = 0; i < index - 1; ++i)
            prev = prev->next;
        Node* node = new Node(value);
        node->next = prev->next;
        prev->next = node;
        ++size_;
    }

    T remove_at(size_type index) {
        if (index >= size_)
            throw std::out_of_range("LinkedList::remove_at: index out of range");
        if (index == 0)
            return pop_front();
        if (index == size_ - 1)
            return pop_back();
        Node* prev = head_;
        for (size_type i = 0; i < index - 1; ++i)
            prev = prev->next;
        Node* node = prev->next;
        T value = std::move(node->value);
        prev->next = node->next;
        delete node;
        --size_;
        return value;
    }

    size_type size() const { return size_; }
    bool empty() const { return size_ == 0; }

    void clear() {
        while (head_ != nullptr) {
            Node* next = head_->next;
            delete head_;
            head_ = next;
        }
        tail_ = nullptr;
        size_ = 0;
    }

    Iterator begin() { return Iterator(head_); }
    Iterator end() { return Iterator(nullptr); }
    ConstIterator begin() const { return ConstIterator(head_); }
    ConstIterator end() const { return ConstIterator(nullptr); }

private:
    Node* head_;
    Node* tail_;
    size_type size_;
};
