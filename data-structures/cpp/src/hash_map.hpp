#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <functional>

template <typename K, typename V>
class HashMap {
private:
    struct Entry {
        K key;
        V value;
        Entry* next;
        Entry(const K& k, const V& v) : key(k), value(v), next(nullptr) {}
        Entry(K&& k, V&& v) : key(std::move(k)), value(std::move(v)), next(nullptr) {}
    };

public:
    using size_type = std::size_t;

    HashMap() : buckets_(new Entry*[16]()), capacity_(16), size_(0) {}

    HashMap(const HashMap& other) : buckets_(new Entry*[other.capacity_]()), capacity_(other.capacity_), size_(0) {
        for (size_type i = 0; i < capacity_; ++i) {
            for (Entry* e = other.buckets_[i]; e != nullptr; e = e->next)
                insert(e->key, e->value);
        }
    }

    HashMap(HashMap&& other) noexcept
        : buckets_(other.buckets_), capacity_(other.capacity_), size_(other.size_) {
        other.buckets_ = nullptr;
        other.capacity_ = 0;
        other.size_ = 0;
    }

    HashMap& operator=(const HashMap& other) {
        if (this != &other) {
            clear();
            delete[] buckets_;
            buckets_ = new Entry*[other.capacity_]();
            capacity_ = other.capacity_;
            for (size_type i = 0; i < capacity_; ++i) {
                for (Entry* e = other.buckets_[i]; e != nullptr; e = e->next)
                    insert(e->key, e->value);
            }
        }
        return *this;
    }

    HashMap& operator=(HashMap&& other) noexcept {
        if (this != &other) {
            clear();
            delete[] buckets_;
            buckets_ = other.buckets_;
            capacity_ = other.capacity_;
            size_ = other.size_;
            other.buckets_ = nullptr;
            other.capacity_ = 0;
            other.size_ = 0;
        }
        return *this;
    }

    ~HashMap() {
        if (buckets_ != nullptr) {
            clear();
            delete[] buckets_;
        }
    }

    void insert(const K& key, const V& value) {
        if (should_rehash())
            rehash();
        size_type idx = bucket_index(key);
        for (Entry* e = buckets_[idx]; e != nullptr; e = e->next) {
            if (e->key == key) {
                e->value = value;
                return;
            }
        }
        Entry* entry = new Entry(key, value);
        entry->next = buckets_[idx];
        buckets_[idx] = entry;
        ++size_;
    }

    void insert(K&& key, V&& value) {
        if (should_rehash())
            rehash();
        size_type idx = bucket_index(key);
        for (Entry* e = buckets_[idx]; e != nullptr; e = e->next) {
            if (e->key == key) {
                e->value = std::move(value);
                return;
            }
        }
        Entry* entry = new Entry(std::move(key), std::move(value));
        entry->next = buckets_[idx];
        buckets_[idx] = entry;
        ++size_;
    }

    V& get(const K& key) {
        Entry* e = find_entry(key);
        if (e == nullptr)
            throw std::out_of_range("HashMap::get: key not found");
        return e->value;
    }

    const V& get(const K& key) const {
        const Entry* e = find_entry(key);
        if (e == nullptr)
            throw std::out_of_range("HashMap::get: key not found");
        return e->value;
    }

    V* find(const K& key) {
        Entry* e = find_entry(key);
        return e ? &e->value : nullptr;
    }

    const V* find(const K& key) const {
        const Entry* e = find_entry(key);
        return e ? &e->value : nullptr;
    }

    bool remove(const K& key) {
        size_type idx = bucket_index(key);
        Entry* prev = nullptr;
        for (Entry* e = buckets_[idx]; e != nullptr; prev = e, e = e->next) {
            if (e->key == key) {
                if (prev == nullptr)
                    buckets_[idx] = e->next;
                else
                    prev->next = e->next;
                delete e;
                --size_;
                return true;
            }
        }
        return false;
    }

    bool contains(const K& key) const {
        return find_entry(key) != nullptr;
    }

    size_type size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_type capacity() const { return capacity_; }

    void clear() {
        if (buckets_ == nullptr)
            return;
        for (size_type i = 0; i < capacity_; ++i) {
            Entry* e = buckets_[i];
            while (e != nullptr) {
                Entry* next = e->next;
                delete e;
                e = next;
            }
            buckets_[i] = nullptr;
        }
        size_ = 0;
    }

    template <typename Container>
    void keys(Container& out) const {
        for (size_type i = 0; i < capacity_; ++i) {
            for (Entry* e = buckets_[i]; e != nullptr; e = e->next)
                out.push_back(e->key);
        }
    }

    template <typename Container>
    void values(Container& out) const {
        for (size_type i = 0; i < capacity_; ++i) {
            for (Entry* e = buckets_[i]; e != nullptr; e = e->next)
                out.push_back(e->value);
        }
    }

private:
    Entry** buckets_;
    size_type capacity_;
    size_type size_;

    size_type bucket_index(const K& key) const {
        return std::hash<K>{}(key) % capacity_;
    }

    bool should_rehash() const {
        return size_ * 4 > capacity_ * 3;
    }

    void rehash() {
        size_type new_capacity = capacity_ * 2;
        Entry** new_buckets = new Entry*[new_capacity]();
        for (size_type i = 0; i < capacity_; ++i) {
            Entry* e = buckets_[i];
            while (e != nullptr) {
                Entry* next = e->next;
                size_type idx = std::hash<K>{}(e->key) % new_capacity;
                e->next = new_buckets[idx];
                new_buckets[idx] = e;
                e = next;
            }
        }
        delete[] buckets_;
        buckets_ = new_buckets;
        capacity_ = new_capacity;
    }

    Entry* find_entry(const K& key) const {
        size_type idx = bucket_index(key);
        for (Entry* e = buckets_[idx]; e != nullptr; e = e->next) {
            if (e->key == key)
                return e;
        }
        return nullptr;
    }
};
