#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

template <typename T>
class BinarySearchTree {
public:
    using value_type = T;
    using size_type = std::size_t;

    BinarySearchTree() : root_(nullptr), size_(0) {}

    BinarySearchTree(const BinarySearchTree& other) : root_(nullptr), size_(0) {
        copy_tree(other.root_);
    }

    BinarySearchTree(BinarySearchTree&& other) noexcept
        : root_(other.root_), size_(other.size_) {
        other.root_ = nullptr;
        other.size_ = 0;
    }

    BinarySearchTree& operator=(const BinarySearchTree& other) {
        if (this != &other) {
            BinarySearchTree temp(other);
            swap(temp);
        }
        return *this;
    }

    BinarySearchTree& operator=(BinarySearchTree&& other) noexcept {
        if (this != &other) {
            clear();
            root_ = other.root_;
            size_ = other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~BinarySearchTree() { clear(); }

    void swap(BinarySearchTree& other) noexcept {
        std::swap(root_, other.root_);
        std::swap(size_, other.size_);
    }

    void insert(const T& value) {
        root_ = insert_node(root_, value);
    }

    void remove(const T& value) {
        root_ = remove_node(root_, value);
    }

    bool contains(const T& value) const {
        return find_node(root_, value) != nullptr;
    }

    const T& min() const {
        if (root_ == nullptr)
            throw std::out_of_range("BinarySearchTree::min: tree is empty");
        return find_min(root_)->value;
    }

    const T& max() const {
        if (root_ == nullptr)
            throw std::out_of_range("BinarySearchTree::max: tree is empty");
        return find_max(root_)->value;
    }

    size_type size() const { return size_; }
    bool empty() const { return size_ == 0; }

    void clear() {
        destroy_tree(root_);
        root_ = nullptr;
        size_ = 0;
    }

    template <typename Container>
    void in_order(Container& result) const {
        in_order_traverse(root_, result);
    }

    template <typename Container>
    void pre_order(Container& result) const {
        pre_order_traverse(root_, result);
    }

    template <typename Container>
    void post_order(Container& result) const {
        post_order_traverse(root_, result);
    }

private:
    struct Node {
        T value;
        Node* left;
        Node* right;

        explicit Node(const T& v) : value(v), left(nullptr), right(nullptr) {}
    };

    Node* root_;
    size_type size_;

    Node* insert_node(Node* node, const T& value) {
        if (node == nullptr) {
            ++size_;
            return new Node(value);
        }
        if (value < node->value)
            node->left = insert_node(node->left, value);
        else if (node->value < value)
            node->right = insert_node(node->right, value);
        return node;
    }

    Node* remove_node(Node* node, const T& value) {
        if (node == nullptr)
            return nullptr;

        if (value < node->value) {
            node->left = remove_node(node->left, value);
        } else if (node->value < value) {
            node->right = remove_node(node->right, value);
        } else {
            if (node->left == nullptr) {
                Node* right = node->right;
                delete node;
                --size_;
                return right;
            }
            if (node->right == nullptr) {
                Node* left = node->left;
                delete node;
                --size_;
                return left;
            }
            Node* successor = find_min(node->right);
            node->value = successor->value;
            node->right = remove_node(node->right, successor->value);
        }
        return node;
    }

    Node* find_node(Node* node, const T& value) const {
        while (node != nullptr) {
            if (value < node->value)
                node = node->left;
            else if (node->value < value)
                node = node->right;
            else
                return node;
        }
        return nullptr;
    }

    Node* find_min(Node* node) const {
        while (node->left != nullptr)
            node = node->left;
        return node;
    }

    Node* find_max(Node* node) const {
        while (node->right != nullptr)
            node = node->right;
        return node;
    }

    void destroy_tree(Node* node) {
        if (node == nullptr)
            return;
        destroy_tree(node->left);
        destroy_tree(node->right);
        delete node;
    }

    void copy_tree(Node* node) {
        if (node == nullptr)
            return;
        insert(node->value);
        copy_tree(node->left);
        copy_tree(node->right);
    }

    template <typename Container>
    void in_order_traverse(Node* node, Container& result) const {
        if (node == nullptr)
            return;
        in_order_traverse(node->left, result);
        result.push_back(node->value);
        in_order_traverse(node->right, result);
    }

    template <typename Container>
    void pre_order_traverse(Node* node, Container& result) const {
        if (node == nullptr)
            return;
        result.push_back(node->value);
        pre_order_traverse(node->left, result);
        pre_order_traverse(node->right, result);
    }

    template <typename Container>
    void post_order_traverse(Node* node, Container& result) const {
        if (node == nullptr)
            return;
        post_order_traverse(node->left, result);
        post_order_traverse(node->right, result);
        result.push_back(node->value);
    }
};
