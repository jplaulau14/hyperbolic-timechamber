#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

template <typename T>
class AVLTree {
public:
    using value_type = T;
    using size_type = std::size_t;

    AVLTree() : root_(nullptr), size_(0) {}

    AVLTree(const AVLTree& other) : root_(nullptr), size_(0) {
        copy_tree(other.root_);
    }

    AVLTree(AVLTree&& other) noexcept
        : root_(other.root_), size_(other.size_) {
        other.root_ = nullptr;
        other.size_ = 0;
    }

    AVLTree& operator=(const AVLTree& other) {
        if (this != &other) {
            AVLTree temp(other);
            swap(temp);
        }
        return *this;
    }

    AVLTree& operator=(AVLTree&& other) noexcept {
        if (this != &other) {
            clear();
            root_ = other.root_;
            size_ = other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~AVLTree() { clear(); }

    void swap(AVLTree& other) noexcept {
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
            throw std::out_of_range("AVLTree::min: tree is empty");
        return find_min(root_)->value;
    }

    const T& max() const {
        if (root_ == nullptr)
            throw std::out_of_range("AVLTree::max: tree is empty");
        return find_max(root_)->value;
    }

    size_type size() const { return size_; }
    bool empty() const { return size_ == 0; }

    size_type height() const {
        return node_height(root_);
    }

    bool is_balanced() const {
        return check_balance(root_);
    }

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
        size_type height;

        explicit Node(const T& v)
            : value(v), left(nullptr), right(nullptr), height(1) {}
    };

    Node* root_;
    size_type size_;

    static size_type node_height(Node* node) {
        return node == nullptr ? 0 : node->height;
    }

    static void update_height(Node* node) {
        size_type left_h = node_height(node->left);
        size_type right_h = node_height(node->right);
        node->height = 1 + (left_h > right_h ? left_h : right_h);
    }

    static int balance_factor(Node* node) {
        if (node == nullptr)
            return 0;
        return static_cast<int>(node_height(node->left)) -
               static_cast<int>(node_height(node->right));
    }

    static Node* right_rotate(Node* y) {
        Node* x = y->left;
        Node* b = x->right;
        x->right = y;
        y->left = b;
        update_height(y);
        update_height(x);
        return x;
    }

    static Node* left_rotate(Node* x) {
        Node* y = x->right;
        Node* b = y->left;
        y->left = x;
        x->right = b;
        update_height(x);
        update_height(y);
        return y;
    }

    static Node* rebalance(Node* node) {
        update_height(node);
        int balance = balance_factor(node);

        if (balance > 1) {
            if (balance_factor(node->left) < 0)
                node->left = left_rotate(node->left);
            return right_rotate(node);
        }

        if (balance < -1) {
            if (balance_factor(node->right) > 0)
                node->right = right_rotate(node->right);
            return left_rotate(node);
        }

        return node;
    }

    Node* insert_node(Node* node, const T& value) {
        if (node == nullptr) {
            ++size_;
            return new Node(value);
        }

        if (value < node->value)
            node->left = insert_node(node->left, value);
        else if (node->value < value)
            node->right = insert_node(node->right, value);
        else
            return node;

        return rebalance(node);
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

        return rebalance(node);
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

    static Node* find_min(Node* node) {
        while (node->left != nullptr)
            node = node->left;
        return node;
    }

    static Node* find_max(Node* node) {
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

    static bool check_balance(Node* node) {
        if (node == nullptr)
            return true;
        int balance = balance_factor(node);
        if (balance < -1 || balance > 1)
            return false;
        return check_balance(node->left) && check_balance(node->right);
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
