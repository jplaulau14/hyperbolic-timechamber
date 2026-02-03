#include "avl_tree.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define TEST(name)                                                             \
    static void test_##name();                                                 \
    struct Register_##name {                                                   \
        Register_##name() { tests[count++] = {#name, test_##name}; }          \
    } reg_##name;                                                              \
    static void test_##name()

struct TestEntry {
    const char* name;
    void (*fn)();
};

static TestEntry tests[128];
static int count = 0;

template <typename T>
bool is_sorted(const std::vector<T>& v) {
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i] < v[i - 1])
            return false;
    }
    return true;
}

template <typename T>
bool verify_bst_property(const AVLTree<T>& avl) {
    std::vector<T> result;
    avl.in_order(result);
    return is_sorted(result);
}

template <typename T>
struct NodeInfo {
    T value;
    int height;
    int balance;
};

TEST(new_tree_is_empty) {
    AVLTree<int> avl;
    assert(avl.size() == 0);
    assert(avl.empty());
    assert(avl.height() == 0);
}

TEST(min_on_empty_throws) {
    AVLTree<int> avl;
    bool threw = false;
    try {
        avl.min();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(max_on_empty_throws) {
    AVLTree<int> avl;
    bool threw = false;
    try {
        avl.max();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(insert_single_element) {
    AVLTree<int> avl;
    avl.insert(42);
    assert(avl.size() == 1);
    assert(!avl.empty());
    assert(avl.contains(42));
    assert(avl.height() == 1);
}

TEST(insert_multiple_elements) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    assert(avl.size() == 3);
    assert(avl.contains(50));
    assert(avl.contains(30));
    assert(avl.contains(70));
}

TEST(insert_duplicate_is_noop) {
    AVLTree<int> avl;
    avl.insert(42);
    avl.insert(42);
    assert(avl.size() == 1);
}

TEST(insert_maintains_bst_property) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.insert(20);
    avl.insert(40);
    avl.insert(60);
    avl.insert(80);
    assert(verify_bst_property(avl));
}

TEST(ll_imbalance_triggers_right_rotation) {
    AVLTree<int> avl;
    avl.insert(30);
    avl.insert(20);
    avl.insert(10);
    assert(avl.height() == 2);
    std::vector<int> result;
    avl.in_order(result);
    std::vector<int> expected = {10, 20, 30};
    assert(result == expected);
}

TEST(rr_imbalance_triggers_left_rotation) {
    AVLTree<int> avl;
    avl.insert(10);
    avl.insert(20);
    avl.insert(30);
    assert(avl.height() == 2);
    std::vector<int> result;
    avl.in_order(result);
    std::vector<int> expected = {10, 20, 30};
    assert(result == expected);
}

TEST(lr_imbalance_triggers_left_right_rotation) {
    AVLTree<int> avl;
    avl.insert(30);
    avl.insert(10);
    avl.insert(20);
    assert(avl.height() == 2);
    std::vector<int> result;
    avl.in_order(result);
    std::vector<int> expected = {10, 20, 30};
    assert(result == expected);
}

TEST(rl_imbalance_triggers_right_left_rotation) {
    AVLTree<int> avl;
    avl.insert(10);
    avl.insert(30);
    avl.insert(20);
    assert(avl.height() == 2);
    std::vector<int> result;
    avl.in_order(result);
    std::vector<int> expected = {10, 20, 30};
    assert(result == expected);
}

TEST(contains_returns_true_for_existing) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    assert(avl.contains(50));
    assert(avl.contains(30));
    assert(avl.contains(70));
}

TEST(contains_returns_false_for_nonexistent) {
    AVLTree<int> avl;
    avl.insert(50);
    assert(!avl.contains(30));
    assert(!avl.contains(70));
}

TEST(contains_after_insert) {
    AVLTree<int> avl;
    assert(!avl.contains(42));
    avl.insert(42);
    assert(avl.contains(42));
}

TEST(contains_after_remove) {
    AVLTree<int> avl;
    avl.insert(42);
    assert(avl.contains(42));
    avl.remove(42);
    assert(!avl.contains(42));
}

TEST(remove_leaf_node) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.remove(30);
    assert(!avl.contains(30));
    assert(avl.contains(50));
    assert(avl.contains(70));
    assert(avl.size() == 2);
}

TEST(remove_node_with_left_child) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.insert(20);
    avl.remove(30);
    assert(!avl.contains(30));
    assert(avl.contains(50));
    assert(avl.contains(20));
    assert(verify_bst_property(avl));
}

TEST(remove_node_with_right_child) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.insert(40);
    avl.remove(30);
    assert(!avl.contains(30));
    assert(avl.contains(50));
    assert(avl.contains(40));
    assert(verify_bst_property(avl));
}

TEST(remove_node_with_two_children) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.insert(20);
    avl.insert(40);
    avl.remove(30);
    assert(!avl.contains(30));
    assert(avl.contains(50));
    assert(avl.contains(70));
    assert(avl.contains(20));
    assert(avl.contains(40));
    assert(verify_bst_property(avl));
}

TEST(remove_root_node) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.remove(50);
    assert(!avl.contains(50));
    assert(avl.contains(30));
    assert(avl.contains(70));
    assert(verify_bst_property(avl));
}

TEST(remove_nonexistent_is_noop) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.remove(100);
    assert(avl.size() == 1);
    assert(avl.contains(50));
}

TEST(size_decrements_after_remove) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    assert(avl.size() == 3);
    avl.remove(30);
    assert(avl.size() == 2);
    avl.remove(70);
    assert(avl.size() == 1);
}

TEST(remove_triggers_rebalancing) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.insert(60);
    avl.insert(80);
    avl.remove(30);
    assert(verify_bst_property(avl));
    assert(avl.height() <= 3);
}

TEST(min_returns_smallest) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.insert(20);
    assert(avl.min() == 20);
}

TEST(max_returns_largest) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.insert(80);
    assert(avl.max() == 80);
}

TEST(min_max_after_insert) {
    AVLTree<int> avl;
    avl.insert(50);
    assert(avl.min() == 50);
    assert(avl.max() == 50);
    avl.insert(30);
    assert(avl.min() == 30);
    avl.insert(70);
    assert(avl.max() == 70);
}

TEST(min_max_after_remove) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.remove(30);
    assert(avl.min() == 50);
    avl.remove(70);
    assert(avl.max() == 50);
}

TEST(height_of_empty_tree) {
    AVLTree<int> avl;
    assert(avl.height() == 0);
}

TEST(height_of_single_node) {
    AVLTree<int> avl;
    avl.insert(42);
    assert(avl.height() == 1);
}

TEST(height_updates_after_insert) {
    AVLTree<int> avl;
    avl.insert(50);
    assert(avl.height() == 1);
    avl.insert(30);
    assert(avl.height() == 2);
    avl.insert(70);
    assert(avl.height() == 2);
    avl.insert(20);
    assert(avl.height() == 3);
}

TEST(height_updates_after_remove) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.insert(20);
    assert(avl.height() == 3);
    avl.remove(20);
    assert(avl.height() == 2);
}

TEST(height_is_log_n_for_sorted_insert) {
    AVLTree<int> avl;
    for (int i = 1; i <= 1000; ++i)
        avl.insert(i);
    double max_height = 1.44 * std::log2(1001);
    assert(avl.height() <= static_cast<std::size_t>(max_height));
}

TEST(in_order_yields_sorted) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.insert(20);
    avl.insert(40);
    std::vector<int> result;
    avl.in_order(result);
    std::vector<int> expected = {20, 30, 40, 50, 70};
    assert(result == expected);
}

TEST(pre_order_yields_correct_sequence) {
    AVLTree<int> avl;
    avl.insert(20);
    avl.insert(10);
    avl.insert(30);
    std::vector<int> result;
    avl.pre_order(result);
    std::vector<int> expected = {20, 10, 30};
    assert(result == expected);
}

TEST(post_order_yields_correct_sequence) {
    AVLTree<int> avl;
    avl.insert(20);
    avl.insert(10);
    avl.insert(30);
    std::vector<int> result;
    avl.post_order(result);
    std::vector<int> expected = {10, 30, 20};
    assert(result == expected);
}

TEST(traversals_on_empty_tree) {
    AVLTree<int> avl;
    std::vector<int> in_result, pre_result, post_result;
    avl.in_order(in_result);
    avl.pre_order(pre_result);
    avl.post_order(post_result);
    assert(in_result.empty());
    assert(pre_result.empty());
    assert(post_result.empty());
}

TEST(clear_makes_tree_empty) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.clear();
    assert(avl.empty());
    assert(avl.size() == 0);
    assert(avl.height() == 0);
}

TEST(clear_on_empty_is_noop) {
    AVLTree<int> avl;
    avl.clear();
    assert(avl.empty());
}

TEST(bst_property_holds_after_operations) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.insert(20);
    avl.insert(40);
    assert(verify_bst_property(avl));
    avl.remove(30);
    assert(verify_bst_property(avl));
    avl.insert(35);
    assert(verify_bst_property(avl));
}

TEST(copy_creates_independent_copy) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    AVLTree<int> copy(avl);
    assert(copy.size() == 3);
    assert(copy.contains(50));
    assert(copy.contains(30));
    assert(copy.contains(70));
}

TEST(insert_to_original_doesnt_affect_copy) {
    AVLTree<int> avl;
    avl.insert(50);
    AVLTree<int> copy(avl);
    avl.insert(30);
    assert(avl.size() == 2);
    assert(copy.size() == 1);
    assert(!copy.contains(30));
}

TEST(remove_from_original_doesnt_affect_copy) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    AVLTree<int> copy(avl);
    avl.remove(30);
    assert(!avl.contains(30));
    assert(copy.contains(30));
}

TEST(single_element_tree) {
    AVLTree<int> avl;
    avl.insert(42);
    assert(avl.min() == 42);
    assert(avl.max() == 42);
    assert(avl.contains(42));
    avl.remove(42);
    assert(avl.empty());
}

TEST(sorted_insert_produces_balanced_tree) {
    AVLTree<int> avl;
    for (int i = 1; i <= 10; ++i)
        avl.insert(i);
    assert(avl.size() == 10);
    assert(avl.min() == 1);
    assert(avl.max() == 10);
    assert(verify_bst_property(avl));
    assert(avl.height() <= 4);
}

TEST(reverse_sorted_insert_produces_balanced_tree) {
    AVLTree<int> avl;
    for (int i = 10; i >= 1; --i)
        avl.insert(i);
    assert(avl.size() == 10);
    assert(avl.min() == 1);
    assert(avl.max() == 10);
    assert(verify_bst_property(avl));
    assert(avl.height() <= 4);
}

TEST(large_number_of_elements) {
    AVLTree<int> avl;
    for (int i = 1; i <= 1000; ++i)
        avl.insert(i);
    assert(avl.size() == 1000);
    assert(avl.min() == 1);
    assert(avl.max() == 1000);
    for (int i = 1; i <= 1000; ++i)
        assert(avl.contains(i));
    double max_height = 1.44 * std::log2(1001);
    assert(avl.height() <= static_cast<std::size_t>(max_height));
}

TEST(negative_numbers) {
    AVLTree<int> avl;
    avl.insert(-10);
    avl.insert(0);
    avl.insert(10);
    avl.insert(-20);
    assert(avl.min() == -20);
    assert(avl.max() == 10);
    assert(avl.contains(-10));
    assert(verify_bst_property(avl));
}

TEST(remove_all_elements_one_by_one) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    avl.insert(20);
    avl.insert(40);
    avl.remove(20);
    avl.remove(40);
    avl.remove(30);
    avl.remove(70);
    avl.remove(50);
    assert(avl.empty());
}

TEST(alternating_insert_remove_maintains_balance) {
    AVLTree<int> avl;
    for (int i = 1; i <= 100; ++i) {
        avl.insert(i);
        if (i % 3 == 0)
            avl.remove(i - 1);
    }
    assert(verify_bst_property(avl));
    double max_height = 1.44 * std::log2(avl.size() + 1);
    assert(avl.height() <= static_cast<std::size_t>(max_height) + 1);
}

TEST(move_constructor) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    avl.insert(70);
    AVLTree<int> moved(std::move(avl));
    assert(moved.size() == 3);
    assert(moved.contains(50));
    assert(avl.empty());
}

TEST(move_assignment) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    AVLTree<int> other;
    other.insert(99);
    other = std::move(avl);
    assert(other.size() == 2);
    assert(other.contains(50));
    assert(avl.empty());
}

TEST(copy_assignment) {
    AVLTree<int> avl;
    avl.insert(50);
    avl.insert(30);
    AVLTree<int> other;
    other.insert(99);
    other = avl;
    assert(other.size() == 2);
    assert(other.contains(50));
    avl.insert(70);
    assert(!other.contains(70));
}

TEST(works_with_strings) {
    AVLTree<std::string> avl;
    avl.insert("banana");
    avl.insert("apple");
    avl.insert("cherry");
    assert(avl.min() == "apple");
    assert(avl.max() == "cherry");
    std::vector<std::string> result;
    avl.in_order(result);
    std::vector<std::string> expected = {"apple", "banana", "cherry"};
    assert(result == expected);
}

TEST(works_with_doubles) {
    AVLTree<double> avl;
    avl.insert(3.14);
    avl.insert(1.41);
    avl.insert(2.71);
    assert(avl.min() == 1.41);
    assert(avl.max() == 3.14);
}

int main() {
    int passed = 0;
    int failed = 0;
    for (int i = 0; i < count; ++i) {
        try {
            tests[i].fn();
            std::printf("  PASS  %s\n", tests[i].name);
            ++passed;
        } catch (const std::exception& e) {
            std::printf("  FAIL  %s: %s\n", tests[i].name, e.what());
            ++failed;
        } catch (...) {
            std::printf("  FAIL  %s: unknown exception\n", tests[i].name);
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
