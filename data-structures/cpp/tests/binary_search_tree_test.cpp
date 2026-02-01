#include "binary_search_tree.hpp"

#include <cassert>
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
bool verify_bst_property(const BinarySearchTree<T>& bst) {
    std::vector<T> result;
    bst.in_order(result);
    return is_sorted(result);
}

TEST(new_tree_is_empty) {
    BinarySearchTree<int> bst;
    assert(bst.size() == 0);
    assert(bst.empty());
}

TEST(min_on_empty_throws) {
    BinarySearchTree<int> bst;
    bool threw = false;
    try {
        bst.min();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(max_on_empty_throws) {
    BinarySearchTree<int> bst;
    bool threw = false;
    try {
        bst.max();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(insert_single_element) {
    BinarySearchTree<int> bst;
    bst.insert(42);
    assert(bst.size() == 1);
    assert(!bst.empty());
    assert(bst.contains(42));
}

TEST(insert_multiple_elements) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    assert(bst.size() == 3);
    assert(bst.contains(50));
    assert(bst.contains(30));
    assert(bst.contains(70));
}

TEST(insert_duplicate_is_noop) {
    BinarySearchTree<int> bst;
    bst.insert(42);
    bst.insert(42);
    assert(bst.size() == 1);
}

TEST(insert_maintains_bst_property) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);
    bst.insert(60);
    bst.insert(80);
    assert(verify_bst_property(bst));
}

TEST(contains_returns_true_for_existing) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    assert(bst.contains(50));
    assert(bst.contains(30));
    assert(bst.contains(70));
}

TEST(contains_returns_false_for_nonexistent) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    assert(!bst.contains(30));
    assert(!bst.contains(70));
}

TEST(contains_after_insert) {
    BinarySearchTree<int> bst;
    assert(!bst.contains(42));
    bst.insert(42);
    assert(bst.contains(42));
}

TEST(contains_after_remove) {
    BinarySearchTree<int> bst;
    bst.insert(42);
    assert(bst.contains(42));
    bst.remove(42);
    assert(!bst.contains(42));
}

TEST(remove_leaf_node) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.remove(30);
    assert(!bst.contains(30));
    assert(bst.contains(50));
    assert(bst.contains(70));
    assert(bst.size() == 2);
}

TEST(remove_node_with_left_child) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(20);
    bst.remove(30);
    assert(!bst.contains(30));
    assert(bst.contains(50));
    assert(bst.contains(20));
    assert(verify_bst_property(bst));
}

TEST(remove_node_with_right_child) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(40);
    bst.remove(30);
    assert(!bst.contains(30));
    assert(bst.contains(50));
    assert(bst.contains(40));
    assert(verify_bst_property(bst));
}

TEST(remove_node_with_two_children) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);
    bst.remove(30);
    assert(!bst.contains(30));
    assert(bst.contains(50));
    assert(bst.contains(70));
    assert(bst.contains(20));
    assert(bst.contains(40));
    assert(verify_bst_property(bst));
}

TEST(remove_root_node) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.remove(50);
    assert(!bst.contains(50));
    assert(bst.contains(30));
    assert(bst.contains(70));
    assert(verify_bst_property(bst));
}

TEST(remove_nonexistent_is_noop) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.remove(100);
    assert(bst.size() == 1);
    assert(bst.contains(50));
}

TEST(size_decrements_after_remove) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    assert(bst.size() == 3);
    bst.remove(30);
    assert(bst.size() == 2);
    bst.remove(70);
    assert(bst.size() == 1);
}

TEST(min_returns_smallest) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    assert(bst.min() == 20);
}

TEST(max_returns_largest) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(80);
    assert(bst.max() == 80);
}

TEST(min_max_after_insert) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    assert(bst.min() == 50);
    assert(bst.max() == 50);
    bst.insert(30);
    assert(bst.min() == 30);
    bst.insert(70);
    assert(bst.max() == 70);
}

TEST(min_max_after_remove) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.remove(30);
    assert(bst.min() == 50);
    bst.remove(70);
    assert(bst.max() == 50);
}

TEST(in_order_yields_sorted) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);
    std::vector<int> result;
    bst.in_order(result);
    std::vector<int> expected = {20, 30, 40, 50, 70};
    assert(result == expected);
}

TEST(pre_order_yields_correct_sequence) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);
    std::vector<int> result;
    bst.pre_order(result);
    std::vector<int> expected = {50, 30, 20, 40, 70};
    assert(result == expected);
}

TEST(post_order_yields_correct_sequence) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);
    std::vector<int> result;
    bst.post_order(result);
    std::vector<int> expected = {20, 40, 30, 70, 50};
    assert(result == expected);
}

TEST(traversals_on_empty_tree) {
    BinarySearchTree<int> bst;
    std::vector<int> in_result, pre_result, post_result;
    bst.in_order(in_result);
    bst.pre_order(pre_result);
    bst.post_order(post_result);
    assert(in_result.empty());
    assert(pre_result.empty());
    assert(post_result.empty());
}

TEST(clear_makes_tree_empty) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.clear();
    assert(bst.empty());
    assert(bst.size() == 0);
}

TEST(clear_on_empty_is_noop) {
    BinarySearchTree<int> bst;
    bst.clear();
    assert(bst.empty());
}

TEST(bst_property_holds_after_operations) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);
    assert(verify_bst_property(bst));
    bst.remove(30);
    assert(verify_bst_property(bst));
    bst.insert(35);
    assert(verify_bst_property(bst));
}

TEST(copy_creates_independent_copy) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    BinarySearchTree<int> copy(bst);
    assert(copy.size() == 3);
    assert(copy.contains(50));
    assert(copy.contains(30));
    assert(copy.contains(70));
}

TEST(insert_to_original_doesnt_affect_copy) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    BinarySearchTree<int> copy(bst);
    bst.insert(30);
    assert(bst.size() == 2);
    assert(copy.size() == 1);
    assert(!copy.contains(30));
}

TEST(remove_from_original_doesnt_affect_copy) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    BinarySearchTree<int> copy(bst);
    bst.remove(30);
    assert(!bst.contains(30));
    assert(copy.contains(30));
}

TEST(single_element_tree) {
    BinarySearchTree<int> bst;
    bst.insert(42);
    assert(bst.min() == 42);
    assert(bst.max() == 42);
    assert(bst.contains(42));
    bst.remove(42);
    assert(bst.empty());
}

TEST(sorted_insert_degenerates_to_list) {
    BinarySearchTree<int> bst;
    for (int i = 1; i <= 10; ++i)
        bst.insert(i);
    assert(bst.size() == 10);
    assert(bst.min() == 1);
    assert(bst.max() == 10);
    assert(verify_bst_property(bst));
}

TEST(large_number_of_elements) {
    BinarySearchTree<int> bst;
    for (int i = 1; i <= 1000; ++i)
        bst.insert(i);
    assert(bst.size() == 1000);
    assert(bst.min() == 1);
    assert(bst.max() == 1000);
    for (int i = 1; i <= 1000; ++i)
        assert(bst.contains(i));
}

TEST(negative_numbers) {
    BinarySearchTree<int> bst;
    bst.insert(-10);
    bst.insert(0);
    bst.insert(10);
    bst.insert(-20);
    assert(bst.min() == -20);
    assert(bst.max() == 10);
    assert(bst.contains(-10));
    assert(verify_bst_property(bst));
}

TEST(remove_all_elements_one_by_one) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);
    bst.remove(20);
    bst.remove(40);
    bst.remove(30);
    bst.remove(70);
    bst.remove(50);
    assert(bst.empty());
}

TEST(move_constructor) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    BinarySearchTree<int> moved(std::move(bst));
    assert(moved.size() == 3);
    assert(moved.contains(50));
    assert(bst.empty());
}

TEST(move_assignment) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    BinarySearchTree<int> other;
    other.insert(99);
    other = std::move(bst);
    assert(other.size() == 2);
    assert(other.contains(50));
    assert(bst.empty());
}

TEST(copy_assignment) {
    BinarySearchTree<int> bst;
    bst.insert(50);
    bst.insert(30);
    BinarySearchTree<int> other;
    other.insert(99);
    other = bst;
    assert(other.size() == 2);
    assert(other.contains(50));
    bst.insert(70);
    assert(!other.contains(70));
}

TEST(works_with_strings) {
    BinarySearchTree<std::string> bst;
    bst.insert("banana");
    bst.insert("apple");
    bst.insert("cherry");
    assert(bst.min() == "apple");
    assert(bst.max() == "cherry");
    std::vector<std::string> result;
    bst.in_order(result);
    std::vector<std::string> expected = {"apple", "banana", "cherry"};
    assert(result == expected);
}

TEST(works_with_doubles) {
    BinarySearchTree<double> bst;
    bst.insert(3.14);
    bst.insert(1.41);
    bst.insert(2.71);
    assert(bst.min() == 1.41);
    assert(bst.max() == 3.14);
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
