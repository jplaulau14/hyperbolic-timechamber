#include "binary_heap.hpp"

#include <cassert>
#include <cstdio>
#include <string>

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

static TestEntry tests[64];
static int count = 0;

// --- Construction & Basic State ---

TEST(new_heap_is_empty) {
    BinaryHeap<int> heap;
    assert(heap.size() == 0);
    assert(heap.empty());
}

TEST(peek_on_empty_throws) {
    BinaryHeap<int> heap;
    bool threw = false;
    try {
        heap.peek();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(pop_on_empty_throws) {
    BinaryHeap<int> heap;
    bool threw = false;
    try {
        heap.pop();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

// --- Push Operations ---

TEST(push_single_element) {
    BinaryHeap<int> heap;
    heap.push(42);
    assert(heap.size() == 1);
    assert(!heap.empty());
    assert(heap.peek() == 42);
}

TEST(push_multiple_maintains_heap_property) {
    BinaryHeap<int> heap;
    heap.push(30);
    heap.push(10);
    heap.push(20);
    assert(heap.peek() == 10);
}

TEST(push_ascending_order) {
    BinaryHeap<int> heap;
    heap.push(1);
    heap.push(2);
    heap.push(3);
    heap.push(4);
    heap.push(5);
    assert(heap.peek() == 1);
}

TEST(push_descending_order) {
    BinaryHeap<int> heap;
    heap.push(5);
    heap.push(4);
    heap.push(3);
    heap.push(2);
    heap.push(1);
    assert(heap.peek() == 1);
}

TEST(push_random_order) {
    BinaryHeap<int> heap;
    heap.push(15);
    heap.push(3);
    heap.push(27);
    heap.push(8);
    heap.push(1);
    heap.push(12);
    assert(heap.peek() == 1);
}

// --- Pop Operations ---

TEST(pop_returns_minimum) {
    BinaryHeap<int> heap;
    heap.push(30);
    heap.push(10);
    heap.push(20);
    assert(heap.pop() == 10);
}

TEST(pop_restores_heap_property) {
    BinaryHeap<int> heap;
    heap.push(30);
    heap.push(10);
    heap.push(20);
    heap.pop();
    assert(heap.peek() == 20);
}

TEST(pop_all_yields_sorted_order) {
    BinaryHeap<int> heap;
    heap.push(5);
    heap.push(3);
    heap.push(8);
    heap.push(1);
    heap.push(4);
    assert(heap.pop() == 1);
    assert(heap.pop() == 3);
    assert(heap.pop() == 4);
    assert(heap.pop() == 5);
    assert(heap.pop() == 8);
    assert(heap.empty());
}

TEST(size_decrements_after_pop) {
    BinaryHeap<int> heap;
    heap.push(1);
    heap.push(2);
    heap.push(3);
    assert(heap.size() == 3);
    heap.pop();
    assert(heap.size() == 2);
    heap.pop();
    assert(heap.size() == 1);
}

// --- Peek Operations ---

TEST(peek_returns_minimum_without_removing) {
    BinaryHeap<int> heap;
    heap.push(30);
    heap.push(10);
    heap.push(20);
    assert(heap.peek() == 10);
    assert(heap.size() == 3);
}

TEST(multiple_peeks_same_value) {
    BinaryHeap<int> heap;
    heap.push(42);
    assert(heap.peek() == 42);
    assert(heap.peek() == 42);
    assert(heap.peek() == 42);
    assert(heap.size() == 1);
}

TEST(peek_after_push_shows_new_min) {
    BinaryHeap<int> heap;
    heap.push(20);
    assert(heap.peek() == 20);
    heap.push(10);
    assert(heap.peek() == 10);
    heap.push(5);
    assert(heap.peek() == 5);
}

// --- Heapify / From Array ---

TEST(from_array_unsorted) {
    int arr[] = {5, 3, 8, 1, 4, 7, 2};
    BinaryHeap<int> heap = BinaryHeap<int>::from_array(arr, 7);
    assert(heap.size() == 7);
    assert(heap.peek() == 1);
}

TEST(from_array_correct_min_at_top) {
    int arr[] = {100, 50, 75, 25, 10};
    BinaryHeap<int> heap = BinaryHeap<int>::from_array(arr, 5);
    assert(heap.peek() == 10);
}

TEST(from_array_all_elements_present) {
    int arr[] = {5, 3, 8, 1, 4};
    BinaryHeap<int> heap = BinaryHeap<int>::from_array(arr, 5);
    assert(heap.pop() == 1);
    assert(heap.pop() == 3);
    assert(heap.pop() == 4);
    assert(heap.pop() == 5);
    assert(heap.pop() == 8);
}

TEST(from_array_empty) {
    BinaryHeap<int> heap = BinaryHeap<int>::from_array(nullptr, 0);
    assert(heap.empty());
}

// --- Clear ---

TEST(clear_makes_heap_empty) {
    BinaryHeap<int> heap;
    heap.push(1);
    heap.push(2);
    heap.push(3);
    heap.clear();
    assert(heap.empty());
    assert(heap.size() == 0);
}

TEST(clear_on_empty_is_noop) {
    BinaryHeap<int> heap;
    heap.clear();
    assert(heap.empty());
}

// --- Heap Property Verification ---

TEST(heap_property_after_operations) {
    BinaryHeap<int> heap;
    heap.push(10);
    heap.push(5);
    heap.push(15);
    heap.push(3);
    heap.push(8);
    heap.pop();
    heap.push(1);
    assert(heap.peek() == 1);
    assert(heap.pop() == 1);
    assert(heap.pop() == 5);
}

TEST(heap_property_with_duplicates) {
    BinaryHeap<int> heap;
    heap.push(5);
    heap.push(5);
    heap.push(3);
    heap.push(3);
    heap.push(1);
    heap.push(1);
    assert(heap.pop() == 1);
    assert(heap.pop() == 1);
    assert(heap.pop() == 3);
    assert(heap.pop() == 3);
    assert(heap.pop() == 5);
    assert(heap.pop() == 5);
}

TEST(heap_property_with_negatives) {
    BinaryHeap<int> heap;
    heap.push(5);
    heap.push(-3);
    heap.push(0);
    heap.push(-10);
    heap.push(7);
    assert(heap.pop() == -10);
    assert(heap.pop() == -3);
    assert(heap.pop() == 0);
}

// --- Copy/Clone ---

TEST(copy_creates_independent_copy) {
    BinaryHeap<int> heap;
    heap.push(30);
    heap.push(10);
    heap.push(20);
    BinaryHeap<int> copy(heap);
    assert(copy.size() == 3);
    assert(copy.peek() == 10);
}

TEST(push_to_original_doesnt_affect_copy) {
    BinaryHeap<int> heap;
    heap.push(30);
    heap.push(10);
    BinaryHeap<int> copy(heap);
    heap.push(5);
    assert(heap.peek() == 5);
    assert(copy.peek() == 10);
    assert(heap.size() == 3);
    assert(copy.size() == 2);
}

TEST(pop_from_original_doesnt_affect_copy) {
    BinaryHeap<int> heap;
    heap.push(30);
    heap.push(10);
    heap.push(20);
    BinaryHeap<int> copy(heap);
    heap.pop();
    assert(heap.peek() == 20);
    assert(copy.peek() == 10);
    assert(copy.size() == 3);
}

// --- Non-trivial Types ---

TEST(works_with_floats) {
    BinaryHeap<double> heap;
    heap.push(3.14);
    heap.push(1.41);
    heap.push(2.71);
    assert(heap.pop() == 1.41);
    assert(heap.pop() == 2.71);
    assert(heap.pop() == 3.14);
}

TEST(works_with_strings) {
    BinaryHeap<std::string> heap;
    heap.push("charlie");
    heap.push("alpha");
    heap.push("bravo");
    assert(heap.pop() == "alpha");
    assert(heap.pop() == "bravo");
    assert(heap.pop() == "charlie");
}

// --- Edge Cases ---

TEST(single_element_heap) {
    BinaryHeap<int> heap;
    heap.push(42);
    assert(heap.size() == 1);
    assert(heap.peek() == 42);
    assert(heap.pop() == 42);
    assert(heap.empty());
}

TEST(two_element_heap) {
    BinaryHeap<int> heap;
    heap.push(20);
    heap.push(10);
    assert(heap.peek() == 10);
    assert(heap.pop() == 10);
    assert(heap.peek() == 20);
    assert(heap.pop() == 20);
    assert(heap.empty());
}

TEST(large_number_of_elements) {
    BinaryHeap<int> heap;
    for (int i = 1000; i >= 1; --i)
        heap.push(i);
    assert(heap.size() == 1000);
    for (int i = 1; i <= 1000; ++i)
        assert(heap.pop() == i);
    assert(heap.empty());
}

TEST(many_push_pop_cycles) {
    BinaryHeap<int> heap;
    for (int cycle = 0; cycle < 100; ++cycle) {
        for (int i = 0; i < 10; ++i)
            heap.push(cycle * 10 + i);
        for (int i = 0; i < 5; ++i)
            heap.pop();
    }
    assert(heap.size() == 500);
}

// --- Move Operations ---

TEST(move_constructor) {
    BinaryHeap<int> heap;
    heap.push(30);
    heap.push(10);
    heap.push(20);
    BinaryHeap<int> moved(std::move(heap));
    assert(moved.size() == 3);
    assert(moved.peek() == 10);
    assert(heap.empty());
}

TEST(move_assignment) {
    BinaryHeap<int> heap;
    heap.push(30);
    heap.push(10);
    BinaryHeap<int> other;
    other.push(99);
    other = std::move(heap);
    assert(other.size() == 2);
    assert(other.peek() == 10);
    assert(heap.empty());
}

TEST(copy_assignment) {
    BinaryHeap<int> heap;
    heap.push(30);
    heap.push(10);
    BinaryHeap<int> other;
    other.push(99);
    other = heap;
    assert(other.size() == 2);
    assert(other.peek() == 10);
    heap.push(5);
    assert(other.peek() == 10);
}

TEST(const_peek) {
    BinaryHeap<int> heap;
    heap.push(42);
    const BinaryHeap<int>& cref = heap;
    assert(cref.peek() == 42);
}

TEST(push_rvalue) {
    BinaryHeap<std::string> heap;
    std::string str = "test";
    heap.push(std::move(str));
    assert(heap.peek() == "test");
}

// --- Runner ---

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
