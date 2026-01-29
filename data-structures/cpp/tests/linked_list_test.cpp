#include "linked_list.hpp"

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

TEST(new_list_is_empty) {
    LinkedList<int> list;
    assert(list.size() == 0);
    assert(list.empty());
}

TEST(front_on_empty_throws) {
    LinkedList<int> list;
    bool threw = false;
    try {
        list.front();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(back_on_empty_throws) {
    LinkedList<int> list;
    bool threw = false;
    try {
        list.back();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

// --- Push Operations ---

TEST(push_front_single) {
    LinkedList<int> list;
    list.push_front(42);
    assert(list.size() == 1);
    assert(list.front() == 42);
    assert(list.back() == 42);
}

TEST(push_front_multiple_reverses_order) {
    LinkedList<int> list;
    list.push_front(1);
    list.push_front(2);
    list.push_front(3);
    assert(list.size() == 3);
    assert(list.front() == 3);
    assert(list.back() == 1);
}

TEST(push_back_single) {
    LinkedList<int> list;
    list.push_back(42);
    assert(list.size() == 1);
    assert(list.front() == 42);
    assert(list.back() == 42);
}

TEST(push_back_multiple_preserves_order) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    assert(list.size() == 3);
    assert(list.front() == 1);
    assert(list.back() == 3);
}

TEST(mixed_push_front_and_back) {
    LinkedList<int> list;
    list.push_back(2);
    list.push_front(1);
    list.push_back(3);
    list.push_front(0);
    assert(list.size() == 4);
    assert(list.at(0) == 0);
    assert(list.at(1) == 1);
    assert(list.at(2) == 2);
    assert(list.at(3) == 3);
}

// --- Pop Operations ---

TEST(pop_front_returns_value_and_decrements) {
    LinkedList<int> list;
    list.push_back(10);
    list.push_back(20);
    list.push_back(30);
    assert(list.pop_front() == 10);
    assert(list.size() == 2);
    assert(list.front() == 20);
}

TEST(pop_front_until_empty) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    assert(list.pop_front() == 1);
    assert(list.pop_front() == 2);
    assert(list.pop_front() == 3);
    assert(list.empty());
}

TEST(pop_front_on_empty_throws) {
    LinkedList<int> list;
    bool threw = false;
    try {
        list.pop_front();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(pop_back_returns_value_and_decrements) {
    LinkedList<int> list;
    list.push_back(10);
    list.push_back(20);
    list.push_back(30);
    assert(list.pop_back() == 30);
    assert(list.size() == 2);
    assert(list.back() == 20);
}

TEST(pop_back_until_empty) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    assert(list.pop_back() == 3);
    assert(list.pop_back() == 2);
    assert(list.pop_back() == 1);
    assert(list.empty());
}

TEST(pop_back_on_empty_throws) {
    LinkedList<int> list;
    bool threw = false;
    try {
        list.pop_back();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

// --- Access Operations ---

TEST(front_returns_first_without_removing) {
    LinkedList<int> list;
    list.push_back(10);
    list.push_back(20);
    assert(list.front() == 10);
    assert(list.size() == 2);
}

TEST(back_returns_last_without_removing) {
    LinkedList<int> list;
    list.push_back(10);
    list.push_back(20);
    assert(list.back() == 20);
    assert(list.size() == 2);
}

TEST(at_zero_returns_first) {
    LinkedList<int> list;
    list.push_back(10);
    list.push_back(20);
    list.push_back(30);
    assert(list.at(0) == 10);
}

TEST(at_size_minus_one_returns_last) {
    LinkedList<int> list;
    list.push_back(10);
    list.push_back(20);
    list.push_back(30);
    assert(list.at(2) == 30);
}

TEST(at_middle_returns_correct) {
    LinkedList<int> list;
    list.push_back(10);
    list.push_back(20);
    list.push_back(30);
    list.push_back(40);
    list.push_back(50);
    assert(list.at(2) == 30);
}

TEST(at_invalid_throws) {
    LinkedList<int> list;
    list.push_back(1);
    bool threw = false;
    try {
        list.at(1);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        list.at(100);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

// --- Insert/Remove at Index ---

TEST(insert_at_zero_same_as_push_front) {
    LinkedList<int> list;
    list.push_back(2);
    list.push_back(3);
    list.insert_at(0, 1);
    assert(list.size() == 3);
    assert(list.front() == 1);
    assert(list.at(1) == 2);
}

TEST(insert_at_size_same_as_push_back) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.insert_at(2, 3);
    assert(list.size() == 3);
    assert(list.back() == 3);
}

TEST(insert_at_middle) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(3);
    list.insert_at(1, 2);
    assert(list.size() == 3);
    assert(list.at(0) == 1);
    assert(list.at(1) == 2);
    assert(list.at(2) == 3);
}

TEST(insert_at_invalid_throws) {
    LinkedList<int> list;
    list.push_back(1);
    bool threw = false;
    try {
        list.insert_at(5, 99);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(remove_at_zero_same_as_pop_front) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    assert(list.remove_at(0) == 1);
    assert(list.size() == 2);
    assert(list.front() == 2);
}

TEST(remove_at_size_minus_one_same_as_pop_back) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    assert(list.remove_at(2) == 3);
    assert(list.size() == 2);
    assert(list.back() == 2);
}

TEST(remove_at_middle) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    assert(list.remove_at(1) == 2);
    assert(list.size() == 2);
    assert(list.at(0) == 1);
    assert(list.at(1) == 3);
}

TEST(remove_at_invalid_throws) {
    LinkedList<int> list;
    list.push_back(1);
    bool threw = false;
    try {
        list.remove_at(1);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

// --- Clear ---

TEST(clear_makes_list_empty) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    list.clear();
    assert(list.size() == 0);
    assert(list.empty());
}

TEST(clear_on_empty_is_noop) {
    LinkedList<int> list;
    list.clear();
    assert(list.empty());
}

// --- Iteration ---

TEST(iterate_correct_order) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    list.push_back(4);
    int expected = 1;
    for (const auto& val : list) {
        assert(val == expected);
        ++expected;
    }
    assert(expected == 5);
}

TEST(iterate_empty_list) {
    LinkedList<int> list;
    int iterations = 0;
    for (const auto& val : list) {
        (void)val;
        ++iterations;
    }
    assert(iterations == 0);
}

// --- Copy/Clone ---

TEST(copy_creates_independent_copy) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    LinkedList<int> copy(list);
    assert(copy.size() == 3);
    assert(copy.at(0) == 1);
    assert(copy.at(1) == 2);
    assert(copy.at(2) == 3);
}

TEST(modifying_original_doesnt_affect_copy) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    LinkedList<int> copy(list);
    list.at(0) = 999;
    list.push_back(3);
    assert(copy.at(0) == 1);
    assert(copy.size() == 2);
}

TEST(copy_assignment) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    LinkedList<int> other;
    other.push_back(99);
    other = list;
    assert(other.size() == 2);
    assert(other.at(0) == 1);
    assert(other.at(1) == 2);
    list.at(0) = 999;
    assert(other.at(0) == 1);
}

TEST(move_constructor) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    LinkedList<int> moved(std::move(list));
    assert(moved.size() == 3);
    assert(moved.at(0) == 1);
    assert(list.size() == 0);
    assert(list.empty());
}

TEST(move_assignment) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    LinkedList<int> other;
    other.push_back(99);
    other = std::move(list);
    assert(other.size() == 2);
    assert(other.at(0) == 1);
    assert(list.size() == 0);
}

// --- Non-trivial Types ---

TEST(works_with_strings) {
    LinkedList<std::string> list;
    list.push_back("hello");
    list.push_back("world");
    assert(list.size() == 2);
    assert(list.front() == "hello");
    assert(list.back() == "world");
    assert(list.pop_front() == "hello");
    assert(list.pop_back() == "world");
    assert(list.empty());
}

// --- Edge Cases ---

TEST(single_element_front_equals_back) {
    LinkedList<int> list;
    list.push_back(42);
    assert(list.front() == list.back());
    assert(list.front() == 42);
}

TEST(two_element_proper_linking) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    assert(list.front() == 1);
    assert(list.back() == 2);
    assert(list.at(0) == 1);
    assert(list.at(1) == 2);
    list.pop_front();
    assert(list.front() == 2);
    assert(list.back() == 2);
}

TEST(const_access) {
    LinkedList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    const auto& cref = list;
    assert(cref.front() == 1);
    assert(cref.back() == 3);
    assert(cref.at(1) == 2);
    assert(cref.size() == 3);
    assert(!cref.empty());
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
