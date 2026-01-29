#include "stack.hpp"

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

// --- Tests ---

TEST(new_stack_is_empty) {
    Stack<int> s;
    assert(s.size() == 0);
    assert(s.empty());
}

TEST(top_on_empty_throws) {
    Stack<int> s;
    bool threw = false;
    try {
        s.top();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(pop_on_empty_throws) {
    Stack<int> s;
    bool threw = false;
    try {
        s.pop();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(push_single_element) {
    Stack<int> s;
    s.push(42);
    assert(s.size() == 1);
    assert(!s.empty());
}

TEST(push_multiple_elements) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    assert(s.size() == 3);
}

TEST(push_many_elements_triggers_growth) {
    Stack<int> s;
    for (int i = 0; i < 100; ++i)
        s.push(i);
    assert(s.size() == 100);
}

TEST(pop_returns_top_element) {
    Stack<int> s;
    s.push(10);
    s.push(20);
    s.push(30);
    assert(s.pop() == 30);
}

TEST(pop_decrements_size) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    s.pop();
    assert(s.size() == 2);
}

TEST(pop_all_elements_until_empty) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    s.pop();
    s.pop();
    s.pop();
    assert(s.empty());
}

TEST(lifo_order) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    assert(s.pop() == 3);
    assert(s.pop() == 2);
    assert(s.pop() == 1);
}

TEST(top_returns_without_removing) {
    Stack<int> s;
    s.push(42);
    assert(s.top() == 42);
    assert(s.size() == 1);
}

TEST(top_after_push_shows_new_element) {
    Stack<int> s;
    s.push(10);
    assert(s.top() == 10);
    s.push(20);
    assert(s.top() == 20);
}

TEST(top_multiple_times_same_value) {
    Stack<int> s;
    s.push(99);
    assert(s.top() == 99);
    assert(s.top() == 99);
    assert(s.top() == 99);
    assert(s.size() == 1);
}

TEST(clear_makes_stack_empty) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    s.clear();
    assert(s.empty());
    assert(s.size() == 0);
}

TEST(clear_on_empty_is_noop) {
    Stack<int> s;
    s.clear();
    assert(s.empty());
}

TEST(push_after_clear) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    s.clear();
    s.push(100);
    assert(s.size() == 1);
    assert(s.top() == 100);
}

TEST(size_correct_after_pushes) {
    Stack<int> s;
    assert(s.size() == 0);
    s.push(1);
    assert(s.size() == 1);
    s.push(2);
    assert(s.size() == 2);
    s.push(3);
    assert(s.size() == 3);
}

TEST(size_correct_after_pops) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    s.pop();
    assert(s.size() == 2);
    s.pop();
    assert(s.size() == 1);
}

TEST(empty_only_when_size_zero) {
    Stack<int> s;
    assert(s.empty());
    s.push(1);
    assert(!s.empty());
    s.pop();
    assert(s.empty());
}

TEST(copy_creates_independent_copy) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    Stack<int> copy(s);
    assert(copy.size() == 3);
    assert(copy.top() == 3);
}

TEST(push_to_original_doesnt_affect_copy) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    Stack<int> copy(s);
    s.push(100);
    assert(s.size() == 3);
    assert(copy.size() == 2);
    assert(s.top() == 100);
    assert(copy.top() == 2);
}

TEST(pop_from_original_doesnt_affect_copy) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    Stack<int> copy(s);
    s.pop();
    assert(s.size() == 2);
    assert(copy.size() == 3);
    assert(copy.top() == 3);
}

TEST(works_with_strings) {
    Stack<std::string> s;
    s.push("hello");
    s.push("world");
    assert(s.top() == "world");
    assert(s.pop() == "world");
    assert(s.pop() == "hello");
    assert(s.empty());
}

TEST(works_with_struct) {
    struct Point { int x; int y; };
    Stack<Point> s;
    s.push({1, 2});
    s.push({3, 4});
    Point p = s.pop();
    assert(p.x == 3 && p.y == 4);
}

TEST(single_element_push_then_pop) {
    Stack<int> s;
    s.push(42);
    assert(s.pop() == 42);
    assert(s.empty());
}

TEST(alternating_push_pop) {
    Stack<int> s;
    s.push(1);
    assert(s.pop() == 1);
    s.push(2);
    s.push(3);
    assert(s.pop() == 3);
    s.push(4);
    assert(s.pop() == 4);
    assert(s.pop() == 2);
    assert(s.empty());
}

TEST(large_number_of_elements) {
    Stack<int> s;
    for (int i = 0; i < 10000; ++i)
        s.push(i);
    assert(s.size() == 10000);
    for (int i = 9999; i >= 0; --i)
        assert(s.pop() == i);
    assert(s.empty());
}

TEST(move_constructor) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    Stack<int> moved(std::move(s));
    assert(moved.size() == 3);
    assert(moved.top() == 3);
    assert(s.empty());
}

TEST(move_assignment) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    Stack<int> other;
    other.push(99);
    other = std::move(s);
    assert(other.size() == 2);
    assert(other.top() == 2);
    assert(s.empty());
}

TEST(copy_assignment) {
    Stack<int> s;
    s.push(1);
    s.push(2);
    Stack<int> other;
    other.push(99);
    other = s;
    assert(other.size() == 2);
    assert(other.top() == 2);
    s.push(100);
    assert(other.top() == 2);
}

TEST(const_top) {
    Stack<int> s;
    s.push(42);
    const Stack<int>& cref = s;
    assert(cref.top() == 42);
}

TEST(push_rvalue) {
    Stack<std::string> s;
    std::string str = "test";
    s.push(std::move(str));
    assert(s.top() == "test");
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
