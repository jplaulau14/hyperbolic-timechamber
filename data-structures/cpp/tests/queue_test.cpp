#include "queue.hpp"

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

TEST(new_queue_is_empty) {
    Queue<int> q;
    assert(q.size() == 0);
    assert(q.empty());
}

TEST(front_on_empty_throws) {
    Queue<int> q;
    bool threw = false;
    try {
        q.front();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(back_on_empty_throws) {
    Queue<int> q;
    bool threw = false;
    try {
        q.back();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(dequeue_on_empty_throws) {
    Queue<int> q;
    bool threw = false;
    try {
        q.dequeue();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(enqueue_single_element) {
    Queue<int> q;
    q.enqueue(42);
    assert(q.size() == 1);
    assert(!q.empty());
}

TEST(enqueue_multiple_elements) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    assert(q.size() == 3);
}

TEST(front_and_back_after_enqueues) {
    Queue<int> q;
    q.enqueue(10);
    q.enqueue(20);
    q.enqueue(30);
    assert(q.front() == 10);
    assert(q.back() == 30);
}

TEST(dequeue_returns_front) {
    Queue<int> q;
    q.enqueue(100);
    q.enqueue(200);
    assert(q.dequeue() == 100);
}

TEST(dequeue_decrements_size) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    q.dequeue();
    assert(q.size() == 1);
}

TEST(dequeue_all_until_empty) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    q.dequeue();
    q.dequeue();
    q.dequeue();
    assert(q.empty());
}

TEST(fifo_order) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    assert(q.dequeue() == 1);
    assert(q.dequeue() == 2);
    assert(q.dequeue() == 3);
}

TEST(front_does_not_remove) {
    Queue<int> q;
    q.enqueue(99);
    assert(q.front() == 99);
    assert(q.size() == 1);
    assert(q.front() == 99);
}

TEST(back_does_not_remove) {
    Queue<int> q;
    q.enqueue(99);
    assert(q.back() == 99);
    assert(q.size() == 1);
    assert(q.back() == 99);
}

TEST(front_and_back_same_single_element) {
    Queue<int> q;
    q.enqueue(42);
    assert(q.front() == 42);
    assert(q.back() == 42);
}

TEST(circular_buffer_wrap_around) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    q.dequeue();
    q.enqueue(3);
    q.dequeue();
    q.enqueue(4);
    assert(q.front() == 3);
    assert(q.back() == 4);
    assert(q.size() == 2);
}

TEST(fill_dequeue_some_enqueue_more) {
    Queue<int> q;
    for (int i = 0; i < 4; ++i) q.enqueue(i);
    q.dequeue();
    q.dequeue();
    q.enqueue(10);
    q.enqueue(11);
    assert(q.size() == 4);
    assert(q.dequeue() == 2);
    assert(q.dequeue() == 3);
    assert(q.dequeue() == 10);
    assert(q.dequeue() == 11);
}

TEST(growth_preserves_order) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    q.dequeue();
    q.enqueue(3);
    q.enqueue(4);
    q.enqueue(5);
    assert(q.dequeue() == 2);
    assert(q.dequeue() == 3);
    assert(q.dequeue() == 4);
    assert(q.dequeue() == 5);
}

TEST(clear_makes_empty) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    q.clear();
    assert(q.empty());
    assert(q.size() == 0);
}

TEST(clear_on_empty_is_noop) {
    Queue<int> q;
    q.clear();
    assert(q.empty());
}

TEST(enqueue_after_clear) {
    Queue<int> q;
    q.enqueue(1);
    q.clear();
    q.enqueue(99);
    assert(q.size() == 1);
    assert(q.front() == 99);
}

TEST(size_after_enqueues) {
    Queue<int> q;
    for (int i = 0; i < 5; ++i) q.enqueue(i);
    assert(q.size() == 5);
}

TEST(size_after_dequeues) {
    Queue<int> q;
    for (int i = 0; i < 5; ++i) q.enqueue(i);
    q.dequeue();
    q.dequeue();
    assert(q.size() == 3);
}

TEST(empty_true_only_when_size_zero) {
    Queue<int> q;
    assert(q.empty());
    q.enqueue(1);
    assert(!q.empty());
    q.dequeue();
    assert(q.empty());
}

TEST(copy_constructor) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    Queue<int> copy(q);
    assert(copy.size() == 3);
    assert(copy.dequeue() == 1);
    assert(copy.dequeue() == 2);
    assert(copy.dequeue() == 3);
}

TEST(copy_is_independent) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    Queue<int> copy(q);
    q.enqueue(99);
    q.dequeue();
    assert(copy.size() == 2);
    assert(copy.front() == 1);
}

TEST(copy_assignment) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    Queue<int> other;
    other.enqueue(99);
    other = q;
    assert(other.size() == 2);
    assert(other.dequeue() == 1);
    q.enqueue(100);
    assert(other.size() == 1);
}

TEST(move_constructor) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    Queue<int> moved(std::move(q));
    assert(moved.size() == 2);
    assert(moved.dequeue() == 1);
    assert(q.empty());
}

TEST(move_assignment) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    Queue<int> other;
    other.enqueue(99);
    other = std::move(q);
    assert(other.size() == 2);
    assert(other.dequeue() == 1);
    assert(q.empty());
}

TEST(works_with_strings) {
    Queue<std::string> q;
    q.enqueue("hello");
    q.enqueue("world");
    assert(q.front() == "hello");
    assert(q.back() == "world");
    assert(q.dequeue() == "hello");
    assert(q.dequeue() == "world");
}

TEST(single_element_enqueue_dequeue) {
    Queue<int> q;
    q.enqueue(42);
    assert(q.dequeue() == 42);
    assert(q.empty());
}

TEST(alternating_enqueue_dequeue) {
    Queue<int> q;
    for (int i = 0; i < 10; ++i) {
        q.enqueue(i);
        assert(q.dequeue() == i);
    }
    assert(q.empty());
}

TEST(large_number_of_elements) {
    Queue<int> q;
    for (int i = 0; i < 1000; ++i) q.enqueue(i);
    assert(q.size() == 1000);
    for (int i = 0; i < 1000; ++i) {
        assert(q.dequeue() == i);
    }
    assert(q.empty());
}

TEST(many_wrap_around_cycles) {
    Queue<int> q;
    for (int cycle = 0; cycle < 100; ++cycle) {
        for (int i = 0; i < 10; ++i) q.enqueue(cycle * 10 + i);
        for (int i = 0; i < 10; ++i) {
            assert(q.dequeue() == cycle * 10 + i);
        }
    }
    assert(q.empty());
}

TEST(const_front_and_back) {
    Queue<int> q;
    q.enqueue(1);
    q.enqueue(2);
    const Queue<int>& cref = q;
    assert(cref.front() == 1);
    assert(cref.back() == 2);
    assert(cref.size() == 2);
    assert(!cref.empty());
}

TEST(enqueue_move) {
    Queue<std::string> q;
    std::string s = "moveme";
    q.enqueue(std::move(s));
    assert(q.front() == "moveme");
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
