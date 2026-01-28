#include "dynamic_array.hpp"

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

TEST(default_construction) {
    DynamicArray<int> arr;
    assert(arr.size() == 0);
    assert(arr.capacity() == 0);
    assert(arr.empty());
}

TEST(sized_construction) {
    DynamicArray<int> arr(5);
    assert(arr.size() == 5);
    assert(arr.capacity() == 5);
    assert(!arr.empty());
    for (std::size_t i = 0; i < arr.size(); ++i)
        assert(arr[i] == 0);
}

TEST(push_back_single) {
    DynamicArray<int> arr;
    arr.push_back(42);
    assert(arr.size() == 1);
    assert(arr[0] == 42);
}

TEST(push_back_multiple) {
    DynamicArray<int> arr;
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);
    assert(arr.size() == 3);
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
}

TEST(push_back_triggers_growth) {
    DynamicArray<int> arr;
    arr.push_back(1);
    assert(arr.capacity() == 1);
    arr.push_back(2);
    assert(arr.capacity() == 2);
    arr.push_back(3);
    assert(arr.capacity() == 4);
    arr.push_back(4);
    arr.push_back(5);
    assert(arr.capacity() == 8);
    assert(arr.size() == 5);
}

TEST(pop_back) {
    DynamicArray<int> arr;
    arr.push_back(10);
    arr.push_back(20);
    arr.push_back(30);
    arr.pop_back();
    assert(arr.size() == 2);
    assert(arr.back() == 20);
    arr.pop_back();
    assert(arr.size() == 1);
    assert(arr.back() == 10);
}

TEST(at_valid_index) {
    DynamicArray<int> arr;
    arr.push_back(100);
    arr.push_back(200);
    arr.push_back(300);
    assert(arr.at(0) == 100);
    assert(arr.at(1) == 200);
    assert(arr.at(2) == 300);
    arr.at(1) = 999;
    assert(arr.at(1) == 999);
}

TEST(at_throws_out_of_range) {
    DynamicArray<int> arr;
    arr.push_back(1);
    bool threw = false;
    try {
        arr.at(1);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        arr.at(100);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);

    DynamicArray<int> empty_arr;
    threw = false;
    try {
        empty_arr.at(0);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(subscript_operator) {
    DynamicArray<int> arr;
    arr.push_back(10);
    arr.push_back(20);
    assert(arr[0] == 10);
    assert(arr[1] == 20);
    arr[0] = 99;
    assert(arr[0] == 99);
}

TEST(front_and_back) {
    DynamicArray<int> arr;
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);
    assert(arr.front() == 1);
    assert(arr.back() == 3);
    arr.front() = 100;
    arr.back() = 300;
    assert(arr[0] == 100);
    assert(arr[2] == 300);
}

TEST(reserve_increases_capacity) {
    DynamicArray<int> arr;
    arr.reserve(10);
    assert(arr.capacity() >= 10);
    assert(arr.size() == 0);
}

TEST(reserve_preserves_elements) {
    DynamicArray<int> arr;
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);
    arr.reserve(100);
    assert(arr.capacity() >= 100);
    assert(arr.size() == 3);
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
}

TEST(reserve_smaller_is_noop) {
    DynamicArray<int> arr;
    arr.reserve(10);
    auto cap = arr.capacity();
    arr.reserve(5);
    assert(arr.capacity() == cap);
}

TEST(clear_resets_size_not_capacity) {
    DynamicArray<int> arr;
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);
    auto cap = arr.capacity();
    arr.clear();
    assert(arr.size() == 0);
    assert(arr.empty());
    assert(arr.capacity() == cap);
}

TEST(copy_constructor) {
    DynamicArray<int> arr;
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);

    DynamicArray<int> copy(arr);
    assert(copy.size() == 3);
    assert(copy[0] == 1);
    assert(copy[1] == 2);
    assert(copy[2] == 3);

    arr[0] = 999;
    assert(copy[0] == 1);
}

TEST(move_constructor) {
    DynamicArray<int> arr;
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);

    DynamicArray<int> moved(std::move(arr));
    assert(moved.size() == 3);
    assert(moved[0] == 1);
    assert(moved[1] == 2);
    assert(moved[2] == 3);
    assert(arr.size() == 0);
    assert(arr.data() == nullptr);
}

TEST(copy_assignment) {
    DynamicArray<int> arr;
    arr.push_back(1);
    arr.push_back(2);

    DynamicArray<int> other;
    other.push_back(99);
    other = arr;

    assert(other.size() == 2);
    assert(other[0] == 1);
    assert(other[1] == 2);

    arr[0] = 999;
    assert(other[0] == 1);
}

TEST(move_assignment) {
    DynamicArray<int> arr;
    arr.push_back(1);
    arr.push_back(2);

    DynamicArray<int> other;
    other.push_back(99);
    other = std::move(arr);

    assert(other.size() == 2);
    assert(other[0] == 1);
    assert(other[1] == 2);
    assert(arr.size() == 0);
    assert(arr.data() == nullptr);
}

TEST(range_based_for) {
    DynamicArray<int> arr;
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);
    arr.push_back(4);

    int sum = 0;
    for (const auto& elem : arr)
        sum += elem;
    assert(sum == 10);
}

TEST(non_trivial_type) {
    DynamicArray<std::string> arr;
    arr.push_back("hello");
    arr.push_back("world");
    assert(arr.size() == 2);
    assert(arr[0] == "hello");
    assert(arr[1] == "world");
    assert(arr.front() == "hello");
    assert(arr.back() == "world");
}

TEST(data_pointer) {
    DynamicArray<int> arr;
    arr.push_back(1);
    arr.push_back(2);
    int* p = arr.data();
    assert(p[0] == 1);
    assert(p[1] == 2);
    p[0] = 100;
    assert(arr[0] == 100);
}

TEST(empty_data_pointer) {
    DynamicArray<int> arr;
    assert(arr.data() == nullptr);
}

TEST(const_access) {
    DynamicArray<int> arr;
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);
    const auto& cref = arr;
    assert(cref[0] == 1);
    assert(cref.at(1) == 2);
    assert(cref.front() == 1);
    assert(cref.back() == 3);
    assert(cref.data() != nullptr);
    assert(cref.size() == 3);
    assert(!cref.empty());
}

TEST(push_back_move) {
    DynamicArray<std::string> arr;
    std::string s = "hello";
    arr.push_back(std::move(s));
    assert(arr[0] == "hello");
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
