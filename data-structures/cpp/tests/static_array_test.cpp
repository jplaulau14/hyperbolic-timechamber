#include "static_array.hpp"

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

TEST(size_and_empty) {
    StaticArray<int, 5> arr;
    assert(arr.size() == 5);
    assert(!arr.empty());

    StaticArray<int, 0> empty_arr;
    assert(empty_arr.size() == 0);
    assert(empty_arr.empty());
}

TEST(fill_and_access) {
    StaticArray<int, 4> arr;
    arr.fill(42);
    for (std::size_t i = 0; i < arr.size(); ++i)
        assert(arr[i] == 42);
}

TEST(subscript_operator) {
    StaticArray<int, 3> arr;
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    assert(arr[0] == 10);
    assert(arr[1] == 20);
    assert(arr[2] == 30);
}

TEST(at_valid_index) {
    StaticArray<int, 3> arr;
    arr.at(0) = 100;
    arr.at(1) = 200;
    arr.at(2) = 300;
    assert(arr.at(0) == 100);
    assert(arr.at(1) == 200);
    assert(arr.at(2) == 300);
}

TEST(at_throws_out_of_range) {
    StaticArray<int, 3> arr;
    bool threw = false;
    try {
        arr.at(3);
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
}

TEST(at_throws_on_zero_size) {
    StaticArray<int, 0> arr;
    bool threw = false;
    try {
        arr.at(0);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(front_and_back) {
    StaticArray<int, 4> arr;
    arr.fill(0);
    arr[0] = 1;
    arr[3] = 99;
    assert(arr.front() == 1);
    assert(arr.back() == 99);
}

TEST(data_pointer) {
    StaticArray<int, 3> arr;
    arr.fill(7);
    int* p = arr.data();
    assert(p[0] == 7);
    assert(p[1] == 7);
    assert(p[2] == 7);

    p[1] = 42;
    assert(arr[1] == 42);
}

TEST(data_pointer_zero_size) {
    StaticArray<int, 0> arr;
    assert(arr.data() == nullptr);
}

TEST(range_based_for) {
    StaticArray<int, 5> arr;
    arr.fill(3);
    int sum = 0;
    for (const auto& elem : arr)
        sum += elem;
    assert(sum == 15);
}

TEST(iterator_arithmetic) {
    StaticArray<int, 4> arr;
    for (std::size_t i = 0; i < 4; ++i)
        arr[i] = static_cast<int>(i * 10);

    auto it = arr.begin();
    assert(*it == 0);
    ++it;
    assert(*it == 10);
    assert(arr.end() - arr.begin() == 4);
}

TEST(const_access) {
    StaticArray<int, 3> arr;
    arr.fill(5);
    const auto& cref = arr;
    assert(cref[0] == 5);
    assert(cref.at(1) == 5);
    assert(cref.front() == 5);
    assert(cref.back() == 5);
    assert(cref.data() != nullptr);
    assert(cref.size() == 3);
    assert(!cref.empty());
}

TEST(non_trivial_type) {
    StaticArray<std::string, 3> arr;
    arr[0] = "hello";
    arr[1] = "world";
    arr[2] = "!";
    assert(arr[0] == "hello");
    assert(arr.at(1) == "world");
    assert(arr.back() == "!");
}

TEST(fill_overwrites) {
    StaticArray<int, 3> arr;
    arr.fill(1);
    arr.fill(2);
    for (std::size_t i = 0; i < arr.size(); ++i)
        assert(arr[i] == 2);
}

TEST(zero_size_iteration) {
    StaticArray<int, 0> arr;
    int count = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it)
        ++count;
    assert(count == 0);
}

TEST(single_element) {
    StaticArray<int, 1> arr;
    arr[0] = 42;
    assert(arr.front() == 42);
    assert(arr.back() == 42);
    assert(arr.size() == 1);
    assert(arr.begin() + 1 == arr.end());
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
