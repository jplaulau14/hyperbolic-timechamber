#include "hash_map.hpp"

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

static TestEntry tests[64];
static int count = 0;

// --- Construction & Basic State ---

TEST(new_map_is_empty) {
    HashMap<int, int> map;
    assert(map.size() == 0);
    assert(map.empty());
}

TEST(default_capacity_is_16) {
    HashMap<int, int> map;
    assert(map.capacity() == 16);
}

// --- Insert Operations ---

TEST(insert_single_pair) {
    HashMap<std::string, int> map;
    map.insert("one", 1);
    assert(map.size() == 1);
    assert(map.get("one") == 1);
}

TEST(insert_multiple_pairs) {
    HashMap<std::string, int> map;
    map.insert("one", 1);
    map.insert("two", 2);
    map.insert("three", 3);
    assert(map.size() == 3);
}

TEST(insert_duplicate_key_updates_value) {
    HashMap<std::string, int> map;
    map.insert("key", 100);
    map.insert("key", 200);
    assert(map.size() == 1);
    assert(map.get("key") == 200);
}

TEST(insert_triggers_rehash) {
    HashMap<int, int> map;
    for (int i = 0; i < 20; ++i)
        map.insert(i, i * 10);
    assert(map.capacity() > 16);
    for (int i = 0; i < 20; ++i)
        assert(map.get(i) == i * 10);
}

// --- Get Operations ---

TEST(get_existing_key) {
    HashMap<std::string, int> map;
    map.insert("hello", 42);
    assert(map.get("hello") == 42);
}

TEST(get_nonexistent_key_throws) {
    HashMap<std::string, int> map;
    bool threw = false;
    try {
        map.get("missing");
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

TEST(get_after_update) {
    HashMap<std::string, int> map;
    map.insert("key", 1);
    map.insert("key", 2);
    assert(map.get("key") == 2);
}

// --- Find Operations ---

TEST(find_existing_key) {
    HashMap<std::string, int> map;
    map.insert("key", 42);
    int* ptr = map.find("key");
    assert(ptr != nullptr);
    assert(*ptr == 42);
}

TEST(find_nonexistent_key_returns_nullptr) {
    HashMap<std::string, int> map;
    assert(map.find("missing") == nullptr);
}

// --- Remove Operations ---

TEST(remove_existing_key) {
    HashMap<std::string, int> map;
    map.insert("key", 42);
    assert(map.remove("key"));
    assert(map.size() == 0);
}

TEST(remove_nonexistent_key_returns_false) {
    HashMap<std::string, int> map;
    assert(!map.remove("missing"));
}

TEST(remove_decrements_size) {
    HashMap<std::string, int> map;
    map.insert("a", 1);
    map.insert("b", 2);
    map.insert("c", 3);
    map.remove("b");
    assert(map.size() == 2);
}

TEST(get_after_remove_throws) {
    HashMap<std::string, int> map;
    map.insert("key", 42);
    map.remove("key");
    bool threw = false;
    try {
        map.get("key");
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

// --- Contains ---

TEST(contains_returns_true_for_existing) {
    HashMap<std::string, int> map;
    map.insert("key", 42);
    assert(map.contains("key"));
}

TEST(contains_returns_false_for_nonexistent) {
    HashMap<std::string, int> map;
    assert(!map.contains("missing"));
}

TEST(contains_after_remove_returns_false) {
    HashMap<std::string, int> map;
    map.insert("key", 42);
    map.remove("key");
    assert(!map.contains("key"));
}

// --- Clear ---

TEST(clear_makes_map_empty) {
    HashMap<std::string, int> map;
    map.insert("a", 1);
    map.insert("b", 2);
    map.clear();
    assert(map.empty());
    assert(map.size() == 0);
}

TEST(clear_on_empty_is_noop) {
    HashMap<std::string, int> map;
    map.clear();
    assert(map.empty());
}

// --- Keys/Values ---

TEST(keys_returns_all_keys) {
    HashMap<std::string, int> map;
    map.insert("a", 1);
    map.insert("b", 2);
    map.insert("c", 3);
    std::vector<std::string> keys;
    map.keys(keys);
    assert(keys.size() == 3);
}

TEST(values_returns_all_values) {
    HashMap<std::string, int> map;
    map.insert("a", 1);
    map.insert("b", 2);
    map.insert("c", 3);
    std::vector<int> values;
    map.values(values);
    assert(values.size() == 3);
    int sum = 0;
    for (int v : values) sum += v;
    assert(sum == 6);
}

TEST(keys_and_values_consistent) {
    HashMap<int, int> map;
    map.insert(1, 10);
    map.insert(2, 20);
    map.insert(3, 30);
    std::vector<int> keys, values;
    map.keys(keys);
    map.values(values);
    assert(keys.size() == values.size());
    for (std::size_t i = 0; i < keys.size(); ++i)
        assert(map.get(keys[i]) == values[i]);
}

// --- Collision Handling ---

struct CollidingHash {
    int val;
    bool operator==(const CollidingHash& other) const { return val == other.val; }
};

namespace std {
    template<> struct hash<CollidingHash> {
        std::size_t operator()(const CollidingHash&) const { return 42; }
    };
}

TEST(insert_keys_that_collide) {
    HashMap<CollidingHash, int> map;
    map.insert({1}, 100);
    map.insert({2}, 200);
    map.insert({3}, 300);
    assert(map.size() == 3);
}

TEST(get_works_with_collisions) {
    HashMap<CollidingHash, int> map;
    map.insert({1}, 100);
    map.insert({2}, 200);
    map.insert({3}, 300);
    assert(map.get({1}) == 100);
    assert(map.get({2}) == 200);
    assert(map.get({3}) == 300);
}

TEST(remove_works_with_collisions) {
    HashMap<CollidingHash, int> map;
    map.insert({1}, 100);
    map.insert({2}, 200);
    map.insert({3}, 300);
    map.remove({2});
    assert(map.size() == 2);
    assert(map.get({1}) == 100);
    assert(map.get({3}) == 300);
    assert(!map.contains({2}));
}

// --- Rehashing ---

TEST(map_grows_when_load_factor_exceeded) {
    HashMap<int, int> map;
    std::size_t initial = map.capacity();
    for (int i = 0; i < 100; ++i)
        map.insert(i, i);
    assert(map.capacity() > initial);
}

TEST(all_entries_accessible_after_rehash) {
    HashMap<int, int> map;
    for (int i = 0; i < 100; ++i)
        map.insert(i, i * 2);
    for (int i = 0; i < 100; ++i)
        assert(map.get(i) == i * 2);
}

TEST(size_unchanged_after_rehash) {
    HashMap<int, int> map;
    for (int i = 0; i < 50; ++i)
        map.insert(i, i);
    assert(map.size() == 50);
}

// --- Copy/Clone ---

TEST(copy_creates_independent_copy) {
    HashMap<std::string, int> map;
    map.insert("a", 1);
    map.insert("b", 2);
    HashMap<std::string, int> copy(map);
    assert(copy.size() == 2);
    assert(copy.get("a") == 1);
    assert(copy.get("b") == 2);
}

TEST(modify_original_doesnt_affect_copy) {
    HashMap<std::string, int> map;
    map.insert("a", 1);
    HashMap<std::string, int> copy(map);
    map.insert("b", 2);
    map.insert("a", 100);
    assert(copy.size() == 1);
    assert(copy.get("a") == 1);
    assert(!copy.contains("b"));
}

// --- Non-trivial Types ---

TEST(works_with_string_keys) {
    HashMap<std::string, int> map;
    map.insert("hello", 1);
    map.insert("world", 2);
    assert(map.get("hello") == 1);
    assert(map.get("world") == 2);
}

TEST(works_with_string_values) {
    HashMap<int, std::string> map;
    map.insert(1, "one");
    map.insert(2, "two");
    assert(map.get(1) == "one");
    assert(map.get(2) == "two");
}

TEST(works_with_struct_values) {
    struct Point { int x; int y; };
    HashMap<std::string, Point> map;
    map.insert("origin", {0, 0});
    map.insert("unit", {1, 1});
    Point p = map.get("unit");
    assert(p.x == 1 && p.y == 1);
}

// --- Edge Cases ---

TEST(single_element_map) {
    HashMap<int, int> map;
    map.insert(42, 100);
    assert(map.size() == 1);
    assert(map.get(42) == 100);
    map.remove(42);
    assert(map.empty());
}

TEST(large_number_of_elements) {
    HashMap<int, int> map;
    for (int i = 0; i < 1000; ++i)
        map.insert(i, i * 2);
    assert(map.size() == 1000);
    for (int i = 0; i < 1000; ++i)
        assert(map.get(i) == i * 2);
}

TEST(many_insertions_and_removals) {
    HashMap<int, int> map;
    for (int i = 0; i < 500; ++i)
        map.insert(i, i);
    for (int i = 0; i < 250; ++i)
        map.remove(i);
    assert(map.size() == 250);
    for (int i = 250; i < 500; ++i)
        assert(map.contains(i));
}

// --- Move semantics ---

TEST(move_constructor) {
    HashMap<std::string, int> map;
    map.insert("a", 1);
    map.insert("b", 2);
    HashMap<std::string, int> moved(std::move(map));
    assert(moved.size() == 2);
    assert(moved.get("a") == 1);
    assert(map.empty());
}

TEST(move_assignment) {
    HashMap<std::string, int> map;
    map.insert("a", 1);
    HashMap<std::string, int> other;
    other.insert("x", 99);
    other = std::move(map);
    assert(other.size() == 1);
    assert(other.get("a") == 1);
    assert(map.empty());
}

TEST(copy_assignment) {
    HashMap<std::string, int> map;
    map.insert("a", 1);
    HashMap<std::string, int> other;
    other.insert("x", 99);
    other = map;
    assert(other.size() == 1);
    assert(other.get("a") == 1);
    map.insert("b", 2);
    assert(other.size() == 1);
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
