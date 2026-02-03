# C++ Data Structures

Requires CMake and a C++17 compiler.

## Build & Test

```bash
cmake -B build
cmake --build build
./build/static_array_test
./build/dynamic_array_test
./build/linked_list_test
./build/stack_test
./build/queue_test
./build/hash_map_test
./build/binary_heap_test
./build/binary_search_tree_test
./build/avl_tree_test
```

## Static Array

`StaticArray<T, N>` in `src/static_array.hpp`

Stack-allocated, fixed-size array. Header-only.

```cpp
#include "static_array.hpp"

StaticArray<int, 5> arr;
arr.fill(0);
arr[0] = 10;
arr.at(1) = 20;       // throws std::out_of_range if out of bounds

for (auto& v : arr) { /* iteration works */ }
```

Also supports: `front()`, `back()`, `data()`, `size()`, `empty()`, `begin()`/`end()`.

Zero-size (`StaticArray<T, 0>`) is handled — `data()` returns `nullptr`, `at()` always throws.

## Dynamic Array

`DynamicArray<T>` in `src/dynamic_array.hpp`

Heap-allocated, resizable array. Header-only.

```cpp
#include "dynamic_array.hpp"

DynamicArray<int> arr;
arr.push_back(10);
arr.push_back(20);
arr.at(0);            // throws std::out_of_range if out of bounds
arr.reserve(100);     // pre-allocate capacity
arr.pop_back();
arr.clear();

for (auto& v : arr) { /* iteration works */ }
```

Also supports: `front()`, `back()`, `data()`, `size()`, `capacity()`, `empty()`, `begin()`/`end()`.

Copy/move constructors and assignment operators are implemented. Growth strategy doubles capacity when exceeded.

## Linked List

`LinkedList<T>` in `src/linked_list.hpp`

Singly linked list with head/tail pointers. Header-only.

```cpp
#include "linked_list.hpp"

LinkedList<int> list;
list.push_back(10);
list.push_front(5);
list.front();         // 5
list.back();          // 10
list.at(1);           // throws std::out_of_range if out of bounds
list.pop_front();
list.pop_back();

for (auto& v : list) { /* iteration works */ }
```

Also supports: `insert_at()`, `remove_at()`, `size()`, `empty()`, `clear()`, `begin()`/`end()`.

Copy/move constructors and assignment operators are implemented.

## Stack

`Stack<T>` in `src/stack.hpp`

LIFO container built on dynamic array. Header-only.

```cpp
#include "stack.hpp"

Stack<int> s;
s.push(10);
s.push(20);
s.top();      // 20
s.pop();      // returns 20
s.size();     // 1
```

Methods: `push()`, `pop()`, `top()`, `size()`, `empty()`, `clear()`.

## Queue

`Queue<T>` in `src/queue.hpp`

FIFO container using circular buffer. Header-only.

```cpp
#include "queue.hpp"

Queue<int> q;
q.enqueue(10);
q.enqueue(20);
q.front();    // 10
q.dequeue();  // returns 10
q.size();     // 1
```

Methods: `enqueue()`, `dequeue()`, `front()`, `back()`, `size()`, `empty()`, `clear()`.

## Hash Map

`HashMap<K, V>` in `src/hash_map.hpp`

Key-value store with O(1) average operations using separate chaining.

```cpp
#include "hash_map.hpp"

HashMap<std::string, int> map;
map.insert("one", 1);
map.insert("two", 2);
map.get("one");      // 1
map.contains("two"); // true
map.remove("one");
```

Methods: `insert()`, `get()`, `find()`, `remove()`, `contains()`, `size()`, `empty()`, `clear()`, `keys()`, `values()`.

## Binary Heap

`BinaryHeap<T>` in `src/binary_heap.hpp`

Min-heap with O(log n) push/pop.

```cpp
#include "binary_heap.hpp"

BinaryHeap<int> heap;
heap.push(30);
heap.push(10);
heap.push(20);
heap.peek();  // 10
heap.pop();   // returns 10
heap.pop();   // returns 20

int arr[] = {5, 3, 8, 1};
auto heap2 = BinaryHeap<int>::from_array(arr, 4);
```

Methods: `push()`, `pop()`, `peek()`, `size()`, `empty()`, `clear()`, `from_array()`.

## Binary Search Tree

`BinarySearchTree<T>` in `src/binary_search_tree.hpp`

Ordered binary tree with O(log n) average operations.

```cpp
#include "binary_search_tree.hpp"

BinarySearchTree<int> bst;
bst.insert(50);
bst.insert(30);
bst.insert(70);
bst.contains(30);  // true
bst.min();         // 30
bst.max();         // 70
bst.remove(30);
std::vector<int> sorted;
bst.in_order(sorted);  // [50, 70]
```

Methods: `insert()`, `remove()`, `contains()`, `min()`, `max()`, `size()`, `empty()`, `clear()`, `in_order()`, `pre_order()`, `post_order()`.

## AVL Tree

`AVLTree<T>` in `src/avl_tree.hpp`

Self-balancing binary search tree with O(log n) guaranteed operations.

```cpp
#include "avl_tree.hpp"

AVLTree<int> avl;
avl.insert(50);
avl.insert(30);
avl.insert(70);
avl.insert(20);       // Tree stays balanced
avl.height();         // O(log n) guaranteed
avl.contains(30);     // true
avl.min();            // 20
avl.max();            // 70
avl.is_balanced();    // true
avl.remove(30);
std::vector<int> sorted;
avl.in_order(sorted); // [20, 50, 70]
```

Methods: `insert()`, `remove()`, `contains()`, `min()`, `max()`, `size()`, `empty()`, `clear()`, `height()`, `is_balanced()`, `in_order()`, `pre_order()`, `post_order()`.
