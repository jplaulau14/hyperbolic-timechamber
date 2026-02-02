# Rust Data Structures

Requires Rust (cargo).

## Build & Test

```bash
cargo test
```

## Static Array

`StaticArray<T, N>` in `src/static_array.rs`

Fixed-size array using const generics.

```rust
use data_structures::StaticArray;

let mut arr: StaticArray<i32, 5> = StaticArray::new();
arr.fill(0);
arr[0] = 10;
arr.at(1);            // checked access (returns Option<&T>)

for v in &arr { /* iteration works */ }
```

Also supports: `at_mut()`, `front()`, `back()`, `data()`, `size()`, `is_empty()`, `iter()`, `iter_mut()`.

Zero-size (`StaticArray<T, 0>`) works — `data()` returns empty slice, `at()`/`front()`/`back()` return `None`.

## Dynamic Array

`DynamicArray<T>` in `src/dynamic_array.rs`

Heap-allocated, resizable array using manual memory management.

```rust
use data_structures::DynamicArray;

let mut arr: DynamicArray<i32> = DynamicArray::new();
arr.push_back(10);
arr.push_back(20);
arr.at(0);            // checked access (returns Option<&T>)
arr.reserve(100);     // pre-allocate capacity
arr.pop_back();       // returns Option<T>
arr.clear();

for v in &arr { /* iteration works */ }
```

Also supports: `at_mut()`, `front()`, `back()`, `data()`, `size()`, `capacity()`, `is_empty()`, `iter()`, `clone()`.

Growth strategy doubles capacity when exceeded. Implements `Drop` for proper cleanup.

## Linked List

`LinkedList<T>` in `src/linked_list.rs`

Singly linked list with head/tail pointers using `Option<Box<Node<T>>>`.

```rust
use data_structures::LinkedList;

let mut list: LinkedList<i32> = LinkedList::new();
list.push_back(10);
list.push_front(5);
list.front();         // checked access (returns Option<&T>)
list.back();
list.at(1);           // checked access (returns Option<&T>)
list.pop_front();     // returns Option<T>
list.pop_back();

for v in &list { /* iteration works */ }
```

Also supports: `insert_at()`, `remove_at()`, `size()`, `is_empty()`, `clear()`, `iter()`, `clone()`.

Implements `Drop` for proper cleanup, `Clone` for deep copy, `Default`, and `IntoIterator`.

## Stack

`Stack<T>` in `src/stack.rs`

LIFO container built on Vec.

```rust
use data_structures::Stack;

let mut s: Stack<i32> = Stack::new();
s.push(10);
s.push(20);
s.top();      // Some(&20)
s.pop();      // Some(20)
s.size();     // 1
```

Methods: `push()`, `pop()`, `top()`, `size()`, `is_empty()`, `clear()`.

Implements `Clone` and `Default`.

## Queue

`Queue<T>` in `src/queue.rs`

FIFO container using circular buffer with raw allocation.

```rust
use data_structures::Queue;

let mut q: Queue<i32> = Queue::new();
q.enqueue(10);
q.enqueue(20);
q.front();    // Some(&10)
q.dequeue();  // Some(10)
q.size();     // 1
```

Methods: `enqueue()`, `dequeue()`, `front()`, `back()`, `size()`, `is_empty()`, `clear()`.

Implements `Drop`, `Clone`, and `Default`.

## Hash Map

`HashMap<K, V>` in `src/hash_map.rs`

Key-value store with O(1) average operations using separate chaining.

```rust
use data_structures::HashMap;

let mut map: HashMap<&str, i32> = HashMap::new();
map.insert("one", 1);
map.insert("two", 2);
map.get(&"one");       // Some(&1)
map.contains(&"two");  // true
map.remove(&"one");    // Some(1)
```

Methods: `insert()`, `get()`, `get_mut()`, `remove()`, `contains()`, `size()`, `is_empty()`, `clear()`, `keys()`, `values()`, `capacity()`.

Implements `Clone` (where K, V: Clone) and `Default`. Rehashes when load factor exceeds 0.75.

## Binary Heap

`BinaryHeap<T>` in `src/binary_heap.rs`

Min-heap with O(log n) push/pop.

```rust
use data_structures::BinaryHeap;

let mut heap: BinaryHeap<i32> = BinaryHeap::new();
heap.push(30);
heap.push(10);
heap.push(20);
heap.peek();  // Some(&10)
heap.pop();   // Some(10)
heap.pop();   // Some(20)
```

Methods: `push()`, `pop()`, `peek()`, `size()`, `is_empty()`, `clear()`, `from_vec()`.

Implements `Clone` (where T: Clone), `Default`, and `FromIterator`.

## Binary Search Tree

`BinarySearchTree<T>` in `src/binary_search_tree.rs`

Ordered binary tree with O(log n) average operations.

```rust
use data_structures::BinarySearchTree;

let mut bst: BinarySearchTree<i32> = BinarySearchTree::new();
bst.insert(50);
bst.insert(30);
bst.insert(70);
bst.contains(&30);  // true
bst.min();          // Some(&30)
bst.max();          // Some(&70)
bst.remove(&30);
bst.in_order();     // [&50, &70]
```

Methods: `insert()`, `remove()`, `contains()`, `min()`, `max()`, `size()`, `is_empty()`, `clear()`, `in_order()`, `pre_order()`, `post_order()`.

Implements `Clone` (where T: Clone), `Default`, and `IntoIterator`.
