# Python Data Structures

Requires Python 3. No external dependencies.

## Run Tests

```bash
python3 -m unittest discover tests -v
```

## Static Array

`StaticArray` in `src/static_array.py`

Fixed-size array.

```python
from static_array import StaticArray

arr = StaticArray(5)
arr.fill(0)
arr[0] = 10
arr.at(1)             # raises IndexError if out of bounds
arr.set_at(1, 20)     # raises IndexError if out of bounds

for v in arr:         # iteration works
    pass
```

Also supports: `front()`, `back()`, `data()`, `size()`, `empty()`, `len()`.

Zero-size (`StaticArray(0)`) is handled — `data()` returns `None`, `at()` always raises.

## Dynamic Array

`DynamicArray` in `src/dynamic_array.py`

Resizable array.

```python
from dynamic_array import DynamicArray

arr = DynamicArray()
arr.push_back(10)
arr.push_back(20)
arr.at(0)             # raises IndexError if out of bounds
arr.reserve(100)      # pre-allocate capacity
arr.pop_back()
arr.clear()

for v in arr:         # iteration works
    pass
```

Also supports: `front()`, `back()`, `data()`, `size()`, `capacity()`, `empty()`, `copy()`, `len()`.

Growth strategy doubles capacity when exceeded.

## Linked List

`LinkedList` in `src/linked_list.py`

Singly linked list with head/tail pointers.

```python
from linked_list import LinkedList

lst = LinkedList()
lst.push_back(10)
lst.push_front(5)
lst.front()           # 5
lst.back()            # 10
lst.at(1)             # raises IndexError if out of bounds
lst.pop_front()
lst.pop_back()

for v in lst:         # iteration works
    pass
```

Also supports: `insert_at()`, `remove_at()`, `size()`, `is_empty()`, `clear()`, `copy()`, `len()`.

## Stack

`Stack` in `src/stack.py`

LIFO container built on Python list.

```python
from stack import Stack

s = Stack()
s.push(10)
s.push(20)
s.top()       # 20
s.pop()       # returns 20
s.size()      # 1
```

Methods: `push()`, `pop()`, `top()`, `size()`, `is_empty()`, `clear()`, `copy()`, `len()`, `bool()`.

## Queue

`Queue` in `src/queue.py`

FIFO container using circular buffer.

```python
from queue import Queue

q = Queue()
q.enqueue(10)
q.enqueue(20)
q.front()     # 10
q.dequeue()   # returns 10
q.size()      # 1
```

Methods: `enqueue()`, `dequeue()`, `front()`, `back()`, `size()`, `is_empty()`, `clear()`, `copy()`, `len()`, `bool()`.

## Hash Map

`HashMap` in `src/hash_map.py`

Key-value store with O(1) average operations using separate chaining.

```python
from hash_map import HashMap

m = HashMap()
m.put("one", 1)
m.put("two", 2)
m.get("one")       # 1
m.contains("two")  # True
m.remove("one")

m["key"] = "value" # __setitem__
m["key"]           # __getitem__
del m["key"]       # __delitem__
"key" in m         # __contains__
```

Methods: `put()`, `get()`, `get_or()`, `remove()`, `contains()`, `size()`, `is_empty()`, `clear()`, `keys()`, `values()`, `copy()`.

Rehashes when load factor exceeds 0.75.

## Binary Heap

`BinaryHeap` in `src/binary_heap.py`

Min-heap with O(log n) push/pop.

```python
from binary_heap import BinaryHeap

heap = BinaryHeap()
heap.push(30)
heap.push(10)
heap.push(20)
heap.peek()   # 10
heap.pop()    # returns 10
heap.pop()    # returns 20

heap = BinaryHeap.from_array([5, 3, 8, 1])  # O(n) heapify
```

Methods: `push()`, `pop()`, `peek()`, `size()`, `is_empty()`, `clear()`, `copy()`, `from_array()`, `len()`, `bool()`.

## Binary Search Tree

`BinarySearchTree` in `src/binary_search_tree.py`

Ordered binary tree with O(log n) average operations.

```python
from binary_search_tree import BinarySearchTree

bst = BinarySearchTree()
bst.insert(50)
bst.insert(30)
bst.insert(70)
bst.contains(30)   # True
bst.min()          # 30
bst.max()          # 70
bst.remove(30)
bst.in_order()     # [50, 70]
```

Methods: `insert()`, `remove()`, `contains()`, `min()`, `max()`, `size()`, `is_empty()`, `clear()`, `copy()`, `in_order()`, `pre_order()`, `post_order()`, `len()`, `in`, `iter()`.
