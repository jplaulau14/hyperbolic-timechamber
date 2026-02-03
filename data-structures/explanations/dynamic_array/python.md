# Dynamic Array — Python Implementation

## Why Dynamic Array?

### The Problem It Solves

Imagine you're building an application that collects user inputs. You don't know in advance how many inputs there will be—it could be 5, it could be 5,000. You need a container that:

1. Lets you access any element instantly by position (like a list)
2. Grows automatically when you add more elements
3. Doesn't waste huge amounts of memory "just in case"

A fixed-size array fails requirement #2—you'd have to guess the maximum size upfront and either waste memory or risk running out of space. A linked list fails requirement #1—finding the 500th element requires traversing 499 nodes first.

The **dynamic array** solves this by combining the best of both: instant access to any position AND automatic growth when needed.

### Real-World Analogies

**A filing cabinet with expandable folders**: You have a folder that currently holds 10 documents. When you need to add the 11th, instead of throwing away the folder and buying one twice as big, you keep working with what you have. But when it's truly full, you do upgrade to a bigger folder, move everything over, and continue. The upgrade is expensive, but it happens rarely.

**A concert venue with flexible seating**: The venue starts with 100 chairs. When ticket sales exceed 100, they don't add one chair at a time (too much setup work each time). Instead, they double the seating capacity. Most of the time, adding a guest is instant—just assign them a chair. Occasionally, there's a "reorganization day" where everything gets moved to a bigger space.

**A smartphone photo gallery**: Your phone doesn't allocate storage for exactly one photo at a time. It reserves space in chunks. When you take a photo, it usually goes into already-reserved space (fast). When that space fills up, the system allocates a bigger chunk (slower, but infrequent).

### When to Use It

- **Use a dynamic array when you need**:
  - Fast access to elements by index: O(1)
  - Fast append to the end: O(1) amortized
  - A contiguous block of elements (good for cache performance)
  - Memory efficiency (no per-element overhead like linked lists)

- **Unlike a linked list**, dynamic arrays give you instant access to any position without traversing from the head.

- **Unlike a fixed array**, dynamic arrays grow as needed—you don't have to predict the final size.

- **Avoid dynamic arrays when**:
  - You frequently insert/delete at the beginning or middle (O(n) shifts required)
  - You need guaranteed O(1) for every single operation (occasional O(n) resizes)

---

## Core Concept

### The Big Idea

A dynamic array maintains two key numbers: **size** (how many elements you're actually using) and **capacity** (how many elements the underlying storage can hold). Size is always less than or equal to capacity.

When you add an element and size equals capacity, the array allocates new storage with double the capacity, copies everything over, and then adds your element. This "double when full" strategy means that even though individual resizes are expensive (O(n)), they happen so rarely that the average cost per insertion stays O(1).

### Visual Representation: Size vs Capacity

```
Dynamic Array with size=3, capacity=6:

Indices:    [0]   [1]   [2]   [3]   [4]   [5]
           +-----+-----+-----+-----+-----+-----+
Values:    |  5  |  8  |  3  |     |     |     |
           +-----+-----+-----+-----+-----+-----+
            ^----- size=3 ----^
            ^---------- capacity=6 -----------^

- Slots 0-2: Active elements (within size)
- Slots 3-5: Reserved but unused (within capacity, beyond size)
```

The user sees an array with 3 elements. Internally, there's room for 6. The next 3 `push_back` operations will be instant—no reallocation needed.

### Key Terminology

- **Size**: The number of elements currently stored. This is what `len(arr)` returns.
- **Capacity**: The total number of elements the internal storage can hold before needing to grow.
- **Amortized O(1)**: "On average, O(1)" — individual operations might be O(n), but spread over many operations, the average is O(1).
- **Doubling strategy**: When capacity is exhausted, allocate 2x the current capacity. This ensures amortized O(1) insertion.
- **Reallocation**: Creating a larger backing store and copying all elements to it.

---

## How It Works: Step-by-Step

### Operation 1: `push_back(value)`

**What it does**: Adds an element to the end of the array.

**Step-by-step walkthrough**:

Starting state (size=2, capacity=2):
```
           [0]   [1]
          +-----+-----+
          |  5  |  8  |
          +-----+-----+
           ^-- size --^
           ^- capacity-^
```

We want to `push_back(3)`, but size == capacity. No room!

Step 1: Calculate new capacity (double it: 2 * 2 = 4)
```
New capacity = 4
```

Step 2: Allocate new storage with capacity 4
```
New storage:
           [0]   [1]   [2]   [3]
          +-----+-----+-----+-----+
          |     |     |     |     |
          +-----+-----+-----+-----+
```

Step 3: Copy existing elements to new storage
```
           [0]   [1]   [2]   [3]
          +-----+-----+-----+-----+
          |  5  |  8  |     |     |
          +-----+-----+-----+-----+
```

Step 4: Place new element at index `size` (which is 2)
```
           [0]   [1]   [2]   [3]
          +-----+-----+-----+-----+
          |  5  |  8  |  3  |     |
          +-----+-----+-----+-----+
```

Step 5: Increment size (2 -> 3)
```
Final state: size=3, capacity=4

           [0]   [1]   [2]   [3]
          +-----+-----+-----+-----+
          |  5  |  8  |  3  |     |
          +-----+-----+-----+-----+
           ^---- size=3 ----^
           ^----- capacity=4 -----^
```

**Why this approach?** The doubling strategy is key. If we only added 1 slot each time we ran out, every single insertion would require copying everything—that's O(n) per operation. By doubling, we ensure that after a resize, we get many "free" insertions before the next resize. This is what gives us amortized O(1).

### Operation 2: `pop_back()`

**What it does**: Removes the last element from the array.

**Step-by-step walkthrough**:

Starting state (size=3, capacity=4):
```
           [0]   [1]   [2]   [3]
          +-----+-----+-----+-----+
          |  5  |  8  |  3  |     |
          +-----+-----+-----+-----+
           ^---- size=3 ----^
```

We call `pop_back()`.

Step 1: Check if array is empty (size == 0). It's not, so continue.

Step 2: Decrement size (3 -> 2)
```
Final state: size=2, capacity=4

           [0]   [1]   [2]   [3]
          +-----+-----+-----+-----+
          |  5  |  8  |  3  |     |
          +-----+-----+-----+-----+
           ^- size=2 -^

Note: The value 3 is still in slot [2], but it's now
"beyond" size—invisible to the user and will be
overwritten by the next push_back.
```

**Why this approach?** We don't actually erase the value—we just move the size boundary. This makes `pop_back()` O(1). The value will be overwritten when a new element is pushed, or the slot will be reclaimed if we ever shrink the array.

### Operation 3: `reserve(new_cap)`

**What it does**: Ensures the array has at least `new_cap` capacity, allocating more space if needed.

**Step-by-step walkthrough**:

Starting state (size=2, capacity=2):
```
           [0]   [1]
          +-----+-----+
          |  5  |  8  |
          +-----+-----+
```

We call `reserve(5)` because we know we'll be adding 3 more elements.

Step 1: Check if new_cap > current capacity. 5 > 2, so proceed.

Step 2: Allocate new storage with capacity 5:
```
           [0]   [1]   [2]   [3]   [4]
          +-----+-----+-----+-----+-----+
          |     |     |     |     |     |
          +-----+-----+-----+-----+-----+
```

Step 3: Copy existing elements:
```
           [0]   [1]   [2]   [3]   [4]
          +-----+-----+-----+-----+-----+
          |  5  |  8  |     |     |     |
          +-----+-----+-----+-----+-----+
           ^-size=2-^
           ^------- capacity=5 --------^
```

**Why this approach?** If you know in advance how many elements you'll add, calling `reserve()` once avoids multiple reallocations. Adding 1000 elements to an empty array without `reserve` requires about 10 reallocations (1->2->4->8->16->32->64->128->256->512->1024). Calling `reserve(1000)` first means zero reallocations during insertion.

### Worked Example: Complete Sequence

Let's trace through building an array from scratch:

```
Operation 1: arr = DynamicArray()
             size=0, capacity=0
             [ empty ]

Operation 2: arr.push_back(10)
             size was 0 == capacity 0, must grow!
             new_cap = 1 (special case: 0 -> 1)

             [0]
            +----+
            | 10 |
            +----+
             size=1, capacity=1

Operation 3: arr.push_back(20)
             size=1 == capacity=1, must grow!
             new_cap = 2

             [0]   [1]
            +----+----+
            | 10 | 20 |
            +----+----+
             size=2, capacity=2

Operation 4: arr.push_back(30)
             size=2 == capacity=2, must grow!
             new_cap = 4

             [0]   [1]   [2]   [3]
            +----+----+----+----+
            | 10 | 20 | 30 |    |
            +----+----+----+----+
             size=3, capacity=4

Operation 5: arr.push_back(40)
             size=3 < capacity=4, no growth needed!

             [0]   [1]   [2]   [3]
            +----+----+----+----+
            | 10 | 20 | 30 | 40 |
            +----+----+----+----+
             size=4, capacity=4

Operation 6: arr.pop_back()
             size decrements: 4 -> 3

             [0]   [1]   [2]   [3]
            +----+----+----+----+
            | 10 | 20 | 30 | 40 |  <- 40 still here but invisible
            +----+----+----+----+
             size=3, capacity=4

Operation 7: arr.at(1) returns 20
             Direct access to index 1

Operation 8: arr.set_at(0, 99)

             [0]   [1]   [2]   [3]
            +----+----+----+----+
            | 99 | 20 | 30 | 40 |
            +----+----+----+----+
             size=3, capacity=4

After 8 operations:
- We grew the array 3 times (at operations 2, 3, 4)
- Operations 5-8 required no reallocation
- Current size: 3 elements
- Current capacity: 4 slots
```

---

## From Concept to Code

### The Data Structure

Before showing code, let's understand what we need to track:

| Field | Purpose |
|-------|---------|
| `_size` | Number of elements currently in the array (what the user sees) |
| `_capacity` | Total slots available in the backing storage |
| `_data` | The actual Python list that holds our elements |

Why separate `_size` and `_capacity`? The Python list `_data` always has length equal to `_capacity`. The `_size` tells us how many of those slots are "real" elements vs. unused placeholders.

### Python Implementation

```python
class DynamicArray:
    def __init__(self, size=0):
        if not isinstance(size, int) or size < 0:
            raise ValueError("size must be a non-negative integer")
        self._size = size
        self._capacity = size
        self._data = [None] * size if size > 0 else []
```

**Line-by-line breakdown**:

- `def __init__(self, size=0)`: Constructor with optional initial size. Default is empty array.
- `if not isinstance(size, int) or size < 0`: Input validation—reject floats, strings, and negative numbers.
- `raise ValueError(...)`: Python's standard exception for invalid arguments.
- `self._size = size`: Track how many elements we have.
- `self._capacity = size`: Initial capacity equals initial size (no extra room).
- `self._data = [None] * size if size > 0 else []`: Create backing storage. `[None] * 5` creates `[None, None, None, None, None]`. For size 0, we use an empty list to avoid creating a list with 0 Nones.

The underscore prefix (`_size`, `_data`) is a Python convention meaning "private—don't access directly from outside the class."

### Implementing `push_back`

**The algorithm in plain English**:
1. First, check if we're at capacity (size == capacity)
2. If at capacity, double the capacity (or go from 0 to 1)
3. Put the new value at index `size`
4. Increment `size` by 1

**The code**:
```python
def push_back(self, value):
    if self._size == self._capacity:
        new_cap = 1 if self._capacity == 0 else self._capacity * 2
        self.reserve(new_cap)
    self._data[self._size] = value
    self._size += 1
```

**Understanding the tricky parts**:

- `new_cap = 1 if self._capacity == 0 else self._capacity * 2`: A ternary expression. When starting from empty (capacity 0), we can't double 0, so we start with 1. Otherwise, double the current capacity.
- `self.reserve(new_cap)`: Delegate the actual reallocation to `reserve()`. This is good design—one place to handle reallocation logic.
- `self._data[self._size] = value`: Place the element at the first unused slot. Remember, `_size` is the count of elements, which also happens to be the index of the next empty slot (since indices are 0-based).
- `self._size += 1`: Now we have one more element.

### Implementing `reserve`

**The algorithm in plain English**:
1. If requested capacity isn't larger than current, do nothing
2. Create a new list with the new capacity (filled with None)
3. Copy all existing elements to the new list
4. Replace the old list with the new one
5. Update the capacity

**The code**:
```python
def reserve(self, new_cap):
    if new_cap <= self._capacity:
        return
    new_data = [None] * new_cap
    for i in range(self._size):
        new_data[i] = self._data[i]
    self._data = new_data
    self._capacity = new_cap
```

**Understanding the tricky parts**:

- `if new_cap <= self._capacity: return`: Early exit if no growth needed. Important for efficiency.
- `new_data = [None] * new_cap`: Allocate new storage. This is O(new_cap).
- `for i in range(self._size)`: Only copy elements up to `_size`, not `_capacity`. Slots beyond `_size` contain garbage we don't care about.
- `self._data = new_data`: Python's garbage collector will clean up the old list once nothing references it.

### Implementing Access Methods

**Bounds-checked access (`at` and `set_at`)**:
```python
def at(self, index):
    if index < 0 or index >= self._size:
        raise IndexError("DynamicArray.at: index out of range")
    return self._data[index]

def set_at(self, index, value):
    if index < 0 or index >= self._size:
        raise IndexError("DynamicArray.set_at: index out of range")
    self._data[index] = value
```

These methods explicitly reject negative indices. In Python, `list[-1]` gives you the last element, but that behavior can mask bugs. `at()` says "give me exactly index 0, 1, 2, etc."

**Pythonic subscript access (`__getitem__` and `__setitem__`)**:
```python
def __getitem__(self, index):
    return self._data[index]

def __setitem__(self, index, value):
    self._data[index] = value
```

These "dunder" (double-underscore) methods let you use `arr[0]` syntax. They delegate directly to the Python list, so negative indices work: `arr[-1]` returns the last element. No bounds checking against `_size`—this is a design choice matching how Python lists work.

### Implementing Pythonic Protocol Methods

**`__len__`**: Enables `len(arr)`
```python
def __len__(self):
    return self._size
```

Returns `_size`, not `len(self._data)`. The user cares about how many elements are stored, not the internal capacity.

**`__iter__`**: Enables `for item in arr:`
```python
def __iter__(self):
    for i in range(self._size):
        yield self._data[i]
```

This is a **generator function** (uses `yield`). It produces elements one at a time, only up to `_size`. Elements beyond `_size` are invisible during iteration.

**Why use `yield`?** Instead of creating a new list of all elements (O(n) space), we produce one element at a time (O(1) space). For large arrays, this matters.

---

## Complexity Analysis

### Time Complexity

| Operation | Best | Average | Worst | Why |
|-----------|------|---------|-------|-----|
| `push_back` | O(1) | O(1)* | O(n) | Usually just assigns to a slot. Worst case: must copy n elements to new storage. *Amortized O(1). |
| `pop_back` | O(1) | O(1) | O(1) | Just decrements `_size`. No copying or deallocation. |
| `at`/`set_at` | O(1) | O(1) | O(1) | Direct array index access. |
| `__getitem__`/`__setitem__` | O(1) | O(1) | O(1) | Same as above—delegates to Python list. |
| `front`/`back` | O(1) | O(1) | O(1) | Access index 0 or `size-1`. |
| `reserve` | O(n) | O(n) | O(n) | Must copy all n elements to new storage. |
| `clear` | O(1) | O(1) | O(1) | Just sets `_size = 0`. Doesn't touch data. |
| `data` | O(n) | O(n) | O(n) | Creates a slice copy of all elements. |
| `copy` | O(n) | O(n) | O(n) | Must copy all elements to new array. |

**Understanding amortized O(1) for `push_back`**:

Let's trace the cost of n insertions into an empty array:

```
Insert 1:  resize (1 copy)
Insert 2:  resize (2 copies)
Insert 3:  no resize
Insert 4:  resize (4 copies)
Insert 5-8:  no resize (4 inserts)
Insert 9:  resize (8 copies)
Insert 10-16: no resize (8 inserts)
...
```

Total copies for n insertions: 1 + 2 + 4 + 8 + ... + n = about 2n copies.

So n insertions cost O(2n) = O(n) total work, meaning each insertion costs O(n)/n = O(1) on average. This is what "amortized O(1)" means.

### Space Complexity

- **Overall structure**: O(n) where n is the number of elements. More precisely, O(capacity), but capacity is at most 2n.
- **Per operation**:
  - Most operations: O(1) extra space
  - `reserve`: O(new_cap) to create new storage (old storage is freed)
  - `data`: O(n) for the returned copy
  - `copy`: O(n) for the new array

### The 2x Memory Overhead

With the doubling strategy, capacity can be up to 2x the size. After inserting element number 513 into an array of 512:
- Size: 513
- Capacity: 1024
- Wasted slots: 511 (about 50%)

This is the trade-off for amortized O(1) insertions. If memory is critical, you could shrink capacity when size drops below capacity/4, but this implementation doesn't do that.

---

## Common Mistakes & Pitfalls

### Mistake 1: Confusing `at()` with `__getitem__`

```python
# Scenario: array has 3 elements

# Using at() - strict bounds checking
arr.at(-1)  # Raises IndexError! at() rejects negative indices

# Using [] - Python list semantics
arr[-1]  # Returns the last element (index 2)
```

**Why this matters**: `at()` is designed for when you want strict, explicit indexing—you ask for index 2, you get index 2, no magic. `__getitem__` follows Python conventions where negative indices wrap around. Choose based on whether negative indexing is a feature or a bug in your use case.

### Mistake 2: Storing `None` intentionally

```python
arr = DynamicArray()
arr.push_back(None)  # User wants to store None
arr.push_back(5)
arr.pop_back()
# Now arr has [None] internally but also unused slots with None

# Problem: How do you tell the difference?
print(arr._data)  # [None, 5] - but 5 is "invisible"
```

**Why this matters**: This implementation uses `None` as a placeholder for empty slots. If you need to store `None` as a valid value, you can't easily distinguish between "slot contains None that the user stored" and "slot is empty." Solutions include using a sentinel object or tracking validity separately.

### Mistake 3: Assuming `clear()` frees memory

```python
arr = DynamicArray()
for i in range(1000):
    arr.push_back(i)
# Now: size=1000, capacity=1024

arr.clear()
# Now: size=0, capacity=1024 (!)
# The 1024-slot list is still allocated
```

**Why this matters**: `clear()` sets size to 0 but doesn't shrink capacity. If you add 1 million elements then clear, you still have memory allocated for 1 million+ slots. To truly free memory, you'd need to do `arr._data = []` and `arr._capacity = 0` (or just create a new `DynamicArray`).

### Mistake 4: Shallow copy semantics with mutable objects

```python
original = DynamicArray()
original.push_back([1, 2, 3])  # Store a list

clone = original.copy()
clone.at(0).append(4)  # Modify the list in the clone

print(original.at(0))  # [1, 2, 3, 4] - Original changed too!
```

**Why this matters**: The `copy()` method creates a new DynamicArray but copies references to objects, not the objects themselves. For immutable objects (int, str, tuple), this is fine. For mutable objects (list, dict, other objects), both arrays point to the same underlying objects.

### Mistake 5: Iterating while modifying

```python
arr = DynamicArray()
arr.push_back(1)
arr.push_back(2)
arr.push_back(3)

for item in arr:
    if item == 2:
        arr.pop_back()  # Modifying during iteration!
```

**Why this matters**: This can skip elements or cause unexpected behavior. The iterator doesn't know you changed the array. If you need to remove elements while iterating, collect indices to remove, then remove in a separate pass (in reverse order).

---

## Practice Problems

To solidify your understanding, try implementing:

1. **`insert(index, value)`**: Insert a value at a specific index, shifting all subsequent elements right. Should be O(n) in the worst case.

2. **`remove(index)`**: Remove the element at index, shifting all subsequent elements left. Should be O(n) in the worst case.

3. **`shrink_to_fit()`**: Reduce capacity to match size exactly. Useful when you're done adding elements and want to reclaim memory.

4. **`__contains__(value)`**: Implement the `in` operator so `if 5 in arr:` works. Requires O(n) linear search.

5. **`reverse()`**: Reverse the array in-place in O(n) time without using extra space (beyond a few variables).

6. **Using DynamicArray as a stack**: Implement a `Stack` class that uses your `DynamicArray` internally. `push()` is `push_back()`, `pop()` returns and removes the last element, `peek()` returns without removing.

---

## Summary

### Key Takeaways

- A dynamic array separates **size** (elements stored) from **capacity** (space available).
- The **doubling strategy** ensures that `push_back` is O(1) amortized, even though individual resizes are O(n).
- `at()` provides strict bounds checking; `__getitem__` follows Python's negative-indexing conventions.
- `pop_back()` and `clear()` don't actually erase data—they just move the size boundary.
- Python's garbage collection handles memory automatically, but `clear()` doesn't shrink capacity.
- The `copy()` method is shallow—mutable objects are shared between original and copy.

### Quick Reference

```
DynamicArray — Growable array with O(1) access and amortized O(1) append
|-- push_back(value): O(1) amortized — Add element to end
|-- pop_back():       O(1)           — Remove last element
|-- at(index):        O(1)           — Bounds-checked read access
|-- set_at(i, val):   O(1)           — Bounds-checked write access
|-- __getitem__(i):   O(1)           — Subscript read (allows negative)
|-- __setitem__(i,v): O(1)           — Subscript write (allows negative)
|-- front() / back(): O(1)           — First / last element
|-- reserve(cap):     O(n)           — Pre-allocate capacity
|-- clear():          O(1)           — Remove all elements (keeps capacity)
|-- size() / len():   O(1)           — Number of elements
|-- capacity():       O(1)           — Current storage capacity
|-- empty():          O(1)           — True if size is 0
|-- data():           O(n)           — Copy of elements as list
|-- copy():           O(n)           — Shallow clone of array

Best for: Sequential access, append-heavy workloads, cache-friendly traversal
Avoid when: Frequent insert/delete in middle, need guaranteed O(1) for all ops
```
