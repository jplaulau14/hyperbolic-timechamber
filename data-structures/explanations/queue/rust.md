# Queue - Rust Implementation

## Why Queue?

### The Problem It Solves

Imagine you are at a busy coffee shop. Customers arrive, place their orders, and wait. The barista serves customers in the order they arrived: the first person to order is the first person to receive their coffee. This is fair, predictable, and prevents chaos.

Now imagine if the barista served customers randomly, or always served the most recent arrival first. People who arrived early would wait forever while newcomers cut ahead. The line would become unpredictable and frustrating.

A **queue** solves this problem in software. When you have tasks, messages, or data items that need to be processed in the order they arrived, a queue ensures **First-In, First-Out (FIFO)** ordering. The first item added is the first item removed.

### Real-World Analogies

1. **Checkout line at a grocery store**: Customers join at the back, are served from the front. The person who has been waiting longest gets helped first.

2. **Print job queue**: When you send documents to a printer, they print in the order submitted. Your document does not jump ahead of someone else's just because theirs is longer.

3. **Call center waiting system**: "Your call will be answered in the order it was received." Callers are placed in a queue and served one by one.

### When to Use It

- **Task scheduling**: Process jobs in the order they were submitted
- **Breadth-first search**: Explore graph nodes level by level
- **Message passing**: Handle requests in arrival order (web servers, event systems)
- **Buffering**: Smooth out differences between producer and consumer speeds
- **Simulation**: Model real-world waiting lines (traffic, customers, packets)

**Comparison to alternatives**:
- Unlike a **stack** (LIFO), a queue preserves arrival order
- Unlike a **priority queue**, all items are treated equally based on arrival time
- Unlike a **deque**, a queue only adds at one end and removes from the other

---

## Core Concept

### The Big Idea

A queue maintains two positions: where new items enter (the **back**) and where items leave (the **front**). Items travel through the queue like water through a pipe: what goes in first comes out first.

The key insight of this implementation is the **circular buffer**. Instead of shifting all elements when we remove from the front (which would be O(n)), we use two indices that wrap around the array. When an index reaches the end, it wraps back to position 0. This gives us O(1) operations for both enqueue and dequeue.

### Visual Representation

A queue with capacity 6, containing elements [A, B, C, D]:

```
Indices:     0     1     2     3     4     5
           +-----+-----+-----+-----+-----+-----+
           |  A  |  B  |  C  |  D  |     |     |
           +-----+-----+-----+-----+-----+-----+
              ^                       ^
             head                    tail
            (front)               (next insert)
```

- **head**: Points to the front element (next to be dequeued)
- **tail**: Points to where the next element will be inserted
- Elements occupy indices from `head` to `tail-1`

### Key Terminology

- **Enqueue**: Add an element to the back of the queue
- **Dequeue**: Remove and return the element from the front
- **Front**: The oldest element, next to be removed
- **Back**: The newest element, most recently added
- **Circular buffer**: An array where indices wrap around from the end back to the beginning
- **Capacity**: Total number of slots allocated
- **Size**: Current number of elements stored

---

## How It Works: Step-by-Step

### Operation 1: Enqueue (Add to Back)

**What it does**: Places a new element at the tail position and advances tail.

**Step-by-step walkthrough**:

Starting state (capacity=4, size=2):
```
Indices:     0     1     2     3
           +-----+-----+-----+-----+
           |  10 |  20 |     |     |
           +-----+-----+-----+-----+
              ^           ^
            head=0      tail=2
            size=2
```

Step 1: Check if queue is full (size == capacity). It is not (2 < 4), so no resize needed.

Step 2: Write the value (30) at position `tail` (index 2).
```
Indices:     0     1     2     3
           +-----+-----+-----+-----+
           |  10 |  20 |  30 |     |
           +-----+-----+-----+-----+
              ^                 ^
            head=0           tail=2 (write here)
```

Step 3: Advance tail using modular arithmetic: `tail = (tail + 1) % capacity = (2 + 1) % 4 = 3`.

Step 4: Increment size.

Final state:
```
Indices:     0     1     2     3
           +-----+-----+-----+-----+
           |  10 |  20 |  30 |     |
           +-----+-----+-----+-----+
              ^                 ^
            head=0           tail=3
            size=3
```

**Why this approach?** By writing at tail and advancing it, we append in O(1) time. No elements need to move.

### Operation 2: Dequeue (Remove from Front)

**What it does**: Returns the front element and advances head.

**Step-by-step walkthrough**:

Starting state (continuing from above):
```
Indices:     0     1     2     3
           +-----+-----+-----+-----+
           |  10 |  20 |  30 |     |
           +-----+-----+-----+-----+
              ^                 ^
            head=0           tail=3
            size=3
```

Step 1: Check if queue is empty (size == 0). It is not, so proceed.

Step 2: Read the value at position `head` (index 0). We get 10.
```
           +-----+-----+-----+-----+
           |  10 |  20 |  30 |     |
           +-----+-----+-----+-----+
              ^
            Read 10 from here
```

Step 3: Advance head: `head = (head + 1) % capacity = (0 + 1) % 4 = 1`.

Step 4: Decrement size.

Step 5: Return `Some(10)`.

Final state:
```
Indices:     0     1     2     3
           +-----+-----+-----+-----+
           |  _  |  20 |  30 |     |
           +-----+-----+-----+-----+
                    ^           ^
                  head=1     tail=3
                  size=2
```

Note: Index 0 still contains old data, but it is logically "empty" (not between head and tail).

**Why this approach?** By advancing head instead of shifting elements, we remove in O(1) time.

### Operation 3: Circular Wrap-Around

**What it does**: When tail reaches the end, it wraps to index 0, reusing freed space.

**Step-by-step walkthrough**:

Starting state after several operations (capacity=4):
```
Indices:     0     1     2     3
           +-----+-----+-----+-----+
           |  _  |  _  |  30 |  40 |
           +-----+-----+-----+-----+
                          ^
                        head=2
            tail would be at 4, but capacity is 4...
            size=2
```

Now we enqueue 50. Tail is at 4, but there is no index 4.

Step 1: `tail = 4 % 4 = 0`. Tail wraps to the beginning!

Step 2: Write 50 at index 0.
```
Indices:     0     1     2     3
           +-----+-----+-----+-----+
           |  50 |  _  |  30 |  40 |
           +-----+-----+-----+-----+
              ^           ^
           tail=0       head=2
           (after write, tail becomes 1)
           size=3
```

Step 3: Advance tail to 1, increment size to 3.

**This is the circular magic**: The buffer wraps around, reusing space freed by dequeue. Elements are stored at indices 2, 3, 0 in logical order [30, 40, 50].

### Worked Example: Complete Sequence

Let us trace through a realistic sequence of operations on a queue with initial capacity 4:

```
Operation 1: enqueue(10)
+-----+-----+-----+-----+
|  10 |     |     |     |
+-----+-----+-----+-----+
   ^     ^
  H=0   T=1    size=1


Operation 2: enqueue(20)
+-----+-----+-----+-----+
|  10 |  20 |     |     |
+-----+-----+-----+-----+
   ^           ^
  H=0         T=2    size=2


Operation 3: enqueue(30)
+-----+-----+-----+-----+
|  10 |  20 |  30 |     |
+-----+-----+-----+-----+
   ^                 ^
  H=0               T=3    size=3


Operation 4: dequeue() -> returns Some(10)
+-----+-----+-----+-----+
|  _  |  20 |  30 |     |
+-----+-----+-----+-----+
         ^           ^
        H=1         T=3    size=2


Operation 5: dequeue() -> returns Some(20)
+-----+-----+-----+-----+
|  _  |  _  |  30 |     |
+-----+-----+-----+-----+
               ^     ^
              H=2   T=3    size=1


Operation 6: enqueue(40)
+-----+-----+-----+-----+
|  _  |  _  |  30 |  40 |
+-----+-----+-----+-----+
               ^           (T wraps)
              H=2   T=0    size=2

Wait, T=0? Yes! tail = (3+1) % 4 = 0


Operation 7: enqueue(50)
+-----+-----+-----+-----+
|  50 |  _  |  30 |  40 |
+-----+-----+-----+-----+
   ^           ^
  T=1         H=2    size=3

Logical order: [30, 40, 50] (reading from H forward, wrapping)


Operation 8: front() -> returns Some(&30)
(No state change, just peeks at head)


Operation 9: back() -> returns Some(&50)
(No state change, peeks at tail-1 = index 0)


Operation 10: dequeue() -> returns Some(30)
+-----+-----+-----+-----+
|  50 |  _  |  _  |  40 |
+-----+-----+-----+-----+
   ^                 ^
  T=1               H=3    size=2

Logical order: [40, 50]
```

---

## From Concept to Code

### The Data Structure

Before showing code, let us understand what we need to track:

1. **data**: A pointer to heap-allocated memory holding our elements
2. **head**: Index of the front element
3. **tail**: Index where the next element will be inserted
4. **size**: How many elements are currently in the queue
5. **capacity**: How many slots are allocated

Why each field exists:
- `data` holds the actual storage (on the heap, so it can grow)
- `head` and `tail` let us add/remove in O(1) without shifting
- `size` tells us how many valid elements exist (distinguishes full from empty)
- `capacity` tells us when we need to grow

### Rust Implementation

```rust
pub struct Queue<T> {
    data: *mut T,
    head: usize,
    tail: usize,
    size: usize,
    capacity: usize,
}
```

**Line-by-line breakdown**:

- `pub struct Queue<T>`: A generic queue that works with any type `T`. The `<T>` is a type parameter, like a placeholder that gets filled in when you create a `Queue<i32>` or `Queue<String>`.

- `data: *mut T`: A **raw mutable pointer** to type `T`. Unlike safe references (`&T` or `&mut T`), raw pointers can be null and require `unsafe` code to dereference. We use this for manual memory control.

- `head: usize`, `tail: usize`: Indices into the buffer. `usize` is an unsigned integer sized for array indexing (32-bit on 32-bit systems, 64-bit on 64-bit systems).

- `size: usize`: Current element count. We cannot just use `tail - head` because of wrap-around.

- `capacity: usize`: Total slots allocated. When `size == capacity`, we need to grow.

**Why raw pointers instead of Vec?**

This implementation deliberately uses raw pointers to demonstrate manual memory management in Rust. A production implementation would typically use `Vec<T>` or `VecDeque<T>` from the standard library, which handle allocation safely. Here, we take on that responsibility ourselves to understand what happens under the hood.

### Implementing `new()` and `with_capacity()`

**The algorithm in plain English**:
1. For `new()`: Create an empty queue with no allocation (null pointer, zero capacity)
2. For `with_capacity()`: Allocate memory for the requested number of elements

**The code**:

```rust
impl<T> Queue<T> {
    pub fn new() -> Self {
        Self {
            data: core::ptr::null_mut(),
            head: 0,
            tail: 0,
            size: 0,
            capacity: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        let layout = core::alloc::Layout::array::<T>(capacity).unwrap();
        let data = unsafe { std::alloc::alloc(layout) as *mut T };
        if data.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self {
            data,
            head: 0,
            tail: 0,
            size: 0,
            capacity,
        }
    }
}
```

**Understanding the tricky parts**:

- `core::ptr::null_mut()`: Creates a null raw pointer. This is the "no allocation" state. We cannot dereference a null pointer (that would be undefined behavior), so we must check for it before use.

- `Layout::array::<T>(capacity)`: Creates a memory layout describing how much space we need for `capacity` elements of type `T`, with proper alignment. Different types have different sizes and alignment requirements; this handles that automatically.

- `unsafe { std::alloc::alloc(layout) as *mut T }`: This is where we allocate raw memory from the system. It is `unsafe` because:
  - The returned memory is uninitialized (reading it before writing is undefined behavior)
  - We are responsible for freeing it later
  - We must ensure the layout is valid

- `handle_alloc_error(layout)`: If allocation fails (returns null), this aborts the program. In real code, you might want to return a `Result` instead.

### Implementing `enqueue()`

**The algorithm in plain English**:
1. If the queue is full, grow the buffer
2. Write the new value at the tail position
3. Advance tail (wrapping around if necessary)
4. Increment size

**The code**:

```rust
pub fn enqueue(&mut self, value: T) {
    if self.size == self.capacity {
        self.grow();
    }
    unsafe {
        core::ptr::write(self.data.add(self.tail), value);
    }
    self.tail = (self.tail + 1) % self.capacity;
    self.size += 1;
}
```

**Understanding the tricky parts**:

- `&mut self`: We need mutable access because we are modifying the queue.

- `core::ptr::write(...)`: This writes a value to a memory location **without reading what was there first**. This is crucial! Normal assignment (`*ptr = value`) would:
  1. Read the old value
  2. Drop (destroy) the old value
  3. Write the new value

  But our memory is uninitialized! Reading it is undefined behavior. `ptr::write` skips steps 1-2 and just writes.

- `self.data.add(self.tail)`: Pointer arithmetic. `data.add(n)` gives a pointer to the `n`th element after `data`. This is `unsafe` because we must ensure `n` is within bounds.

- `(self.tail + 1) % self.capacity`: The modulo operator (`%`) causes wrap-around. If `tail` is at the last index and we add 1, modulo brings it back to 0.

### Implementing `dequeue()`

**The algorithm in plain English**:
1. If the queue is empty, return None
2. Read the value at the head position
3. Advance head (wrapping around if necessary)
4. Decrement size
5. Return the value wrapped in Some

**The code**:

```rust
pub fn dequeue(&mut self) -> Option<T> {
    if self.size == 0 {
        return None;
    }
    let value = unsafe { core::ptr::read(self.data.add(self.head)) };
    self.head = (self.head + 1) % self.capacity;
    self.size -= 1;
    Some(value)
}
```

**Understanding the tricky parts**:

- `Option<T>`: Rust's way of handling "might not exist". Instead of returning `null` or throwing an exception, we return:
  - `Some(value)` if there is a value
  - `None` if the queue is empty

  The caller must handle both cases, preventing forgotten null checks.

- `core::ptr::read(...)`: Reads a value from memory **without invalidating the source**. Normally, when you move a value in Rust, the source becomes invalid. But with raw pointers, Rust does not track this. `ptr::read` copies the bytes out, and we are responsible for not reading them again. We accomplish this by advancing `head` past this location.

- Returning `Some(value)`: Wraps the value in the "success" variant of `Option`.

### Implementing `front()` and `back()`

**The code**:

```rust
pub fn front(&self) -> Option<&T> {
    if self.size == 0 {
        None
    } else {
        unsafe { Some(&*self.data.add(self.head)) }
    }
}

pub fn back(&self) -> Option<&T> {
    if self.size == 0 {
        None
    } else {
        let idx = if self.tail == 0 { self.capacity - 1 } else { self.tail - 1 };
        unsafe { Some(&*self.data.add(idx)) }
    }
}
```

**Understanding the tricky parts**:

- `&self`: We only need shared (read-only) access. This allows multiple simultaneous calls to `front()`.

- `Option<&T>`: Returns an optional **reference** to the element. The `&T` means "borrowed reference to a T". The element stays in the queue; we are just looking at it.

- `&*self.data.add(self.head)`: This is a two-step operation:
  1. `self.data.add(self.head)` gives us a raw pointer `*mut T`
  2. `*...` dereferences it to get `T`
  3. `&...` takes a reference to get `&T`

  Combined: "get a reference to the element at position head"

- `let idx = if self.tail == 0 { self.capacity - 1 } else { self.tail - 1 }`: The back element is at `tail - 1`, but if `tail` is 0, we need to wrap to `capacity - 1`. This is the inverse of the modulo trick.

### Implementing `grow()`

**The algorithm in plain English**:
1. Calculate new capacity (double, or 1 if currently 0)
2. Allocate new buffer
3. Copy elements from old buffer to new, linearizing the circular arrangement
4. Deallocate old buffer
5. Update pointers and indices

**The code**:

```rust
fn grow(&mut self) {
    let new_cap = if self.capacity == 0 { 1 } else { self.capacity * 2 };
    let new_layout = core::alloc::Layout::array::<T>(new_cap).unwrap();
    let new_data = unsafe { std::alloc::alloc(new_layout) as *mut T };
    if new_data.is_null() {
        std::alloc::handle_alloc_error(new_layout);
    }

    for i in 0..self.size {
        let src_idx = (self.head + i) % self.capacity;
        unsafe {
            core::ptr::copy_nonoverlapping(self.data.add(src_idx), new_data.add(i), 1);
        }
    }

    if !self.data.is_null() {
        let old_layout = core::alloc::Layout::array::<T>(self.capacity).unwrap();
        unsafe { std::alloc::dealloc(self.data as *mut u8, old_layout) };
    }

    self.data = new_data;
    self.head = 0;
    self.tail = self.size;
    self.capacity = new_cap;
}
```

**Understanding the tricky parts**:

- `copy_nonoverlapping`: Copies bytes from source to destination. "Nonoverlapping" means the source and destination must not overlap in memory (which they do not, since we allocated a new buffer). This is a bitwise copy, not a Rust "move"---the source bytes are not invalidated by Rust's ownership system.

- **Linearizing the circular buffer**: Elements might wrap around (e.g., indices 3, 0, 1, 2 in a 4-element buffer). We copy them to the new buffer in logical order (indices 0, 1, 2, 3), resetting head to 0.

- **Why we do not drop source elements**: After `copy_nonoverlapping`, the old buffer still contains the byte patterns of our elements. But we are not going to read them again---we deallocate the buffer. If we called `Drop` on them, we would be double-dropping (the new buffer also "has" them). So we just deallocate the raw memory.

- `dealloc(self.data as *mut u8, old_layout)`: Frees the old buffer. The cast to `*mut u8` is required by the dealloc signature.

### Implementing `Clone`

The `Clone` trait allows creating a deep copy of the queue.

```rust
impl<T: Clone> Clone for Queue<T> {
    fn clone(&self) -> Self {
        let mut new_queue = Self::with_capacity(self.capacity);
        for i in 0..self.size {
            let idx = (self.head + i) % self.capacity;
            let value = unsafe { (*self.data.add(idx)).clone() };
            new_queue.enqueue(value);
        }
        new_queue
    }
}
```

**Understanding the tricky parts**:

- `T: Clone`: This implementation only exists when the element type `T` implements `Clone`. You cannot clone a `Queue<File>` because files cannot be meaningfully cloned.

- `(*self.data.add(idx)).clone()`:
  1. `self.data.add(idx)` - pointer to element
  2. `*...` - dereference to get the element
  3. `.clone()` - create a copy of the element

- We enqueue each cloned element into the new queue, which handles the circular buffer logic.

### Implementing `Drop`

The `Drop` trait is Rust's destructor. It runs when the queue goes out of scope.

```rust
impl<T> Drop for Queue<T> {
    fn drop(&mut self) {
        self.clear();
        if !self.data.is_null() {
            let layout = core::alloc::Layout::array::<T>(self.capacity).unwrap();
            unsafe { std::alloc::dealloc(self.data as *mut u8, layout) };
        }
    }
}
```

**Understanding the tricky parts**:

- `self.clear()`: Removes all elements by calling `dequeue()` repeatedly. This is crucial because each element might have its own `Drop` implementation (e.g., `String` needs to free its buffer). If we just deallocated the buffer without clearing, those destructors would never run, causing memory leaks.

- Order matters: clear first, then deallocate. If we deallocated first, `clear()` would access freed memory.

### Thread Safety: `Send` and `Sync`

```rust
unsafe impl<T: Send> Send for Queue<T> {}
unsafe impl<T: Sync> Sync for Queue<T> {}
```

**Understanding the tricky parts**:

- `Send`: A type is `Send` if it can be transferred to another thread. `Queue<T>` is `Send` if `T` is `Send`, because we own all our data.

- `Sync`: A type is `Sync` if `&T` (a shared reference) can be sent to another thread. `Queue<T>` is `Sync` if `T` is `Sync`, because read-only access to the queue is safe to share.

- These are `unsafe impl` because we are making a promise the compiler cannot verify: that our raw pointer usage is actually thread-safe. We must manually ensure this is true.

---

## Complexity Analysis

### Time Complexity

| Operation | Best | Average | Worst | Why |
|-----------|------|---------|-------|-----|
| enqueue   | O(1) | O(1)    | O(n)  | Usually just writes at tail. Worst case triggers resize. |
| dequeue   | O(1) | O(1)    | O(1)  | Always just reads at head and increments. |
| front     | O(1) | O(1)    | O(1)  | Direct array access via head index. |
| back      | O(1) | O(1)    | O(1)  | Direct array access via (tail-1). |
| size      | O(1) | O(1)    | O(1)  | Returns stored field. |
| is_empty  | O(1) | O(1)    | O(1)  | Compares size to 0. |
| clear     | O(n) | O(n)    | O(n)  | Must drop each element individually. |
| clone     | O(n) | O(n)    | O(n)  | Must clone each element. |
| grow      | O(n) | O(n)    | O(n)  | Copies all n elements to new buffer. |

**Understanding the "Why" column**:

- **enqueue O(n) worst case**: When full, we allocate a new buffer twice the size and copy all n elements. However, this happens rarely enough that the *amortized* cost is O(1).

- **clear O(n)**: We call `dequeue()` n times. Each dequeue is O(1), so total is O(n). We cannot skip this because elements may have destructors that need to run.

### Space Complexity

- **Overall structure**: O(n) where n is capacity. We allocate an array of capacity slots.
- **Per enqueue**: O(1) normally, O(n) during resize (allocates new array).
- **Per dequeue**: O(1), no allocation.

### Amortized Analysis

**What is amortized analysis?**

When an operation is usually cheap but occasionally expensive, amortized analysis averages the cost over many operations.

**Why enqueue is amortized O(1)**:

Consider inserting n elements into an initially empty queue:
- We resize at sizes 1, 2, 4, 8, 16, ..., n
- Resize costs are 1 + 2 + 4 + 8 + ... + n = 2n - 1 (approximately 2n)
- Plus n regular insertions at O(1) each

Total cost: approximately 3n operations for n insertions.
Average cost per insertion: 3n/n = 3 = O(1).

Even though individual resizes are O(n), they happen infrequently enough that the average is constant.

---

## Common Mistakes and Pitfalls

### Mistake 1: Forgetting to Handle Wrap-Around

```rust
// Wrong: assumes elements are contiguous
pub fn bad_back(&self) -> Option<&T> {
    if self.size == 0 {
        None
    } else {
        unsafe { Some(&*self.data.add(self.tail - 1)) }
    }
}
```

```rust
// Right: handles wrap-around
pub fn back(&self) -> Option<&T> {
    if self.size == 0 {
        None
    } else {
        let idx = if self.tail == 0 { self.capacity - 1 } else { self.tail - 1 };
        unsafe { Some(&*self.data.add(idx)) }
    }
}
```

**Why this matters**: When `tail` is 0, subtracting 1 causes underflow (wraps to maximum usize), accessing invalid memory. The circular buffer means the last element might be at the end of the array even when tail is at the beginning.

### Mistake 2: Using Assignment Instead of ptr::write

```rust
// Wrong: tries to drop uninitialized memory
pub fn bad_enqueue(&mut self, value: T) {
    if self.size == self.capacity {
        self.grow();
    }
    unsafe {
        *self.data.add(self.tail) = value;  // UNDEFINED BEHAVIOR
    }
    self.tail = (self.tail + 1) % self.capacity;
    self.size += 1;
}
```

```rust
// Right: uses ptr::write for uninitialized memory
pub fn enqueue(&mut self, value: T) {
    if self.size == self.capacity {
        self.grow();
    }
    unsafe {
        core::ptr::write(self.data.add(self.tail), value);
    }
    self.tail = (self.tail + 1) % self.capacity;
    self.size += 1;
}
```

**Why this matters**: The `=` operator drops the old value before writing the new one. If the memory is uninitialized (contains garbage), dropping it is undefined behavior. For types with destructors (like `String`), this could try to free invalid pointers.

### Mistake 3: Not Clearing Before Deallocating

```rust
// Wrong: leaks memory for types with destructors
impl<T> Drop for Queue<T> {
    fn drop(&mut self) {
        if !self.data.is_null() {
            let layout = core::alloc::Layout::array::<T>(self.capacity).unwrap();
            unsafe { std::alloc::dealloc(self.data as *mut u8, layout) };
        }
        // Elements never dropped!
    }
}
```

```rust
// Right: drops elements before freeing buffer
impl<T> Drop for Queue<T> {
    fn drop(&mut self) {
        self.clear();  // Drop all elements first
        if !self.data.is_null() {
            let layout = core::alloc::Layout::array::<T>(self.capacity).unwrap();
            unsafe { std::alloc::dealloc(self.data as *mut u8, layout) };
        }
    }
}
```

**Why this matters**: If `T` is `String`, each string owns heap memory. Deallocating the queue's buffer without running each string's destructor leaks that memory. Always drop elements before their container.

### Mistake 4: Confusing Size and Capacity

```rust
// Wrong: uses capacity instead of size
pub fn bad_is_empty(&self) -> bool {
    self.capacity == 0  // Wrong! A queue with capacity but no elements is still empty
}

pub fn bad_full_check(&mut self, value: T) {
    if self.tail == self.capacity {  // Wrong! Doesn't account for wrap-around
        self.grow();
    }
    // ...
}
```

```rust
// Right: uses size for logical emptiness/fullness
pub fn is_empty(&self) -> bool {
    self.size == 0
}

pub fn enqueue(&mut self, value: T) {
    if self.size == self.capacity {
        self.grow();
    }
    // ...
}
```

**Why this matters**: Capacity is how much memory is allocated. Size is how many elements exist. A queue with capacity 100 and size 0 is empty. A queue with capacity 100 and size 100 is full. The circular nature means tail can be less than head, so comparing tail to capacity does not tell you if the queue is full.

---

## Practice Problems

To solidify your understanding, try implementing:

1. **peek_nth()**: Return a reference to the nth element from the front without removing it. Handle wrap-around correctly.

2. **iter()**: Implement the `Iterator` trait to allow `for` loops over queue elements. Consider: should iteration consume the queue or just borrow it?

3. **shrink_to_fit()**: Reduce capacity to match size (or some minimum). Handle the case where elements wrap around.

4. **try_enqueue()**: A version that returns `Result<(), T>` instead of growing. Useful for bounded queues.

5. **Level-order traversal**: Given a binary tree, use your queue to perform breadth-first traversal, printing nodes level by level.

---

## Summary

### Key Takeaways

- A **queue** is a First-In-First-Out data structure: elements leave in the order they arrived.

- The **circular buffer** technique uses modular arithmetic to wrap indices, achieving O(1) enqueue and dequeue without shifting elements.

- **Raw pointers** in Rust require `unsafe` code and manual management:
  - Use `ptr::write` for uninitialized memory
  - Use `ptr::read` to move out without double-dropping
  - Always deallocate what you allocate

- **Option<T>** is Rust's way of representing "might be empty" without null pointers. It forces callers to handle the empty case.

- **Clone** creates independent copies; **Drop** ensures cleanup when the queue is destroyed.

- **Amortized analysis** shows that even though resize is O(n), enqueue averages O(1) over many operations.

### Quick Reference

```
Queue<T> - First-In-First-Out collection using circular buffer

Operations:
  enqueue(value)  O(1) amortized  Add to back
  dequeue()       O(1)            Remove from front, returns Option<T>
  front()         O(1)            Peek front, returns Option<&T>
  back()          O(1)            Peek back, returns Option<&T>
  size()          O(1)            Number of elements
  is_empty()      O(1)            True if size is 0
  clear()         O(n)            Remove all elements

Traits implemented:
  Default         Creates empty queue
  Clone           Deep copy (requires T: Clone)
  Drop            Cleanup on destruction
  Send/Sync       Thread safety (requires T: Send/Sync)

Best for:
  - Processing items in arrival order
  - Breadth-first search algorithms
  - Task scheduling and message passing
  - Buffering between producer and consumer

Avoid when:
  - You need LIFO (use Stack instead)
  - You need priority ordering (use priority queue)
  - You need access to middle elements (use Vec or deque)
```
