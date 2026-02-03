# Dynamic Array - Rust Implementation

## Why Dynamic Array?

### The Problem It Solves

Imagine you're writing a program that needs to collect user input. You don't know how many items the user will enter - it could be 5, or 50, or 5000. With a regular fixed-size array, you'd have to guess:

```rust
let items: [i32; 100] = [0; 100];  // Hope the user doesn't enter more than 100!
```

This is wasteful if they only enter 3 items, and catastrophic if they try to enter 101. What you really want is an array that grows as needed - you push items in, and it magically handles storage for you.

This is exactly what a **dynamic array** provides: the random-access speed of arrays combined with the flexibility to grow on demand.

### Real-World Analogies

**A filing cabinet with expandable folders**: Each folder starts small, but as you add more papers, it expands. If a folder gets too full, you get a bigger folder and transfer everything over.

**A restaurant waitlist**: Starts with space for a few names. As more people arrive, the host gets a new, larger sheet and copies the existing names over, leaving room for more.

**A digital photo album**: When you start, it reserves space for some photos. As you add more, it automatically allocates more storage, moving existing photos to a larger space if needed.

### When to Use It

- **Use a dynamic array when you need:**
  - Fast random access by index: O(1)
  - Primarily adding/removing from the end
  - Contiguous memory layout (cache-friendly iteration)
  - Unknown or variable amount of data

- **Unlike a linked list**, dynamic arrays give you instant access to any element by index.

- **Unlike a fixed array**, you don't need to know the size upfront.

- **Avoid when** you need frequent insertions/deletions in the middle (linked lists are better) or when memory is extremely constrained and you can't afford the occasional reallocation.

---

## Core Concept

### The Big Idea

A dynamic array maintains a block of contiguous memory larger than what's currently being used. It tracks two sizes: the **capacity** (total slots available) and the **size** (slots actually containing elements). When size reaches capacity and you try to add another element, the array allocates a new, larger block (typically double the size), copies everything over, and frees the old block.

This "double when full" strategy gives us **amortized O(1)** insertion - most inserts are instant, and the occasional expensive copy is spread across all the cheap inserts that preceded it.

### Visual Representation

```
Capacity vs Size
================

Initial state (capacity=4, size=0):
+---+---+---+---+
| _ | _ | _ | _ |   (_ = uninitialized)
+---+---+---+---+
  0   1   2   3

After push_back(10), push_back(20) (capacity=4, size=2):
+----+----+---+---+
| 10 | 20 | _ | _ |
+----+----+---+---+
  0    1    2   3
       ^
       size points here (next write position)

After two more pushes (capacity=4, size=4 - FULL!):
+----+----+----+----+
| 10 | 20 | 30 | 40 |
+----+----+----+----+
  0    1    2    3

Next push_back(50) triggers growth to capacity=8:
+----+----+----+----+----+---+---+---+
| 10 | 20 | 30 | 40 | 50 | _ | _ | _ |
+----+----+----+----+----+---+---+---+
  0    1    2    3    4    5   6   7
```

### Key Terminology

- **Size**: The number of elements currently stored. Valid indices are 0 to size-1.
- **Capacity**: The total number of elements the array can hold before needing to reallocate.
- **Reallocation/Growth**: The process of allocating a larger buffer, copying existing elements, and freeing the old buffer.
- **Amortized O(1)**: The average time per operation over a sequence of operations, accounting for occasional expensive operations.
- **Raw pointer (`*mut T`)**: A pointer with no ownership or borrowing guarantees - Rust's escape hatch for manual memory management.

---

## How It Works: Step-by-Step

### Operation 1: push_back

**What it does**: Adds an element to the end of the array, growing capacity if necessary.

**Step-by-step walkthrough**:

Starting state (capacity=2, size=2 - array is full):
```
+---+---+
| 5 | 3 |
+---+---+
  0   1
capacity=2, size=2
```

Step 1: Check if we have room. size == capacity, so we need to grow.
```
new_capacity = capacity * 2 = 4
Allocate new buffer of size 4
```

Step 2: Copy existing elements to new buffer.
```
Old:  +---+---+
      | 5 | 3 |
      +---+---+
        |   |
        v   v
New:  +---+---+---+---+
      | 5 | 3 | _ | _ |
      +---+---+---+---+
```

Step 3: Free old buffer, update pointer and capacity.
```
+---+---+---+---+
| 5 | 3 | _ | _ |
+---+---+---+---+
capacity=4, size=2
```

Step 4: Write the new element at index `size`, then increment size.
```
+---+---+---+---+
| 5 | 3 | 8 | _ |
+---+---+---+---+
  0   1   2   3
capacity=4, size=3
```

**Why this approach?** Doubling the capacity means we copy n elements only after n insertions without copying. This amortizes the copy cost across all insertions, giving O(1) amortized time.

### Operation 2: pop_back

**What it does**: Removes and returns the last element, or returns `None` if empty.

**Step-by-step walkthrough**:

Starting state:
```
+---+---+---+---+
| 5 | 3 | 8 | _ |
+---+---+---+---+
capacity=4, size=3
```

Step 1: Check if empty. size > 0, so we can proceed.

Step 2: Decrement size first (from 3 to 2).
```
+---+---+---+---+
| 5 | 3 | 8 | _ |
+---+---+---+---+
          ^
          This is now at index `size` (out of bounds for access)
capacity=4, size=2
```

Step 3: Read the value at the old position and return it.
```
Return: Some(8)

Logical view (size=2, the "8" is still in memory but not accessible):
+---+---+---+---+
| 5 | 3 | ? | _ |
+---+---+---+---+
capacity=4, size=2
```

**Why this approach?** We use `ptr::read()` to move the value out of the array without leaving a duplicate. The memory still holds the bit pattern, but we've logically removed the element by decrementing size. The caller now owns the returned value.

### Operation 3: reserve

**What it does**: Ensures capacity is at least the specified amount, reallocating if necessary.

Starting state:
```
+---+---+
| 1 | 2 |
+---+---+
capacity=2, size=2
```

Call `reserve(5)`:

Step 1: Check if new_cap > capacity. 5 > 2, so proceed.

Step 2: Allocate new buffer of capacity 5.

Step 3: Copy existing elements.
```
+---+---+---+---+---+
| 1 | 2 | _ | _ | _ |
+---+---+---+---+---+
```

Step 4: Deallocate old buffer, update pointer and capacity.
```
capacity=5, size=2
```

**Why this approach?** Pre-reserving capacity when you know how many elements you'll add avoids multiple reallocations. If you're going to add 1000 elements, calling `reserve(1000)` first means one allocation instead of ~10 (since we double each time: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024).

### Worked Example: Complete Sequence

Let's trace through a realistic sequence of operations:

```
Operation 1: let arr = DynamicArray::new()
State: data=null, capacity=0, size=0
+---+
|   |  (no allocation yet)
+---+
```

```
Operation 2: arr.push_back(10)
- capacity=0, need to grow
- new_cap = 1 (special case: 0 becomes 1)
- allocate 1 slot, write 10
State: capacity=1, size=1
+----+
| 10 |
+----+
```

```
Operation 3: arr.push_back(20)
- size=1, capacity=1, need to grow
- new_cap = 2
- allocate 2 slots, copy 10, write 20
State: capacity=2, size=2
+----+----+
| 10 | 20 |
+----+----+
```

```
Operation 4: arr.push_back(30)
- size=2, capacity=2, need to grow
- new_cap = 4
- allocate 4 slots, copy [10,20], write 30
State: capacity=4, size=3
+----+----+----+---+
| 10 | 20 | 30 | _ |
+----+----+----+---+
```

```
Operation 5: arr.push_back(40)
- size=3, capacity=4, have room
- write 40 at index 3
State: capacity=4, size=4
+----+----+----+----+
| 10 | 20 | 30 | 40 |
+----+----+----+----+
```

```
Operation 6: arr.pop_back()
- size=4 > 0, can pop
- decrement size to 3
- read and return value at index 3
Returns: Some(40)
State: capacity=4, size=3
+----+----+----+----+
| 10 | 20 | 30 | ?? |  (40's bits still there, but inaccessible)
+----+----+----+----+
```

```
Operation 7: arr[1]
- Index operator, bounds check: 1 < 3, OK
- Return reference to element at index 1
Returns: &20
```

```
Operation 8: arr.clear()
- Pop each element: 30, then 20, then 10
- Capacity unchanged
State: capacity=4, size=0
+----+----+----+----+
| ?? | ?? | ?? | ?? |  (memory allocated but no valid elements)
+----+----+----+----+
```

---

## From Concept to Code

### The Data Structure

Before showing code, let's understand what we need to track:

1. **`data: *mut T`** - A raw pointer to the beginning of our heap-allocated buffer. Why a raw pointer? Rust's safe types like `Vec<T>` and `Box<[T]>` would handle memory for us - but we're building this from scratch to learn the mechanics.

2. **`size: usize`** - How many elements are currently stored. This is the logical length.

3. **`capacity: usize`** - How many elements we have room for. This determines when we need to reallocate.

### Rust Implementation

```rust
pub struct DynamicArray<T> {
    data: *mut T,
    size: usize,
    capacity: usize,
}
```

**Line-by-line breakdown**:

- `pub struct DynamicArray<T>` - A generic struct that works with any type `T`. The `<T>` makes this a template that the compiler instantiates for each concrete type used.

- `data: *mut T` - A raw mutable pointer to type `T`. Unlike references (`&T`, `&mut T`), raw pointers:
  - Can be null
  - Don't have lifetime tracking
  - Don't prevent data races
  - Require `unsafe` blocks to dereference

  This is the "escape hatch" from Rust's safety guarantees - necessary for manual memory management.

- `size: usize` and `capacity: usize` - `usize` is an unsigned integer sized to the platform's pointer width (64 bits on 64-bit systems). It's the standard type for sizes and indices in Rust.

### Implementing new() and with_capacity()

**The algorithm in plain English**:
1. For `new()`: Return a struct with null pointer, size 0, capacity 0. No allocation needed.
2. For `with_capacity(n)`: Allocate space for n elements, return struct pointing to it with size 0.

**The code**:

```rust
impl<T> DynamicArray<T> {
    pub fn new() -> Self {
        Self {
            data: core::ptr::null_mut(),
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
            size: 0,
            capacity,
        }
    }
}
```

**Understanding the tricky parts**:

- `core::ptr::null_mut()` - Creates a null pointer. In Rust, we don't have "uninitialized" pointers - we explicitly set it to null to represent "no allocation."

- `core::alloc::Layout::array::<T>(capacity)` - Creates a memory layout describing the size and alignment requirements for an array of `capacity` elements of type `T`. Different types have different alignment requirements (e.g., `i32` must be 4-byte aligned, `i64` must be 8-byte aligned). `Layout` captures both the total size and the alignment.

- `std::alloc::alloc(layout) as *mut T` - Calls the global allocator to get raw bytes, then casts to a typed pointer. This is `unsafe` because:
  1. The allocator might fail (we check for null)
  2. The returned memory is uninitialized
  3. We're responsible for deallocating with the same layout

- `handle_alloc_error(layout)` - If allocation fails (returns null), this function aborts the program. In production code, you might want to return a `Result` instead.

### Implementing push_back

**The algorithm in plain English**:
1. If size equals capacity, we need more room. Double the capacity (or set to 1 if currently 0).
2. Write the new value at index `size` using `ptr::write`.
3. Increment size.

**The code**:

```rust
pub fn push_back(&mut self, value: T) {
    if self.size == self.capacity {
        let new_cap = if self.capacity == 0 { 1 } else { self.capacity * 2 };
        self.reserve(new_cap);
    }
    unsafe {
        core::ptr::write(self.data.add(self.size), value);
    }
    self.size += 1;
}
```

**Understanding the tricky parts**:

- `&mut self` - This method needs exclusive mutable access to the array. Rust's borrow checker ensures no other code can access the array while we're modifying it.

- `self.data.add(self.size)` - Pointer arithmetic. This computes the address of the slot at index `size`. Unlike C's pointer arithmetic (which is in terms of bytes), Rust's `.add(n)` moves the pointer by `n * size_of::<T>()` bytes automatically.

- `core::ptr::write(ptr, value)` - This is crucial. You might think we could just do `*ptr = value`, but that's wrong for uninitialized memory! Why?

  When you do `*ptr = value` in Rust:
  1. Rust reads what's currently at `*ptr`
  2. Rust drops that old value (calls its destructor)
  3. Rust writes the new value

  But our slot is uninitialized - it contains garbage! Trying to "drop" garbage could crash or corrupt memory. `ptr::write` skips steps 1 and 2, directly writing without reading or dropping what was there.

### Implementing pop_back

**The algorithm in plain English**:
1. If empty, return None.
2. Decrement size.
3. Read and return the value that was at the old last position.

**The code**:

```rust
pub fn pop_back(&mut self) -> Option<T> {
    if self.size == 0 {
        None
    } else {
        self.size -= 1;
        unsafe { Some(core::ptr::read(self.data.add(self.size))) }
    }
}
```

**Understanding the tricky parts**:

- `core::ptr::read(ptr)` - The inverse of `ptr::write`. This reads the value out of memory and returns it, but doesn't drop the value in place. After `ptr::read`:
  - The caller owns the returned value
  - The memory still contains the bit pattern (not zeroed)
  - We must not read it again (would create two owners of the same value)

  By decrementing `size` first, we ensure the slot is now "out of bounds" and won't be accessed again.

- `Option<T>` - Rust's way of handling "might not exist." Instead of returning a special sentinel value or panicking, we return `Some(value)` on success or `None` if the array is empty. Callers must handle both cases.

### Implementing reserve

**The algorithm in plain English**:
1. If new capacity is not larger than current, do nothing.
2. Allocate a new buffer with the new capacity.
3. Copy all existing elements to the new buffer.
4. Deallocate the old buffer (if any).
5. Update our pointer and capacity.

**The code**:

```rust
pub fn reserve(&mut self, new_cap: usize) {
    if new_cap <= self.capacity {
        return;
    }
    let new_layout = core::alloc::Layout::array::<T>(new_cap).unwrap();
    let new_data = unsafe { std::alloc::alloc(new_layout) as *mut T };
    if new_data.is_null() {
        std::alloc::handle_alloc_error(new_layout);
    }
    if !self.data.is_null() {
        unsafe {
            core::ptr::copy_nonoverlapping(self.data, new_data, self.size);
            let old_layout = core::alloc::Layout::array::<T>(self.capacity).unwrap();
            std::alloc::dealloc(self.data as *mut u8, old_layout);
        }
    }
    self.data = new_data;
    self.capacity = new_cap;
}
```

**Understanding the tricky parts**:

- `copy_nonoverlapping(src, dst, count)` - Copies `count` elements from `src` to `dst`. The "nonoverlapping" part means source and destination must not overlap (which is always true when copying to a newly allocated buffer). This is a bitwise copy - it doesn't call `Clone::clone()` or any constructors/destructors. This is safe because:
  1. Rust types are always safe to move via bitwise copy (no self-referential pointers)
  2. We immediately deallocate the old buffer, so there's no duplicate ownership

- `dealloc(ptr as *mut u8, layout)` - Frees memory. We must use the same `Layout` that was used to allocate. The cast to `*mut u8` is required by the dealloc API (it works with raw bytes, not typed pointers).

- Why not `realloc`? We could use `std::alloc::realloc`, but it's actually more complex (handles both growth and shrinkage, may or may not move the data). For clarity, we explicitly allocate, copy, and deallocate.

### Implementing Drop

**The algorithm in plain English**:
1. Drop each element (call destructors).
2. Deallocate the buffer.

**The code**:

```rust
impl<T> Drop for DynamicArray<T> {
    fn drop(&mut self) {
        self.clear();  // Pops each element, running destructors
        if !self.data.is_null() {
            let layout = core::alloc::Layout::array::<T>(self.capacity).unwrap();
            unsafe { std::alloc::dealloc(self.data as *mut u8, layout) };
        }
    }
}
```

**Understanding the tricky parts**:

- `Drop` trait - This is Rust's destructor. It's called automatically when the `DynamicArray` goes out of scope. This is RAII (Resource Acquisition Is Initialization) - we tie the memory's lifetime to the struct's lifetime.

- `self.clear()` - This pops each element, which calls `ptr::read` on each. When the returned values go out of scope (immediately, since `pop_back`'s result is unused), their destructors run. This is crucial for types like `String` or `Vec` that own heap memory - without this, we'd leak their allocations.

- Order matters: We must drop elements before deallocating the buffer. If we deallocated first, the destructors would access freed memory.

### Implementing Index Traits

**The code**:

```rust
impl<T> core::ops::Index<usize> for DynamicArray<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        self.at(index).expect("index out of bounds")
    }
}

impl<T> core::ops::IndexMut<usize> for DynamicArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.at_mut(index).expect("index out of bounds")
    }
}
```

**Understanding the tricky parts**:

- `Index` and `IndexMut` traits enable `arr[i]` and `arr[i] = x` syntax.

- `type Output = T` - An associated type declaring what type indexing returns.

- `.expect("message")` - Unwraps an `Option`, panicking with the message if it's `None`. This matches `Vec`'s behavior where out-of-bounds indexing panics.

- The difference from `at()`: `at()` returns `Option<&T>` (safe, caller handles the None case). `Index` returns `&T` directly, panicking on failure. Both are useful - `at()` for when bounds might be exceeded, `[]` for when you know the index is valid.

### Thread Safety Markers

**The code**:

```rust
unsafe impl<T: Send> Send for DynamicArray<T> {}
unsafe impl<T: Sync> Sync for DynamicArray<T> {}
```

**Understanding the tricky parts**:

- `Send` - "This type can be transferred to another thread." Raw pointers aren't `Send` by default because the compiler can't verify they're used safely across threads.

- `Sync` - "References to this type can be shared across threads." Also not automatic with raw pointers.

- `unsafe impl` - We're telling the compiler "trust us, this is safe." It's safe because:
  1. If `T` is `Send`, our `DynamicArray<T>` can be moved to another thread
  2. If `T` is `Sync`, shared references to our array are safe across threads

  We're just wrapping a contiguous allocation - there's nothing thread-unsafe about the container itself.

---

## Complexity Analysis

### Time Complexity

| Operation | Best | Average | Worst | Why |
|-----------|------|---------|-------|-----|
| push_back | O(1) | O(1)* | O(n) | Usually just writes to next slot. Worst case copies all n elements to new buffer. |
| pop_back | O(1) | O(1) | O(1) | Just decrements size and reads one value. |
| at / at_mut | O(1) | O(1) | O(1) | Direct pointer arithmetic: data + index * sizeof(T). |
| Index / IndexMut | O(1) | O(1) | O(1) | Same as at/at_mut plus bounds check. |
| front / back | O(1) | O(1) | O(1) | Just calls at(0) or at(size-1). |
| reserve | O(n) | O(n) | O(n) | Must copy all n existing elements. |
| clear | O(n) | O(n) | O(n) | Must drop each element (destructor could do work). |
| clone | O(n) | O(n) | O(n) | Must clone each of the n elements. |

*The asterisk on push_back's average indicates "amortized O(1)" - see below.

**Understanding the "Why" column**:

- **push_back O(n) worst case**: When we're full (size == capacity), we must allocate a new buffer and copy all elements. With n elements, this is O(n) work.

- **push_back O(1) amortized**: Consider adding n elements starting from empty. We copy at sizes 1, 2, 4, 8, ... up to n. Total copies: 1 + 2 + 4 + ... + n/2 = n - 1. That's roughly n copies over n insertions, so ~1 copy per insertion on average.

- **clear O(n)**: Even if dropping each element is O(1), we have n elements to drop. For simple types like `i32`, the optimizer might skip this entirely. For types with destructors (like `String`), each drop does real work.

### Space Complexity

- **Overall structure**: O(n) where n is the capacity. We always have a buffer of size `capacity`, even if only `size` elements are used.

- **Wasted space**: At worst, just after doubling, we're using only half the capacity. So space usage is at most 2x the logical size.

- **Per operation**:
  - `reserve` uses O(n) temporary space (the new buffer exists alongside the old during copying)
  - All other operations use O(1) extra space

### Amortized Analysis

**What "amortized" means**: Instead of looking at the worst case of a single operation, we look at the average cost across a sequence of operations. Some operations are expensive, but they "pay off" future cheap operations.

**Why push_back is amortized O(1)**:

Think of it like a "savings account" for copying work:
- Each push that doesn't trigger growth "deposits" 2 credits
- When growth happens (say from capacity n to 2n), we need to copy n elements
- But we've done n pushes since the last growth, depositing 2n credits
- The n copies "spend" n credits, leaving n credits for the future

The math works out so every push costs O(1) on average, even accounting for occasional O(n) growth operations.

---

## Common Mistakes & Pitfalls

### Mistake 1: Using Assignment Instead of ptr::write for Uninitialized Memory

```rust
// WRONG: Tries to drop garbage!
unsafe {
    *self.data.add(self.size) = value;
}

// RIGHT: Writes without reading/dropping what's there
unsafe {
    core::ptr::write(self.data.add(self.size), value);
}
```

**Why this matters**: When you use `=` to assign through a pointer, Rust:
1. Reads the old value at that location
2. Calls `drop()` on the old value
3. Writes the new value

If the memory is uninitialized, step 2 tries to drop garbage bits, potentially calling a destructor on random memory. For a `String`, this might try to free a garbage pointer, causing a crash or security vulnerability.

### Mistake 2: Forgetting to Drop Elements Before Deallocating

```rust
// WRONG: Leaks memory for types with destructors!
impl<T> Drop for DynamicArray<T> {
    fn drop(&mut self) {
        if !self.data.is_null() {
            let layout = Layout::array::<T>(self.capacity).unwrap();
            unsafe { std::alloc::dealloc(self.data as *mut u8, layout) };
        }
    }
}

// RIGHT: Drop each element first
impl<T> Drop for DynamicArray<T> {
    fn drop(&mut self) {
        self.clear();  // Drops each element
        if !self.data.is_null() {
            let layout = Layout::array::<T>(self.capacity).unwrap();
            unsafe { std::alloc::dealloc(self.data as *mut u8, layout) };
        }
    }
}
```

**Why this matters**: If your array contains `String`s and you just deallocate the buffer, those `String`s never get dropped. Each `String` owns a heap allocation - now leaked forever. You've freed the pointers without freeing what they pointed to.

### Mistake 3: Mismatched Layouts for Alloc/Dealloc

```rust
// WRONG: Using wrong layout for dealloc!
let layout = Layout::array::<T>(self.size).unwrap();  // BUG: should be capacity!
unsafe { std::alloc::dealloc(self.data as *mut u8, layout) };

// RIGHT: Use the same layout we allocated with
let layout = Layout::array::<T>(self.capacity).unwrap();
unsafe { std::alloc::dealloc(self.data as *mut u8, layout) };
```

**Why this matters**: The allocator tracks memory by size. If you deallocate with a different size than you allocated, the allocator gets confused. This is undefined behavior - it might corrupt the heap, crash later, or appear to work while subtly corrupting memory.

### Mistake 4: Reading After pop_back Without Decrementing First

```rust
// WRONG: Reading what we're about to make inaccessible
pub fn pop_back(&mut self) -> Option<T> {
    if self.size == 0 { return None; }
    let value = unsafe { core::ptr::read(self.data.add(self.size - 1)) };
    self.size -= 1;  // What if this line was forgotten?
    Some(value)
}

// RIGHT (our implementation): Decrement first, then read
pub fn pop_back(&mut self) -> Option<T> {
    if self.size == 0 {
        None
    } else {
        self.size -= 1;
        unsafe { Some(core::ptr::read(self.data.add(self.size))) }
    }
}
```

**Why this matters**: The first version would work, but decrementing before reading has a subtle advantage: `self.size` already points to the element we want to read. Also, if we had some concurrent access (we don't, but hypothetically), decrementing first marks the slot as "taken" before we read it.

### Mistake 5: Not Handling Zero Capacity in with_capacity

```rust
// WRONG: Allocating 0 bytes, getting weird behavior
pub fn with_capacity(capacity: usize) -> Self {
    let layout = Layout::array::<T>(capacity).unwrap();  // Layout for 0 elements
    let data = unsafe { std::alloc::alloc(layout) as *mut T };
    // ...
}

// RIGHT: Special-case zero
pub fn with_capacity(capacity: usize) -> Self {
    if capacity == 0 {
        return Self::new();  // No allocation for zero capacity
    }
    // ... normal allocation
}
```

**Why this matters**: Allocating 0 bytes is weird - some allocators return null, others return a unique non-null pointer. By special-casing it, we have consistent behavior: null pointer means no allocation.

---

## Practice Problems

To solidify your understanding, try implementing:

1. **`insert(index, value)`**: Insert an element at any position, shifting subsequent elements right. Think about bounds checking and what happens at capacity.

2. **`remove(index) -> Option<T>`**: Remove an element at any position, shifting subsequent elements left. Handle edge cases.

3. **`shrink_to_fit()`**: Reallocate to make capacity equal to size, freeing wasted space. Useful after removing many elements.

4. **`extend<I: IntoIterator<Item = T>>(&mut self, iter: I)`**: Append all elements from an iterator. Can you do it efficiently with `reserve`?

5. **`into_iter(self)`**: Implement `IntoIterator` for `DynamicArray<T>` (not just `&DynamicArray<T>`). This consumes the array and yields owned elements.

---

## Summary

### Key Takeaways

- A dynamic array is a contiguous, growable block of memory that provides O(1) random access and amortized O(1) append.

- **Capacity vs Size**: Capacity is how much room we have; size is how much we're using. Reallocation happens when size reaches capacity.

- **Doubling strategy**: Growing by 2x each time gives amortized O(1) insertion because the total copy cost is proportional to the number of insertions.

- **Raw pointers in Rust** (`*mut T`) require `unsafe` blocks and manual memory management. Use `ptr::write` for uninitialized memory, `ptr::read` to move values out, and `ptr::copy_nonoverlapping` for bulk moves.

- **Drop is crucial**: For types with destructors, you must drop each element before deallocating the buffer, or you'll leak memory.

- **`Option<T>`** is Rust's idiomatic way to handle "might not exist" - safer than panicking or returning sentinels.

### Quick Reference

```
DynamicArray<T> - A growable array with contiguous storage
|-- new()           : O(1)   - Create empty with no allocation
|-- with_capacity(n): O(1)   - Create empty with n slots pre-allocated
|-- push_back(v)    : O(1)*  - Add to end (* = amortized)
|-- pop_back()      : O(1)   - Remove from end, returns Option<T>
|-- at(i)/at_mut(i) : O(1)   - Bounds-checked access, returns Option
|-- [i]/[i]=v       : O(1)   - Index access, panics if out of bounds
|-- reserve(n)      : O(n)   - Ensure capacity >= n
|-- clear()         : O(n)   - Remove all elements, keep capacity
|-- size()          : O(1)   - Number of elements
|-- capacity()      : O(1)   - Number of slots
|-- is_empty()      : O(1)   - Check if size == 0

Best for: Random access, iteration, adding/removing from end
Avoid when: Frequent mid-array insertions/deletions (use linked list)
Memory: Uses up to 2x the space of contained elements
```
