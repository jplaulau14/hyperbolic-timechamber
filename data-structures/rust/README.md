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
