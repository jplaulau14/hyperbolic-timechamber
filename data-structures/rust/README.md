# Rust Data Structures

## Static Array

`StaticArray<T, N>` — fixed-size array using const generics. No external dependencies.

### Run tests

```bash
cd data-structures/rust
cargo test
```

### API

```rust
use data_structures::StaticArray;

let mut arr: StaticArray<i32, 5> = StaticArray::new();

arr.fill(0);              // fill all elements
arr[0] = 10;              // unchecked access (panics if out of bounds)
arr.at(1);                // checked access (returns Option<&T>)
arr.at_mut(1);            // checked mutable access (returns Option<&mut T>)
arr.front();              // first element (Option)
arr.back();               // last element (Option)
arr.data();               // &[T] slice
arr.size();               // returns N
arr.is_empty();           // true if N == 0

for v in &arr { /* works */ }
```

### Zero-size arrays

`StaticArray<T, 0>` works out of the box. `data()` returns an empty slice, `at()`/`front()`/`back()` return `None`, iteration is a no-op.
