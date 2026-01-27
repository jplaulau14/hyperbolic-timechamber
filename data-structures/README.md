# Data Structures

Implementations from scratch, no standard library containers. Personal reference.

## C++

Requires CMake and a C++17 compiler.

### Build & test

```bash
cd data-structures/cpp
cmake -B build
cmake --build build
./build/static_array_test
```

### Implementations

**Static Array** — `StaticArray<T, N>` in `cpp/src/static_array.hpp`

Stack-allocated, fixed-size array. Header-only, just include and use:

```cpp
#include "static_array.hpp"

StaticArray<int, 5> arr;
arr.fill(0);
arr[0] = 10;
arr.at(1) = 20;       // throws std::out_of_range if out of bounds

for (auto& v : arr) { /* works */ }
```

Also supports: `front()`, `back()`, `data()`, `size()`, `empty()`, `begin()`/`end()`.

Zero-size (`StaticArray<T, 0>`) is handled — `data()` returns `nullptr`, `at()` always throws.

## Python

Requires Python 3. No external dependencies.

### Run tests

```bash
cd data-structures/python
python tests/test_static_array.py
```

### Implementations

**Static Array** — `StaticArray` in `python/src/static_array.py`

Fixed-size array. Import and use:

```python
from static_array import StaticArray

arr = StaticArray(5)
arr.fill(0)
arr[0] = 10
arr.set_at(1, 20)    # raises IndexError if out of bounds

for v in arr:         # iteration works
    pass
```

Also supports: `front()`, `back()`, `data()`, `size()`, `empty()`, `at()`, `len()`.

Zero-size (`StaticArray(0)`) is handled — `data()` returns `None`, `at()` always raises.

## Rust

Requires Rust (cargo).

### Build & test

```bash
cd data-structures/rust
cargo test
```

### Implementations

**Static Array** — `StaticArray<T, N>` in `rust/src/lib.rs`

Fixed-size array using const generics:

```rust
use data_structures::StaticArray;

let mut arr: StaticArray<i32, 5> = StaticArray::new();
arr.fill(0);
arr[0] = 10;
arr.at(1);                // checked access (returns Option<&T>)

for v in &arr { /* works */ }
```

Also supports: `at_mut()`, `front()`, `back()`, `data()`, `size()`, `is_empty()`.

Zero-size (`StaticArray<T, 0>`) works out of the box — `data()` returns an empty slice, `at()`/`front()`/`back()` return `None`.

## Go

Requires Go 1.21+.

### Build & test

```bash
cd data-structures/go
go test -v ./...
```

### Implementations

**Static Array** — `StaticArray[T]` in `go/static_array.go`

Fixed-size generic array:

```go
arr := staticarray.New[int](5)
arr.Fill(0)
arr.Set(0, 10)
arr.At(1)                 // checked access (returns value, error)

for _, v := range arr.Data() { /* works */ }
```

Also supports: `SetAt()`, `Get()`, `Front()`, `Back()`, `Data()`, `Size()`, `IsEmpty()`.

Zero-size (`New[T](0)`) is handled — `Data()` returns `nil`, `At()` always returns `ErrOutOfRange`.
