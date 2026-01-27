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
