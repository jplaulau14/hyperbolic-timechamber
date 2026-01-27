# C++ Data Structures

## Static Array

Header-only `StaticArray<T, N>` — fixed-size, stack-allocated array. No STL containers, just `<stdexcept>` and `<cstddef>`.

### Build & run tests

```bash
cd data-structures/cpp
cmake -B build
cmake --build build
./build/static_array_test
```

### API

```cpp
#include "static_array.hpp"

StaticArray<int, 5> arr;

arr.fill(0);          // fill all elements
arr[0] = 10;          // unchecked access
arr.at(1) = 20;       // checked access (throws std::out_of_range)
arr.front();          // first element
arr.back();           // last element
arr.data();           // raw T* pointer
arr.size();           // returns N (compile-time constant)
arr.empty();          // true if N == 0

for (auto& v : arr) { /* range-based for works */ }
```

### Zero-size arrays

`StaticArray<T, 0>` is a valid partial specialization. `data()` returns `nullptr`, `at()` always throws, iterators are `nullptr` so range-for is a no-op.
