# C++ Data Structures

Requires CMake and a C++17 compiler.

## Build & Test

```bash
cmake -B build
cmake --build build
./build/static_array_test
./build/dynamic_array_test
```

## Static Array

`StaticArray<T, N>` in `src/static_array.hpp`

Stack-allocated, fixed-size array. Header-only.

```cpp
#include "static_array.hpp"

StaticArray<int, 5> arr;
arr.fill(0);
arr[0] = 10;
arr.at(1) = 20;       // throws std::out_of_range if out of bounds

for (auto& v : arr) { /* iteration works */ }
```

Also supports: `front()`, `back()`, `data()`, `size()`, `empty()`, `begin()`/`end()`.

Zero-size (`StaticArray<T, 0>`) is handled — `data()` returns `nullptr`, `at()` always throws.

## Dynamic Array

`DynamicArray<T>` in `src/dynamic_array.hpp`

Heap-allocated, resizable array. Header-only.

```cpp
#include "dynamic_array.hpp"

DynamicArray<int> arr;
arr.push_back(10);
arr.push_back(20);
arr.at(0);            // throws std::out_of_range if out of bounds
arr.reserve(100);     // pre-allocate capacity
arr.pop_back();
arr.clear();

for (auto& v : arr) { /* iteration works */ }
```

Also supports: `front()`, `back()`, `data()`, `size()`, `capacity()`, `empty()`, `begin()`/`end()`.

Copy/move constructors and assignment operators are implemented. Growth strategy doubles capacity when exceeded.
