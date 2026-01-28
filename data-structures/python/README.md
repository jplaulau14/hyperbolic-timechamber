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
