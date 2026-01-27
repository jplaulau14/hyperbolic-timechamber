# Python Data Structures

## Static Array

`StaticArray` — fixed-size array. No external dependencies.

### Run tests

```bash
cd data-structures/python
python tests/test_static_array.py
```

### API

```python
from static_array import StaticArray

arr = StaticArray(5)

arr.fill(0)           # fill all elements
arr[0] = 10           # unchecked access
arr.at(1)             # checked access (raises IndexError)
arr.set_at(1, 20)     # checked write (raises IndexError)
arr.front()           # first element
arr.back()            # last element
arr.data()            # underlying list reference
arr.size()            # returns the fixed size
arr.empty()           # True if size is 0
len(arr)              # also returns the fixed size

for v in arr:         # iteration works
    pass
```

### Zero-size arrays

`StaticArray(0)` is valid. `data()` returns `None`, `at()` always raises, iteration is a no-op.
