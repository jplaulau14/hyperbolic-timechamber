# Go Data Structures

## Static Array

`StaticArray[T]` — fixed-size generic array. No external dependencies.

### Run tests

```bash
cd data-structures/go
go test -v ./...
```

### API

```go
import "staticarray"

arr := staticarray.New[int](5)

arr.Fill(0)               // fill all elements
arr.Set(0, 10)            // unchecked write
arr.Get(0)                // unchecked read
arr.SetAt(1, 20)          // checked write (returns error)
arr.At(1)                 // checked read (returns value, error)
arr.Front()               // first element
arr.Back()                // last element
arr.Data()                // underlying slice
arr.Size()                // returns the fixed size
arr.IsEmpty()             // true if size is 0

for _, v := range arr.Data() { /* works */ }
```

### Zero-size arrays

`New[T](0)` is valid. `Data()` returns `nil`, `At()` always returns `ErrOutOfRange`, iteration over `Data()` is a no-op.
