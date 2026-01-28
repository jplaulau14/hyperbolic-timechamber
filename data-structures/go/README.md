# Go Data Structures

Requires Go 1.21+.

## Run Tests

```bash
go test -v ./...
```

## Static Array

`StaticArray[T]` in `src/staticarray/static_array.go`

Fixed-size generic array.

```go
import "github.com/hyperbolic-timechamber/data-structures-go/src/staticarray"

arr := staticarray.New[int](5)
arr.Fill(0)
arr.Set(0, 10)
arr.At(1)             // checked access (returns value, error)
arr.Get(1)            // unchecked access

for _, v := range arr.Data() { /* iteration works */ }
```

Also supports: `SetAt()`, `Front()`, `Back()`, `Data()`, `Size()`, `IsEmpty()`.

Zero-size (`New[T](0)`) is handled — `Data()` returns `nil`, `At()` always returns `ErrOutOfRange`.

## Dynamic Array

`DynamicArray[T]` in `src/dynamicarray/dynamic_array.go`

Resizable generic array.

```go
import "github.com/hyperbolic-timechamber/data-structures-go/src/dynamicarray"

arr := dynamicarray.New[int]()
arr.PushBack(10)
arr.PushBack(20)
arr.At(0)             // checked access (returns value, error)
arr.Reserve(100)      // pre-allocate capacity
arr.PopBack()
arr.Clear()

for _, v := range arr.Data() { /* iteration works */ }
```

Also supports: `SetAt()`, `Get()`, `Set()`, `Front()`, `Back()`, `Data()`, `Size()`, `Capacity()`, `IsEmpty()`, `Clone()`.

Growth strategy doubles capacity when exceeded.
