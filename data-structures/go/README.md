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

## Linked List

`LinkedList[T]` in `src/linkedlist/linked_list.go`

Singly linked list with head/tail pointers.

```go
import "github.com/hyperbolic-timechamber/data-structures-go/src/linkedlist"

list := linkedlist.New[int]()
list.PushBack(10)
list.PushFront(5)
list.Front()          // checked access (returns value, error)
list.Back()
list.At(1)            // checked access (returns value, error)
list.PopFront()
list.PopBack()

for _, v := range list.Values() { /* iteration works */ }
```

Also supports: `InsertAt()`, `RemoveAt()`, `Size()`, `IsEmpty()`, `Clear()`, `Clone()`.

Exports `ErrOutOfRange` and `ErrEmptyList` sentinel errors.

## Stack

`Stack[T]` in `src/stack/stack.go`

LIFO container built on slice.

```go
import "github.com/hyperbolic-timechamber/data-structures-go/src/stack"

s := stack.New[int]()
s.Push(10)
s.Push(20)
s.Top()       // (20, nil)
s.Pop()       // (20, nil)
s.Size()      // 1
```

Methods: `Push()`, `Pop()`, `Top()`, `Size()`, `IsEmpty()`, `Clear()`, `Clone()`.

Exports `ErrEmptyStack` sentinel error.

## Queue

`Queue[T]` in `src/queue/queue.go`

FIFO container using circular buffer.

```go
import "github.com/hyperbolic-timechamber/data-structures-go/src/queue"

q := queue.New[int]()
q.Enqueue(10)
q.Enqueue(20)
q.Front()     // (10, nil)
q.Dequeue()   // (10, nil)
q.Size()      // 1
```

Methods: `Enqueue()`, `Dequeue()`, `Front()`, `Back()`, `Size()`, `IsEmpty()`, `Clear()`, `Clone()`.

Exports `ErrEmptyQueue` sentinel error.

## Hash Map

`HashMap[K comparable, V any]` in `src/hashmap/hash_map.go`

Key-value store with O(1) average operations using separate chaining.

```go
import "github.com/hyperbolic-timechamber/data-structures-go/src/hashmap"

m := hashmap.New[string, int]()
m.Put("one", 1)
m.Put("two", 2)
v, ok := m.Get("one")    // (1, true)
m.Contains("two")        // true
m.Remove("one")
```

Methods: `Put()`, `Get()`, `Remove()`, `Contains()`, `Size()`, `IsEmpty()`, `Clear()`, `Keys()`, `Values()`, `Clone()`.

Rehashes automatically when load factor exceeds 0.75.

## Binary Heap

`BinaryHeap[T cmp.Ordered]` in `src/binaryheap/binary_heap.go`

Min-heap with O(log n) push/pop.

```go
import "github.com/hyperbolic-timechamber/data-structures-go/src/binaryheap"

h := binaryheap.New[int]()
h.Push(30)
h.Push(10)
h.Push(20)
h.Peek()     // (10, nil)
h.Pop()      // (10, nil)
h.Pop()      // (20, nil)

// Build from slice in O(n)
h2 := binaryheap.FromSlice([]int{5, 2, 8, 1})
```

Methods: `Push()`, `Pop()`, `Peek()`, `Size()`, `IsEmpty()`, `Clear()`, `Clone()`, `FromSlice()`.

Exports `ErrEmptyHeap` sentinel error.
