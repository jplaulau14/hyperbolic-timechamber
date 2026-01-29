package dynamicarray_test

import (
	"errors"
	"testing"

	"github.com/hyperbolic-timechamber/data-structures-go/src/dynamicarray"
)

func TestDefaultConstruction(t *testing.T) {
	arr := dynamicarray.New[int]()
	if arr.Size() != 0 {
		t.Fatalf("expected size 0, got %d", arr.Size())
	}
	if arr.Capacity() != 0 {
		t.Fatalf("expected capacity 0, got %d", arr.Capacity())
	}
	if !arr.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestSizedConstruction(t *testing.T) {
	arr := dynamicarray.NewWithSize[int](5)
	if arr.Size() != 5 {
		t.Fatalf("expected size 5, got %d", arr.Size())
	}
	if arr.Capacity() != 5 {
		t.Fatalf("expected capacity 5, got %d", arr.Capacity())
	}
	if arr.IsEmpty() {
		t.Fatal("expected non-empty")
	}
	for i := 0; i < arr.Size(); i++ {
		if arr.Get(i) != 0 {
			t.Fatalf("index %d: expected 0, got %d", i, arr.Get(i))
		}
	}
}

func TestPushBackSingle(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(42)
	if arr.Size() != 1 {
		t.Fatalf("expected size 1, got %d", arr.Size())
	}
	if arr.Get(0) != 42 {
		t.Fatalf("expected 42, got %d", arr.Get(0))
	}
}

func TestPushBackMultiple(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(1)
	arr.PushBack(2)
	arr.PushBack(3)
	if arr.Size() != 3 {
		t.Fatalf("expected size 3, got %d", arr.Size())
	}
	if arr.Get(0) != 1 || arr.Get(1) != 2 || arr.Get(2) != 3 {
		t.Fatal("values mismatch")
	}
}

func TestPushBackTriggersGrowth(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(1)
	if arr.Capacity() != 1 {
		t.Fatalf("expected capacity 1, got %d", arr.Capacity())
	}
	arr.PushBack(2)
	if arr.Capacity() != 2 {
		t.Fatalf("expected capacity 2, got %d", arr.Capacity())
	}
	arr.PushBack(3)
	if arr.Capacity() != 4 {
		t.Fatalf("expected capacity 4, got %d", arr.Capacity())
	}
	arr.PushBack(4)
	arr.PushBack(5)
	if arr.Capacity() != 8 {
		t.Fatalf("expected capacity 8, got %d", arr.Capacity())
	}
	if arr.Size() != 5 {
		t.Fatalf("expected size 5, got %d", arr.Size())
	}
}

func TestPopBack(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(10)
	arr.PushBack(20)
	arr.PushBack(30)
	if err := arr.PopBack(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if arr.Size() != 2 {
		t.Fatalf("expected size 2, got %d", arr.Size())
	}
	back, err := arr.Back()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if back != 20 {
		t.Fatalf("expected back 20, got %d", back)
	}
	if err := arr.PopBack(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if arr.Size() != 1 {
		t.Fatalf("expected size 1, got %d", arr.Size())
	}
	back, err = arr.Back()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if back != 10 {
		t.Fatalf("expected back 10, got %d", back)
	}
}

func TestAtValidIndex(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(100)
	arr.PushBack(200)
	arr.PushBack(300)
	for i, expected := range []int{100, 200, 300} {
		v, err := arr.At(i)
		if err != nil {
			t.Fatalf("unexpected error at index %d", i)
		}
		if v != expected {
			t.Fatalf("index %d: expected %d, got %d", i, expected, v)
		}
	}
	arr.SetAt(1, 999)
	v, _ := arr.At(1)
	if v != 999 {
		t.Fatalf("expected 999, got %d", v)
	}
}

func TestAtOutOfRange(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(1)
	_, err := arr.At(1)
	if !errors.Is(err, dynamicarray.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for index 1")
	}
	_, err = arr.At(100)
	if !errors.Is(err, dynamicarray.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for index 100")
	}
	empty := dynamicarray.New[int]()
	_, err = empty.At(0)
	if !errors.Is(err, dynamicarray.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange on empty array")
	}
}

func TestGetSet(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(10)
	arr.PushBack(20)
	if arr.Get(0) != 10 || arr.Get(1) != 20 {
		t.Fatal("get mismatch")
	}
	arr.Set(0, 99)
	if arr.Get(0) != 99 {
		t.Fatalf("expected 99, got %d", arr.Get(0))
	}
}

func TestFrontAndBack(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(1)
	arr.PushBack(2)
	arr.PushBack(3)
	front, err := arr.Front()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if front != 1 {
		t.Fatalf("expected front 1, got %d", front)
	}
	back, err := arr.Back()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if back != 3 {
		t.Fatalf("expected back 3, got %d", back)
	}
}

func TestReserveIncreasesCapacity(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.Reserve(10)
	if arr.Capacity() < 10 {
		t.Fatalf("expected capacity >= 10, got %d", arr.Capacity())
	}
	if arr.Size() != 0 {
		t.Fatalf("expected size 0, got %d", arr.Size())
	}
}

func TestReservePreservesElements(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(1)
	arr.PushBack(2)
	arr.PushBack(3)
	arr.Reserve(100)
	if arr.Capacity() < 100 {
		t.Fatalf("expected capacity >= 100, got %d", arr.Capacity())
	}
	if arr.Size() != 3 {
		t.Fatalf("expected size 3, got %d", arr.Size())
	}
	if arr.Get(0) != 1 || arr.Get(1) != 2 || arr.Get(2) != 3 {
		t.Fatal("values not preserved")
	}
}

func TestReserveSmallerIsNoop(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.Reserve(10)
	cap := arr.Capacity()
	arr.Reserve(5)
	if arr.Capacity() != cap {
		t.Fatal("reserve smaller should be no-op")
	}
}

func TestClearResetsSizeNotCapacity(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(1)
	arr.PushBack(2)
	arr.PushBack(3)
	cap := arr.Capacity()
	arr.Clear()
	if arr.Size() != 0 {
		t.Fatalf("expected size 0, got %d", arr.Size())
	}
	if !arr.IsEmpty() {
		t.Fatal("expected empty")
	}
	if arr.Capacity() != cap {
		t.Fatal("capacity should be unchanged")
	}
}

func TestClone(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(1)
	arr.PushBack(2)
	arr.PushBack(3)

	clone := arr.Clone()
	if clone.Size() != 3 {
		t.Fatalf("expected size 3, got %d", clone.Size())
	}
	if clone.Get(0) != 1 || clone.Get(1) != 2 || clone.Get(2) != 3 {
		t.Fatal("clone values mismatch")
	}

	arr.Set(0, 999)
	if clone.Get(0) != 1 {
		t.Fatal("clone should be independent")
	}
}

func TestIteration(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(1)
	arr.PushBack(2)
	arr.PushBack(3)
	arr.PushBack(4)

	sum := 0
	for _, v := range arr.Data() {
		sum += v
	}
	if sum != 10 {
		t.Fatalf("expected sum 10, got %d", sum)
	}
}

func TestNonTrivialType(t *testing.T) {
	arr := dynamicarray.New[string]()
	arr.PushBack("hello")
	arr.PushBack("world")
	if arr.Size() != 2 {
		t.Fatalf("expected size 2, got %d", arr.Size())
	}
	if arr.Get(0) != "hello" || arr.Get(1) != "world" {
		t.Fatal("values mismatch")
	}
	front, err := arr.Front()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if front != "hello" {
		t.Fatal("expected front hello")
	}
	back, err := arr.Back()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if back != "world" {
		t.Fatal("expected back world")
	}
}

func TestDataSlice(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(1)
	arr.PushBack(2)
	d := arr.Data()
	if d[0] != 1 || d[1] != 2 {
		t.Fatal("data slice mismatch")
	}
	d[0] = 100
	if arr.Get(0) != 100 {
		t.Fatal("data slice should share memory")
	}
}

func TestEmptyDataSlice(t *testing.T) {
	arr := dynamicarray.New[int]()
	if arr.Data() != nil {
		t.Fatal("expected nil data for empty array")
	}
}

func TestNegativeIndexAt(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(1)
	_, err := arr.At(-1)
	if !errors.Is(err, dynamicarray.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for negative index")
	}
}

func TestSetAtOutOfRange(t *testing.T) {
	arr := dynamicarray.New[int]()
	arr.PushBack(1)
	if err := arr.SetAt(1, 0); !errors.Is(err, dynamicarray.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for SetAt index 1")
	}
	if err := arr.SetAt(-1, 0); !errors.Is(err, dynamicarray.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for SetAt index -1")
	}
}

func TestFrontEmpty(t *testing.T) {
	arr := dynamicarray.New[int]()
	_, err := arr.Front()
	if !errors.Is(err, dynamicarray.ErrEmpty) {
		t.Fatal("expected ErrEmpty for Front on empty array")
	}
}

func TestBackEmpty(t *testing.T) {
	arr := dynamicarray.New[int]()
	_, err := arr.Back()
	if !errors.Is(err, dynamicarray.ErrEmpty) {
		t.Fatal("expected ErrEmpty for Back on empty array")
	}
}

func TestPopBackEmpty(t *testing.T) {
	arr := dynamicarray.New[int]()
	err := arr.PopBack()
	if !errors.Is(err, dynamicarray.ErrEmpty) {
		t.Fatal("expected ErrEmpty for PopBack on empty array")
	}
}

func TestClearZerosElements(t *testing.T) {
	arr := dynamicarray.New[*int]()
	a, b, c := 1, 2, 3
	arr.PushBack(&a)
	arr.PushBack(&b)
	arr.PushBack(&c)
	arr.Clear()
	if arr.Size() != 0 {
		t.Fatalf("expected size 0, got %d", arr.Size())
	}
}
