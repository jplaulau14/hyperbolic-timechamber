package staticarray

import (
	"errors"
	"testing"
)

func TestSizeAndEmpty(t *testing.T) {
	arr := New[int](5)
	if arr.Size() != 5 {
		t.Fatalf("expected size 5, got %d", arr.Size())
	}
	if arr.IsEmpty() {
		t.Fatal("expected non-empty")
	}

	empty := New[int](0)
	if empty.Size() != 0 {
		t.Fatalf("expected size 0, got %d", empty.Size())
	}
	if !empty.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestFillAndAccess(t *testing.T) {
	arr := New[int](4)
	arr.Fill(42)
	for i := 0; i < arr.Size(); i++ {
		if arr.Get(i) != 42 {
			t.Fatalf("index %d: expected 42, got %d", i, arr.Get(i))
		}
	}
}

func TestGetSet(t *testing.T) {
	arr := New[int](3)
	arr.Set(0, 10)
	arr.Set(1, 20)
	arr.Set(2, 30)
	if arr.Get(0) != 10 || arr.Get(1) != 20 || arr.Get(2) != 30 {
		t.Fatal("get/set mismatch")
	}
}

func TestAtValidIndex(t *testing.T) {
	arr := New[int](3)
	arr.SetAt(0, 100)
	arr.SetAt(1, 200)
	arr.SetAt(2, 300)

	for i, expected := range []int{100, 200, 300} {
		v, err := arr.At(i)
		if err != nil {
			t.Fatalf("unexpected error at index %d", i)
		}
		if v != expected {
			t.Fatalf("index %d: expected %d, got %d", i, expected, v)
		}
	}
}

func TestAtOutOfRange(t *testing.T) {
	arr := New[int](3)
	_, err := arr.At(3)
	if !errors.Is(err, ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for index 3")
	}
	_, err = arr.At(100)
	if !errors.Is(err, ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for index 100")
	}
}

func TestAtOnZeroSize(t *testing.T) {
	arr := New[int](0)
	_, err := arr.At(0)
	if !errors.Is(err, ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange on zero-size array")
	}
}

func TestSetAtOutOfRange(t *testing.T) {
	arr := New[int](3)
	if err := arr.SetAt(3, 0); !errors.Is(err, ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for SetAt index 3")
	}
	if err := arr.SetAt(-1, 0); !errors.Is(err, ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for SetAt index -1")
	}
}

func TestFrontAndBack(t *testing.T) {
	arr := New[int](4)
	arr.Fill(0)
	arr.Set(0, 1)
	arr.Set(3, 99)
	if arr.Front() != 1 {
		t.Fatalf("expected front 1, got %d", arr.Front())
	}
	if arr.Back() != 99 {
		t.Fatalf("expected back 99, got %d", arr.Back())
	}
}

func TestDataSlice(t *testing.T) {
	arr := New[int](3)
	arr.Fill(7)
	d := arr.Data()
	if d[0] != 7 || d[1] != 7 || d[2] != 7 {
		t.Fatal("data slice values mismatch")
	}

	d[1] = 42
	if arr.Get(1) != 42 {
		t.Fatal("data slice should share underlying memory")
	}
}

func TestDataZeroSize(t *testing.T) {
	arr := New[int](0)
	if arr.Data() != nil {
		t.Fatal("expected nil data for zero-size array")
	}
}

func TestIteration(t *testing.T) {
	arr := New[int](5)
	arr.Fill(3)
	sum := 0
	for _, v := range arr.Data() {
		sum += v
	}
	if sum != 15 {
		t.Fatalf("expected sum 15, got %d", sum)
	}
}

func TestNonTrivialType(t *testing.T) {
	arr := New[string](3)
	arr.Set(0, "hello")
	arr.Set(1, "world")
	arr.Set(2, "!")
	if arr.Get(0) != "hello" {
		t.Fatal("expected hello")
	}
	v, _ := arr.At(1)
	if v != "world" {
		t.Fatal("expected world")
	}
	if arr.Back() != "!" {
		t.Fatal("expected !")
	}
}

func TestFillOverwrites(t *testing.T) {
	arr := New[int](3)
	arr.Fill(1)
	arr.Fill(2)
	for i := 0; i < arr.Size(); i++ {
		if arr.Get(i) != 2 {
			t.Fatalf("index %d: expected 2, got %d", i, arr.Get(i))
		}
	}
}

func TestZeroSizeIteration(t *testing.T) {
	arr := New[int](0)
	count := 0
	for range arr.Data() {
		count++
	}
	if count != 0 {
		t.Fatal("expected zero iterations")
	}
}

func TestSingleElement(t *testing.T) {
	arr := New[int](1)
	arr.Set(0, 42)
	if arr.Front() != 42 {
		t.Fatalf("expected front 42, got %d", arr.Front())
	}
	if arr.Back() != 42 {
		t.Fatalf("expected back 42, got %d", arr.Back())
	}
	if arr.Size() != 1 {
		t.Fatalf("expected size 1, got %d", arr.Size())
	}
}

func TestNegativeIndexAt(t *testing.T) {
	arr := New[int](3)
	_, err := arr.At(-1)
	if !errors.Is(err, ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for negative index")
	}
}
