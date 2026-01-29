package stack_test

import (
	"errors"
	"testing"

	"github.com/hyperbolic-timechamber/data-structures-go/src/stack"
)

func TestNewStackIsEmpty(t *testing.T) {
	s := stack.New[int]()
	if s.Size() != 0 {
		t.Fatalf("expected size 0, got %d", s.Size())
	}
	if !s.IsEmpty() {
		t.Fatal("expected empty stack")
	}
}

func TestTopOnEmptyStack(t *testing.T) {
	s := stack.New[int]()
	_, err := s.Top()
	if !errors.Is(err, stack.ErrEmptyStack) {
		t.Fatal("expected ErrEmptyStack")
	}
}

func TestPopOnEmptyStack(t *testing.T) {
	s := stack.New[int]()
	_, err := s.Pop()
	if !errors.Is(err, stack.ErrEmptyStack) {
		t.Fatal("expected ErrEmptyStack")
	}
}

func TestPushSingleElement(t *testing.T) {
	s := stack.New[int]()
	s.Push(42)
	if s.Size() != 1 {
		t.Fatalf("expected size 1, got %d", s.Size())
	}
}

func TestPushMultipleElements(t *testing.T) {
	s := stack.New[int]()
	s.Push(1)
	s.Push(2)
	s.Push(3)
	if s.Size() != 3 {
		t.Fatalf("expected size 3, got %d", s.Size())
	}
}

func TestPushManyElements(t *testing.T) {
	s := stack.New[int]()
	for i := 0; i < 1000; i++ {
		s.Push(i)
	}
	if s.Size() != 1000 {
		t.Fatalf("expected size 1000, got %d", s.Size())
	}
}

func TestPopReturnsTopElement(t *testing.T) {
	s := stack.New[int]()
	s.Push(10)
	s.Push(20)
	s.Push(30)
	v, err := s.Pop()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v != 30 {
		t.Fatalf("expected 30, got %d", v)
	}
}

func TestPopDecrementsSize(t *testing.T) {
	s := stack.New[int]()
	s.Push(1)
	s.Push(2)
	s.Pop()
	if s.Size() != 1 {
		t.Fatalf("expected size 1, got %d", s.Size())
	}
}

func TestPopAllElements(t *testing.T) {
	s := stack.New[int]()
	s.Push(1)
	s.Push(2)
	s.Push(3)
	s.Pop()
	s.Pop()
	s.Pop()
	if !s.IsEmpty() {
		t.Fatal("expected empty stack")
	}
}

func TestLIFOOrder(t *testing.T) {
	s := stack.New[int]()
	s.Push(1)
	s.Push(2)
	s.Push(3)

	v1, _ := s.Pop()
	v2, _ := s.Pop()
	v3, _ := s.Pop()

	if v1 != 3 || v2 != 2 || v3 != 1 {
		t.Fatalf("expected 3,2,1 got %d,%d,%d", v1, v2, v3)
	}
}

func TestTopReturnsWithoutRemoving(t *testing.T) {
	s := stack.New[int]()
	s.Push(42)
	v, err := s.Top()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v != 42 {
		t.Fatalf("expected 42, got %d", v)
	}
	if s.Size() != 1 {
		t.Fatalf("expected size 1, got %d", s.Size())
	}
}

func TestTopAfterPush(t *testing.T) {
	s := stack.New[int]()
	s.Push(10)
	s.Push(20)
	v, _ := s.Top()
	if v != 20 {
		t.Fatalf("expected 20, got %d", v)
	}
}

func TestTopMultipleCallsReturnsSameValue(t *testing.T) {
	s := stack.New[int]()
	s.Push(99)
	for i := 0; i < 5; i++ {
		v, _ := s.Top()
		if v != 99 {
			t.Fatalf("expected 99, got %d", v)
		}
	}
	if s.Size() != 1 {
		t.Fatalf("expected size 1, got %d", s.Size())
	}
}

func TestClearMakesStackEmpty(t *testing.T) {
	s := stack.New[int]()
	s.Push(1)
	s.Push(2)
	s.Push(3)
	s.Clear()
	if !s.IsEmpty() {
		t.Fatal("expected empty stack")
	}
	if s.Size() != 0 {
		t.Fatalf("expected size 0, got %d", s.Size())
	}
}

func TestClearOnEmptyStack(t *testing.T) {
	s := stack.New[int]()
	s.Clear()
	if !s.IsEmpty() {
		t.Fatal("expected empty stack")
	}
}

func TestPushAfterClear(t *testing.T) {
	s := stack.New[int]()
	s.Push(1)
	s.Push(2)
	s.Clear()
	s.Push(99)
	if s.Size() != 1 {
		t.Fatalf("expected size 1, got %d", s.Size())
	}
	v, _ := s.Top()
	if v != 99 {
		t.Fatalf("expected 99, got %d", v)
	}
}

func TestSizeAfterPushes(t *testing.T) {
	s := stack.New[int]()
	for i := 1; i <= 10; i++ {
		s.Push(i)
		if s.Size() != i {
			t.Fatalf("expected size %d, got %d", i, s.Size())
		}
	}
}

func TestSizeAfterPops(t *testing.T) {
	s := stack.New[int]()
	for i := 0; i < 5; i++ {
		s.Push(i)
	}
	for i := 4; i >= 0; i-- {
		s.Pop()
		if s.Size() != i {
			t.Fatalf("expected size %d, got %d", i, s.Size())
		}
	}
}

func TestIsEmptyOnlyWhenSizeZero(t *testing.T) {
	s := stack.New[int]()
	if !s.IsEmpty() {
		t.Fatal("expected empty")
	}
	s.Push(1)
	if s.IsEmpty() {
		t.Fatal("expected non-empty")
	}
	s.Pop()
	if !s.IsEmpty() {
		t.Fatal("expected empty after pop")
	}
}

func TestCloneCreatesIndependentCopy(t *testing.T) {
	s := stack.New[int]()
	s.Push(1)
	s.Push(2)
	s.Push(3)

	clone := s.Clone()
	if clone.Size() != 3 {
		t.Fatalf("expected clone size 3, got %d", clone.Size())
	}

	v, _ := clone.Top()
	if v != 3 {
		t.Fatalf("expected clone top 3, got %d", v)
	}
}

func TestPushToOriginalDoesNotAffectClone(t *testing.T) {
	s := stack.New[int]()
	s.Push(1)
	s.Push(2)

	clone := s.Clone()
	s.Push(3)

	if clone.Size() != 2 {
		t.Fatalf("expected clone size 2, got %d", clone.Size())
	}
	v, _ := clone.Top()
	if v != 2 {
		t.Fatalf("expected clone top 2, got %d", v)
	}
}

func TestPopFromOriginalDoesNotAffectClone(t *testing.T) {
	s := stack.New[int]()
	s.Push(1)
	s.Push(2)
	s.Push(3)

	clone := s.Clone()
	s.Pop()

	if clone.Size() != 3 {
		t.Fatalf("expected clone size 3, got %d", clone.Size())
	}
	v, _ := clone.Top()
	if v != 3 {
		t.Fatalf("expected clone top 3, got %d", v)
	}
}

func TestWorksWithStrings(t *testing.T) {
	s := stack.New[string]()
	s.Push("hello")
	s.Push("world")

	v, _ := s.Pop()
	if v != "world" {
		t.Fatalf("expected world, got %s", v)
	}
	v, _ = s.Pop()
	if v != "hello" {
		t.Fatalf("expected hello, got %s", v)
	}
}

func TestWorksWithStructs(t *testing.T) {
	type Point struct {
		X, Y int
	}

	s := stack.New[Point]()
	s.Push(Point{1, 2})
	s.Push(Point{3, 4})

	v, _ := s.Pop()
	if v.X != 3 || v.Y != 4 {
		t.Fatalf("expected {3,4}, got {%d,%d}", v.X, v.Y)
	}
}

func TestSingleElementPushPop(t *testing.T) {
	s := stack.New[int]()
	s.Push(42)
	v, err := s.Pop()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v != 42 {
		t.Fatalf("expected 42, got %d", v)
	}
	if !s.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestAlternatingPushPop(t *testing.T) {
	s := stack.New[int]()
	for i := 0; i < 100; i++ {
		s.Push(i)
		v, _ := s.Pop()
		if v != i {
			t.Fatalf("expected %d, got %d", i, v)
		}
	}
	if !s.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestLargeNumberOfElements(t *testing.T) {
	s := stack.New[int]()
	n := 10000
	for i := 0; i < n; i++ {
		s.Push(i)
	}
	if s.Size() != n {
		t.Fatalf("expected size %d, got %d", n, s.Size())
	}
	for i := n - 1; i >= 0; i-- {
		v, err := s.Pop()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if v != i {
			t.Fatalf("expected %d, got %d", i, v)
		}
	}
	if !s.IsEmpty() {
		t.Fatal("expected empty after popping all")
	}
}
