package linkedlist_test

import (
	"errors"
	"testing"

	"github.com/hyperbolic-timechamber/data-structures-go/src/linkedlist"
)

func TestNewListIsEmpty(t *testing.T) {
	list := linkedlist.New[int]()
	if list.Size() != 0 {
		t.Fatalf("expected size 0, got %d", list.Size())
	}
	if !list.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestFrontBackOnEmptyList(t *testing.T) {
	list := linkedlist.New[int]()
	_, err := list.Front()
	if !errors.Is(err, linkedlist.ErrEmptyList) {
		t.Fatal("expected ErrEmptyList for Front")
	}
	_, err = list.Back()
	if !errors.Is(err, linkedlist.ErrEmptyList) {
		t.Fatal("expected ErrEmptyList for Back")
	}
}

func TestPushFrontSingle(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushFront(42)
	if list.Size() != 1 {
		t.Fatalf("expected size 1, got %d", list.Size())
	}
	front, _ := list.Front()
	back, _ := list.Back()
	if front != 42 || back != 42 {
		t.Fatal("front and back should be 42")
	}
}

func TestPushFrontMultiple(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushFront(1)
	list.PushFront(2)
	list.PushFront(3)
	values := list.Values()
	expected := []int{3, 2, 1}
	for i, v := range expected {
		if values[i] != v {
			t.Fatalf("index %d: expected %d, got %d", i, v, values[i])
		}
	}
}

func TestPushBackSingle(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(42)
	if list.Size() != 1 {
		t.Fatalf("expected size 1, got %d", list.Size())
	}
	front, _ := list.Front()
	back, _ := list.Back()
	if front != 42 || back != 42 {
		t.Fatal("front and back should be 42")
	}
}

func TestPushBackMultiple(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.PushBack(3)
	values := list.Values()
	expected := []int{1, 2, 3}
	for i, v := range expected {
		if values[i] != v {
			t.Fatalf("index %d: expected %d, got %d", i, v, values[i])
		}
	}
}

func TestMixedPushFrontAndBack(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(2)
	list.PushFront(1)
	list.PushBack(3)
	list.PushFront(0)
	values := list.Values()
	expected := []int{0, 1, 2, 3}
	for i, v := range expected {
		if values[i] != v {
			t.Fatalf("index %d: expected %d, got %d", i, v, values[i])
		}
	}
}

func TestPopFrontReturnsCorrectValue(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.PushBack(3)
	v, err := list.PopFront()
	if err != nil {
		t.Fatal("unexpected error")
	}
	if v != 1 {
		t.Fatalf("expected 1, got %d", v)
	}
	if list.Size() != 2 {
		t.Fatalf("expected size 2, got %d", list.Size())
	}
}

func TestPopFrontUntilEmpty(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.PopFront()
	list.PopFront()
	if !list.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestPopFrontOnEmptyList(t *testing.T) {
	list := linkedlist.New[int]()
	_, err := list.PopFront()
	if !errors.Is(err, linkedlist.ErrEmptyList) {
		t.Fatal("expected ErrEmptyList")
	}
}

func TestPopBackReturnsCorrectValue(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.PushBack(3)
	v, err := list.PopBack()
	if err != nil {
		t.Fatal("unexpected error")
	}
	if v != 3 {
		t.Fatalf("expected 3, got %d", v)
	}
	if list.Size() != 2 {
		t.Fatalf("expected size 2, got %d", list.Size())
	}
}

func TestPopBackUntilEmpty(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.PopBack()
	list.PopBack()
	if !list.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestPopBackOnEmptyList(t *testing.T) {
	list := linkedlist.New[int]()
	_, err := list.PopBack()
	if !errors.Is(err, linkedlist.ErrEmptyList) {
		t.Fatal("expected ErrEmptyList")
	}
}

func TestFrontReturnsWithoutRemoving(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	v, _ := list.Front()
	if v != 1 {
		t.Fatalf("expected 1, got %d", v)
	}
	if list.Size() != 2 {
		t.Fatal("front should not remove element")
	}
}

func TestBackReturnsWithoutRemoving(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	v, _ := list.Back()
	if v != 2 {
		t.Fatalf("expected 2, got %d", v)
	}
	if list.Size() != 2 {
		t.Fatal("back should not remove element")
	}
}

func TestAtFirst(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(10)
	list.PushBack(20)
	list.PushBack(30)
	v, err := list.At(0)
	if err != nil {
		t.Fatal("unexpected error")
	}
	if v != 10 {
		t.Fatalf("expected 10, got %d", v)
	}
}

func TestAtLast(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(10)
	list.PushBack(20)
	list.PushBack(30)
	v, err := list.At(2)
	if err != nil {
		t.Fatal("unexpected error")
	}
	if v != 30 {
		t.Fatalf("expected 30, got %d", v)
	}
}

func TestAtMiddle(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(10)
	list.PushBack(20)
	list.PushBack(30)
	v, err := list.At(1)
	if err != nil {
		t.Fatal("unexpected error")
	}
	if v != 20 {
		t.Fatalf("expected 20, got %d", v)
	}
}

func TestAtOutOfRange(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	_, err := list.At(-1)
	if !errors.Is(err, linkedlist.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for negative index")
	}
	_, err = list.At(1)
	if !errors.Is(err, linkedlist.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for index >= size")
	}
	_, err = list.At(100)
	if !errors.Is(err, linkedlist.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for large index")
	}
}

func TestInsertAtZero(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(2)
	list.PushBack(3)
	list.InsertAt(0, 1)
	values := list.Values()
	expected := []int{1, 2, 3}
	for i, v := range expected {
		if values[i] != v {
			t.Fatalf("index %d: expected %d, got %d", i, v, values[i])
		}
	}
}

func TestInsertAtEnd(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.InsertAt(2, 3)
	values := list.Values()
	expected := []int{1, 2, 3}
	for i, v := range expected {
		if values[i] != v {
			t.Fatalf("index %d: expected %d, got %d", i, v, values[i])
		}
	}
}

func TestInsertAtMiddle(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(3)
	list.InsertAt(1, 2)
	values := list.Values()
	expected := []int{1, 2, 3}
	for i, v := range expected {
		if values[i] != v {
			t.Fatalf("index %d: expected %d, got %d", i, v, values[i])
		}
	}
}

func TestInsertAtInvalid(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	err := list.InsertAt(-1, 0)
	if !errors.Is(err, linkedlist.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for negative index")
	}
	err = list.InsertAt(5, 0)
	if !errors.Is(err, linkedlist.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for index > size")
	}
}

func TestRemoveAtZero(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.PushBack(3)
	v, err := list.RemoveAt(0)
	if err != nil {
		t.Fatal("unexpected error")
	}
	if v != 1 {
		t.Fatalf("expected 1, got %d", v)
	}
	values := list.Values()
	expected := []int{2, 3}
	for i, exp := range expected {
		if values[i] != exp {
			t.Fatalf("index %d: expected %d, got %d", i, exp, values[i])
		}
	}
}

func TestRemoveAtLast(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.PushBack(3)
	v, err := list.RemoveAt(2)
	if err != nil {
		t.Fatal("unexpected error")
	}
	if v != 3 {
		t.Fatalf("expected 3, got %d", v)
	}
	back, _ := list.Back()
	if back != 2 {
		t.Fatalf("expected back to be 2, got %d", back)
	}
}

func TestRemoveAtMiddle(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.PushBack(3)
	v, err := list.RemoveAt(1)
	if err != nil {
		t.Fatal("unexpected error")
	}
	if v != 2 {
		t.Fatalf("expected 2, got %d", v)
	}
	values := list.Values()
	expected := []int{1, 3}
	for i, exp := range expected {
		if values[i] != exp {
			t.Fatalf("index %d: expected %d, got %d", i, exp, values[i])
		}
	}
}

func TestRemoveAtInvalid(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	_, err := list.RemoveAt(-1)
	if !errors.Is(err, linkedlist.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for negative index")
	}
	_, err = list.RemoveAt(1)
	if !errors.Is(err, linkedlist.ErrOutOfRange) {
		t.Fatal("expected ErrOutOfRange for index >= size")
	}
}

func TestClearMakesListEmpty(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.PushBack(3)
	list.Clear()
	if !list.IsEmpty() {
		t.Fatal("expected empty after clear")
	}
	if list.Size() != 0 {
		t.Fatalf("expected size 0, got %d", list.Size())
	}
}

func TestClearOnEmptyListIsNoop(t *testing.T) {
	list := linkedlist.New[int]()
	list.Clear()
	if !list.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestIterationOrder(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.PushBack(3)
	list.PushBack(4)
	sum := 0
	for _, v := range list.Values() {
		sum += v
	}
	if sum != 10 {
		t.Fatalf("expected sum 10, got %d", sum)
	}
}

func TestIterationEmptyList(t *testing.T) {
	list := linkedlist.New[int]()
	count := 0
	for range list.Values() {
		count++
	}
	if count != 0 {
		t.Fatal("expected zero iterations")
	}
}

func TestCloneCreatesIndependentCopy(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	list.PushBack(3)
	clone := list.Clone()
	if clone.Size() != 3 {
		t.Fatalf("expected size 3, got %d", clone.Size())
	}
	cloneValues := clone.Values()
	expected := []int{1, 2, 3}
	for i, v := range expected {
		if cloneValues[i] != v {
			t.Fatalf("index %d: expected %d, got %d", i, v, cloneValues[i])
		}
	}
}

func TestModifyingOriginalDoesNotAffectClone(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	clone := list.Clone()
	list.PopFront()
	list.PushBack(999)
	cloneValues := clone.Values()
	expected := []int{1, 2}
	if clone.Size() != 2 {
		t.Fatalf("clone size should be 2, got %d", clone.Size())
	}
	for i, v := range expected {
		if cloneValues[i] != v {
			t.Fatalf("clone index %d: expected %d, got %d", i, v, cloneValues[i])
		}
	}
}

func TestWorksWithStrings(t *testing.T) {
	list := linkedlist.New[string]()
	list.PushBack("hello")
	list.PushBack("world")
	if list.Size() != 2 {
		t.Fatalf("expected size 2, got %d", list.Size())
	}
	front, _ := list.Front()
	back, _ := list.Back()
	if front != "hello" {
		t.Fatalf("expected front hello, got %s", front)
	}
	if back != "world" {
		t.Fatalf("expected back world, got %s", back)
	}
}

func TestSingleElementList(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(42)
	front, _ := list.Front()
	back, _ := list.Back()
	if front != back {
		t.Fatal("front and back should be equal for single element")
	}
	v, _ := list.PopFront()
	if v != 42 {
		t.Fatalf("expected 42, got %d", v)
	}
	if !list.IsEmpty() {
		t.Fatal("expected empty after pop")
	}
}

func TestTwoElementList(t *testing.T) {
	list := linkedlist.New[int]()
	list.PushBack(1)
	list.PushBack(2)
	front, _ := list.Front()
	back, _ := list.Back()
	if front != 1 {
		t.Fatalf("expected front 1, got %d", front)
	}
	if back != 2 {
		t.Fatalf("expected back 2, got %d", back)
	}
	list.PopFront()
	front, _ = list.Front()
	back, _ = list.Back()
	if front != back || front != 2 {
		t.Fatal("after pop, front and back should both be 2")
	}
}

func TestValuesReturnsNilForEmptyList(t *testing.T) {
	list := linkedlist.New[int]()
	if list.Values() != nil {
		t.Fatal("expected nil for empty list")
	}
}
