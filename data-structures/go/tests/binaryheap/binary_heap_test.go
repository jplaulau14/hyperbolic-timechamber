package binaryheap_test

import (
	"cmp"
	"errors"
	"sort"
	"testing"

	"github.com/hyperbolic-timechamber/data-structures-go/src/binaryheap"
)

func TestNewHeapIsEmpty(t *testing.T) {
	h := binaryheap.New[int]()
	if h.Size() != 0 {
		t.Fatalf("expected size 0, got %d", h.Size())
	}
	if !h.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestPeekOnEmptyHeap(t *testing.T) {
	h := binaryheap.New[int]()
	_, err := h.Peek()
	if !errors.Is(err, binaryheap.ErrEmptyHeap) {
		t.Fatal("expected ErrEmptyHeap")
	}
}

func TestPopOnEmptyHeap(t *testing.T) {
	h := binaryheap.New[int]()
	_, err := h.Pop()
	if !errors.Is(err, binaryheap.ErrEmptyHeap) {
		t.Fatal("expected ErrEmptyHeap")
	}
}

func TestPushSingleElement(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(42)
	if h.Size() != 1 {
		t.Fatalf("expected size 1, got %d", h.Size())
	}
	v, _ := h.Peek()
	if v != 42 {
		t.Fatalf("expected 42, got %d", v)
	}
}

func TestPushMultipleElements(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(30)
	h.Push(10)
	h.Push(20)
	if h.Size() != 3 {
		t.Fatalf("expected size 3, got %d", h.Size())
	}
	v, _ := h.Peek()
	if v != 10 {
		t.Fatalf("expected min 10, got %d", v)
	}
}

func TestPushAscendingOrder(t *testing.T) {
	h := binaryheap.New[int]()
	for i := 1; i <= 5; i++ {
		h.Push(i)
	}
	v, _ := h.Peek()
	if v != 1 {
		t.Fatalf("expected min 1, got %d", v)
	}
}

func TestPushDescendingOrder(t *testing.T) {
	h := binaryheap.New[int]()
	for i := 5; i >= 1; i-- {
		h.Push(i)
	}
	v, _ := h.Peek()
	if v != 1 {
		t.Fatalf("expected min 1, got %d", v)
	}
}

func TestPushRandomOrder(t *testing.T) {
	h := binaryheap.New[int]()
	values := []int{15, 3, 8, 1, 12, 6}
	for _, v := range values {
		h.Push(v)
	}
	v, _ := h.Peek()
	if v != 1 {
		t.Fatalf("expected min 1, got %d", v)
	}
}

func TestPopReturnsMinimum(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(30)
	h.Push(10)
	h.Push(20)
	v, _ := h.Pop()
	if v != 10 {
		t.Fatalf("expected 10, got %d", v)
	}
}

func TestPopRestoresHeapProperty(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(30)
	h.Push(10)
	h.Push(20)
	h.Pop()
	v, _ := h.Peek()
	if v != 20 {
		t.Fatalf("expected new min 20, got %d", v)
	}
}

func TestPopAllYieldsSortedOrder(t *testing.T) {
	h := binaryheap.New[int]()
	values := []int{5, 2, 8, 1, 9, 3}
	for _, v := range values {
		h.Push(v)
	}

	var result []int
	for !h.IsEmpty() {
		v, _ := h.Pop()
		result = append(result, v)
	}

	sorted := make([]int, len(values))
	copy(sorted, values)
	sort.Ints(sorted)

	for i := range sorted {
		if result[i] != sorted[i] {
			t.Fatalf("index %d: expected %d, got %d", i, sorted[i], result[i])
		}
	}
}

func TestSizeDecrementsAfterPop(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(1)
	h.Push(2)
	h.Push(3)
	if h.Size() != 3 {
		t.Fatal("expected size 3")
	}
	h.Pop()
	if h.Size() != 2 {
		t.Fatal("expected size 2 after pop")
	}
}

func TestPeekDoesNotRemove(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(10)
	h.Peek()
	h.Peek()
	if h.Size() != 1 {
		t.Fatal("peek should not remove element")
	}
}

func TestMultiplePeeksReturnSameValue(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(10)
	h.Push(5)
	v1, _ := h.Peek()
	v2, _ := h.Peek()
	if v1 != v2 {
		t.Fatal("multiple peeks should return same value")
	}
}

func TestPeekAfterPushShowsNewMin(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(10)
	v, _ := h.Peek()
	if v != 10 {
		t.Fatal("expected 10")
	}
	h.Push(5)
	v, _ = h.Peek()
	if v != 5 {
		t.Fatal("expected new min 5")
	}
}

func TestFromSlice(t *testing.T) {
	arr := []int{5, 2, 8, 1, 9, 3}
	h := binaryheap.FromSlice(arr)
	if h.Size() != len(arr) {
		t.Fatalf("expected size %d, got %d", len(arr), h.Size())
	}
	v, _ := h.Peek()
	if v != 1 {
		t.Fatalf("expected min 1, got %d", v)
	}
}

func TestFromSliceContainsAllElements(t *testing.T) {
	arr := []int{5, 2, 8, 1, 9, 3}
	h := binaryheap.FromSlice(arr)

	var result []int
	for !h.IsEmpty() {
		v, _ := h.Pop()
		result = append(result, v)
	}

	sorted := make([]int, len(arr))
	copy(sorted, arr)
	sort.Ints(sorted)

	if len(result) != len(sorted) {
		t.Fatal("length mismatch")
	}
	for i := range sorted {
		if result[i] != sorted[i] {
			t.Fatalf("index %d: expected %d, got %d", i, sorted[i], result[i])
		}
	}
}

func TestClearMakesHeapEmpty(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(1)
	h.Push(2)
	h.Push(3)
	h.Clear()
	if !h.IsEmpty() {
		t.Fatal("expected empty after clear")
	}
	if h.Size() != 0 {
		t.Fatal("expected size 0 after clear")
	}
}

func TestClearOnEmptyHeap(t *testing.T) {
	h := binaryheap.New[int]()
	h.Clear()
	if !h.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestHeapPropertyWithDuplicates(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(5)
	h.Push(5)
	h.Push(3)
	h.Push(3)
	h.Push(1)
	h.Push(1)

	expected := []int{1, 1, 3, 3, 5, 5}
	for _, exp := range expected {
		v, _ := h.Pop()
		if v != exp {
			t.Fatalf("expected %d, got %d", exp, v)
		}
	}
}

func TestHeapPropertyWithNegativeNumbers(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(5)
	h.Push(-3)
	h.Push(0)
	h.Push(-10)
	h.Push(2)

	expected := []int{-10, -3, 0, 2, 5}
	for _, exp := range expected {
		v, _ := h.Pop()
		if v != exp {
			t.Fatalf("expected %d, got %d", exp, v)
		}
	}
}

func TestCloneCreatesIndependentCopy(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(10)
	h.Push(5)
	h.Push(15)

	clone := h.Clone()
	if clone.Size() != h.Size() {
		t.Fatal("clone size mismatch")
	}

	v1, _ := h.Peek()
	v2, _ := clone.Peek()
	if v1 != v2 {
		t.Fatal("clone should have same min")
	}
}

func TestPushToOriginalDoesNotAffectClone(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(10)
	h.Push(5)

	clone := h.Clone()
	h.Push(1)

	v, _ := clone.Peek()
	if v != 5 {
		t.Fatal("clone should not be affected by push to original")
	}
	if clone.Size() != 2 {
		t.Fatal("clone size should not change")
	}
}

func TestPopFromOriginalDoesNotAffectClone(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(10)
	h.Push(5)

	clone := h.Clone()
	h.Pop()

	v, _ := clone.Peek()
	if v != 5 {
		t.Fatal("clone should not be affected by pop from original")
	}
	if clone.Size() != 2 {
		t.Fatal("clone size should not change")
	}
}

func TestWorksWithFloatingPoint(t *testing.T) {
	h := binaryheap.New[float64]()
	h.Push(3.14)
	h.Push(1.5)
	h.Push(2.71)
	h.Push(0.5)

	v, _ := h.Pop()
	if v != 0.5 {
		t.Fatalf("expected 0.5, got %f", v)
	}
	v, _ = h.Pop()
	if v != 1.5 {
		t.Fatalf("expected 1.5, got %f", v)
	}
}

func TestWorksWithStrings(t *testing.T) {
	h := binaryheap.New[string]()
	h.Push("banana")
	h.Push("apple")
	h.Push("cherry")

	v, _ := h.Pop()
	if v != "apple" {
		t.Fatalf("expected apple, got %s", v)
	}
	v, _ = h.Pop()
	if v != "banana" {
		t.Fatalf("expected banana, got %s", v)
	}
}

func TestSingleElementHeap(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(42)
	v, _ := h.Pop()
	if v != 42 {
		t.Fatal("expected 42")
	}
	if !h.IsEmpty() {
		t.Fatal("expected empty after popping single element")
	}
}

func TestTwoElementHeap(t *testing.T) {
	h := binaryheap.New[int]()
	h.Push(20)
	h.Push(10)

	v, _ := h.Pop()
	if v != 10 {
		t.Fatal("expected 10")
	}
	v, _ = h.Pop()
	if v != 20 {
		t.Fatal("expected 20")
	}
}

func TestLargeNumberOfElements(t *testing.T) {
	h := binaryheap.New[int]()
	n := 1000
	for i := n; i > 0; i-- {
		h.Push(i)
	}
	if h.Size() != n {
		t.Fatalf("expected size %d, got %d", n, h.Size())
	}

	prev := 0
	for !h.IsEmpty() {
		v, _ := h.Pop()
		if v < prev {
			t.Fatal("heap property violated")
		}
		prev = v
	}
}

func TestManyPushPopCycles(t *testing.T) {
	h := binaryheap.New[int]()

	for cycle := 0; cycle < 100; cycle++ {
		h.Push(cycle * 3)
		h.Push(cycle * 2)
		h.Push(cycle)
		h.Pop()
	}

	if h.Size() != 200 {
		t.Fatalf("expected size 200, got %d", h.Size())
	}

	prev := -1
	for !h.IsEmpty() {
		v, _ := h.Pop()
		if v < prev {
			t.Fatal("heap property violated after many cycles")
		}
		prev = v
	}
}

func verifyHeapProperty[T cmp.Ordered](data []T) bool {
	n := len(data)
	for i := 0; i < n; i++ {
		left := 2*i + 1
		right := 2*i + 2
		if left < n && cmp.Less(data[left], data[i]) {
			return false
		}
		if right < n && cmp.Less(data[right], data[i]) {
			return false
		}
	}
	return true
}

func TestHeapPropertyAfterOperations(t *testing.T) {
	h := binaryheap.New[int]()
	values := []int{15, 3, 8, 1, 12, 6, 9, 2}
	for _, v := range values {
		h.Push(v)
	}

	clone := h.Clone()
	var data []int
	for !clone.IsEmpty() {
		v, _ := clone.Pop()
		data = append(data, v)
	}

	for i := 1; i < len(data); i++ {
		if data[i] < data[i-1] {
			t.Fatal("popped elements should be in sorted order")
		}
	}
}

func TestFromSliceDoesNotModifyOriginal(t *testing.T) {
	arr := []int{5, 2, 8, 1, 9, 3}
	original := make([]int, len(arr))
	copy(original, arr)

	binaryheap.FromSlice(arr)

	for i := range arr {
		if arr[i] != original[i] {
			t.Fatal("FromSlice should not modify original slice")
		}
	}
}
