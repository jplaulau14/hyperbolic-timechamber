package queue_test

import (
	"errors"
	"testing"

	"github.com/hyperbolic-timechamber/data-structures-go/src/queue"
)

func TestNewQueueIsEmpty(t *testing.T) {
	q := queue.New[int]()
	if q.Size() != 0 {
		t.Fatalf("expected size 0, got %d", q.Size())
	}
	if !q.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestFrontOnEmptyQueue(t *testing.T) {
	q := queue.New[int]()
	_, err := q.Front()
	if !errors.Is(err, queue.ErrEmptyQueue) {
		t.Fatal("expected ErrEmptyQueue")
	}
}

func TestBackOnEmptyQueue(t *testing.T) {
	q := queue.New[int]()
	_, err := q.Back()
	if !errors.Is(err, queue.ErrEmptyQueue) {
		t.Fatal("expected ErrEmptyQueue")
	}
}

func TestDequeueOnEmptyQueue(t *testing.T) {
	q := queue.New[int]()
	_, err := q.Dequeue()
	if !errors.Is(err, queue.ErrEmptyQueue) {
		t.Fatal("expected ErrEmptyQueue")
	}
}

func TestEnqueueSingleElement(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(42)
	if q.Size() != 1 {
		t.Fatalf("expected size 1, got %d", q.Size())
	}
}

func TestEnqueueMultipleElements(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Enqueue(2)
	q.Enqueue(3)
	if q.Size() != 3 {
		t.Fatalf("expected size 3, got %d", q.Size())
	}
}

func TestFrontAndBackAfterMultipleEnqueues(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(10)
	q.Enqueue(20)
	q.Enqueue(30)
	front, _ := q.Front()
	back, _ := q.Back()
	if front != 10 {
		t.Fatalf("expected front 10, got %d", front)
	}
	if back != 30 {
		t.Fatalf("expected back 30, got %d", back)
	}
}

func TestDequeueReturnsFrontElement(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Enqueue(2)
	val, _ := q.Dequeue()
	if val != 1 {
		t.Fatalf("expected 1, got %d", val)
	}
}

func TestDequeueDecrementsSize(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Enqueue(2)
	q.Dequeue()
	if q.Size() != 1 {
		t.Fatalf("expected size 1, got %d", q.Size())
	}
}

func TestDequeueAllElements(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Enqueue(2)
	q.Enqueue(3)
	q.Dequeue()
	q.Dequeue()
	q.Dequeue()
	if !q.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestFIFOOrder(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Enqueue(2)
	q.Enqueue(3)
	v1, _ := q.Dequeue()
	v2, _ := q.Dequeue()
	v3, _ := q.Dequeue()
	if v1 != 1 || v2 != 2 || v3 != 3 {
		t.Fatalf("expected 1,2,3 got %d,%d,%d", v1, v2, v3)
	}
}

func TestFrontDoesNotRemove(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(5)
	q.Front()
	q.Front()
	if q.Size() != 1 {
		t.Fatalf("expected size 1, got %d", q.Size())
	}
}

func TestBackDoesNotRemove(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(5)
	q.Back()
	q.Back()
	if q.Size() != 1 {
		t.Fatalf("expected size 1, got %d", q.Size())
	}
}

func TestFrontAndBackSameWhenSingleElement(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(99)
	front, _ := q.Front()
	back, _ := q.Back()
	if front != back {
		t.Fatalf("expected front == back, got %d and %d", front, back)
	}
}

func TestCircularBufferWrapAround(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Enqueue(2)
	q.Dequeue()
	q.Enqueue(3)
	q.Dequeue()
	q.Enqueue(4)

	v, _ := q.Front()
	if v != 3 {
		t.Fatalf("expected front 3, got %d", v)
	}
	v, _ = q.Back()
	if v != 4 {
		t.Fatalf("expected back 4, got %d", v)
	}
}

func TestFillDequeueSomeEnqueueMore(t *testing.T) {
	q := queue.New[int]()
	for i := 0; i < 4; i++ {
		q.Enqueue(i)
	}
	q.Dequeue()
	q.Dequeue()
	q.Enqueue(10)
	q.Enqueue(11)

	expected := []int{2, 3, 10, 11}
	for _, exp := range expected {
		v, _ := q.Dequeue()
		if v != exp {
			t.Fatalf("expected %d, got %d", exp, v)
		}
	}
}

func TestGrowthPreservesOrder(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Enqueue(2)
	q.Dequeue()
	q.Enqueue(3)
	q.Enqueue(4)
	q.Enqueue(5)
	q.Enqueue(6)

	expected := []int{2, 3, 4, 5, 6}
	for _, exp := range expected {
		v, _ := q.Dequeue()
		if v != exp {
			t.Fatalf("expected %d, got %d", exp, v)
		}
	}
}

func TestClearMakesEmpty(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Enqueue(2)
	q.Clear()
	if !q.IsEmpty() {
		t.Fatal("expected empty after clear")
	}
	if q.Size() != 0 {
		t.Fatalf("expected size 0, got %d", q.Size())
	}
}

func TestClearOnEmptyQueue(t *testing.T) {
	q := queue.New[int]()
	q.Clear()
	if !q.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestEnqueueAfterClear(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Clear()
	q.Enqueue(2)
	v, _ := q.Front()
	if v != 2 {
		t.Fatalf("expected 2, got %d", v)
	}
}

func TestSizeAfterEnqueues(t *testing.T) {
	q := queue.New[int]()
	for i := 0; i < 5; i++ {
		q.Enqueue(i)
	}
	if q.Size() != 5 {
		t.Fatalf("expected size 5, got %d", q.Size())
	}
}

func TestSizeAfterDequeues(t *testing.T) {
	q := queue.New[int]()
	for i := 0; i < 5; i++ {
		q.Enqueue(i)
	}
	q.Dequeue()
	q.Dequeue()
	if q.Size() != 3 {
		t.Fatalf("expected size 3, got %d", q.Size())
	}
}

func TestIsEmptyOnlyWhenSizeZero(t *testing.T) {
	q := queue.New[int]()
	if !q.IsEmpty() {
		t.Fatal("expected empty")
	}
	q.Enqueue(1)
	if q.IsEmpty() {
		t.Fatal("expected not empty")
	}
	q.Dequeue()
	if !q.IsEmpty() {
		t.Fatal("expected empty after dequeue")
	}
}

func TestCloneCreatesIndependentCopy(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Enqueue(2)
	q.Enqueue(3)

	clone := q.Clone()
	if clone.Size() != 3 {
		t.Fatalf("expected clone size 3, got %d", clone.Size())
	}
}

func TestEnqueueToOriginalDoesNotAffectClone(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	clone := q.Clone()
	q.Enqueue(2)

	if clone.Size() != 1 {
		t.Fatal("clone should be unaffected")
	}
}

func TestDequeueFromOriginalDoesNotAffectClone(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Enqueue(2)
	clone := q.Clone()
	q.Dequeue()

	if clone.Size() != 2 {
		t.Fatal("clone should be unaffected")
	}
	v, _ := clone.Front()
	if v != 1 {
		t.Fatalf("expected clone front 1, got %d", v)
	}
}

func TestWorksWithStrings(t *testing.T) {
	q := queue.New[string]()
	q.Enqueue("hello")
	q.Enqueue("world")
	v, _ := q.Dequeue()
	if v != "hello" {
		t.Fatalf("expected hello, got %s", v)
	}
	v, _ = q.Dequeue()
	if v != "world" {
		t.Fatalf("expected world, got %s", v)
	}
}

type Point struct {
	X, Y int
}

func TestWorksWithStructs(t *testing.T) {
	q := queue.New[Point]()
	q.Enqueue(Point{1, 2})
	q.Enqueue(Point{3, 4})
	v, _ := q.Dequeue()
	if v.X != 1 || v.Y != 2 {
		t.Fatalf("expected {1,2}, got %v", v)
	}
}

func TestSingleElementEnqueueDequeue(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(42)
	v, err := q.Dequeue()
	if err != nil {
		t.Fatal("unexpected error")
	}
	if v != 42 {
		t.Fatalf("expected 42, got %d", v)
	}
	if !q.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestAlternatingEnqueueDequeue(t *testing.T) {
	q := queue.New[int]()
	for i := 0; i < 100; i++ {
		q.Enqueue(i)
		v, _ := q.Dequeue()
		if v != i {
			t.Fatalf("expected %d, got %d", i, v)
		}
	}
	if !q.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestLargeNumberOfElements(t *testing.T) {
	q := queue.New[int]()
	n := 10000
	for i := 0; i < n; i++ {
		q.Enqueue(i)
	}
	if q.Size() != n {
		t.Fatalf("expected size %d, got %d", n, q.Size())
	}
	for i := 0; i < n; i++ {
		v, _ := q.Dequeue()
		if v != i {
			t.Fatalf("expected %d, got %d", i, v)
		}
	}
}

func TestManyWrapAroundCycles(t *testing.T) {
	q := queue.New[int]()
	for cycle := 0; cycle < 100; cycle++ {
		for i := 0; i < 10; i++ {
			q.Enqueue(cycle*10 + i)
		}
		for i := 0; i < 10; i++ {
			v, _ := q.Dequeue()
			expected := cycle*10 + i
			if v != expected {
				t.Fatalf("cycle %d: expected %d, got %d", cycle, expected, v)
			}
		}
	}
	if !q.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestCloneWithWrappedBuffer(t *testing.T) {
	q := queue.New[int]()
	q.Enqueue(1)
	q.Enqueue(2)
	q.Dequeue()
	q.Enqueue(3)
	q.Enqueue(4)

	clone := q.Clone()
	expected := []int{2, 3, 4}
	for _, exp := range expected {
		v, _ := clone.Dequeue()
		if v != exp {
			t.Fatalf("expected %d, got %d", exp, v)
		}
	}
}
