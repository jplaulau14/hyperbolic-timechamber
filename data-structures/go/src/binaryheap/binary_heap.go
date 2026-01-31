package binaryheap

import (
	"cmp"
	"errors"
)

var ErrEmptyHeap = errors.New("BinaryHeap: heap is empty")

type BinaryHeap[T cmp.Ordered] struct {
	data []T
}

func New[T cmp.Ordered]() *BinaryHeap[T] {
	return &BinaryHeap[T]{}
}

func FromSlice[T cmp.Ordered](arr []T) *BinaryHeap[T] {
	h := &BinaryHeap[T]{
		data: make([]T, len(arr)),
	}
	copy(h.data, arr)
	for i := len(h.data)/2 - 1; i >= 0; i-- {
		h.heapifyDown(i)
	}
	return h
}

func (h *BinaryHeap[T]) Push(value T) {
	h.data = append(h.data, value)
	h.heapifyUp(len(h.data) - 1)
}

func (h *BinaryHeap[T]) Pop() (T, error) {
	var zero T
	if len(h.data) == 0 {
		return zero, ErrEmptyHeap
	}
	min := h.data[0]
	last := len(h.data) - 1
	h.data[0] = h.data[last]
	h.data = h.data[:last]
	if len(h.data) > 0 {
		h.heapifyDown(0)
	}
	return min, nil
}

func (h *BinaryHeap[T]) Peek() (T, error) {
	var zero T
	if len(h.data) == 0 {
		return zero, ErrEmptyHeap
	}
	return h.data[0], nil
}

func (h *BinaryHeap[T]) Size() int {
	return len(h.data)
}

func (h *BinaryHeap[T]) IsEmpty() bool {
	return len(h.data) == 0
}

func (h *BinaryHeap[T]) Clear() {
	h.data = h.data[:0]
}

func (h *BinaryHeap[T]) Clone() *BinaryHeap[T] {
	clone := &BinaryHeap[T]{
		data: make([]T, len(h.data)),
	}
	copy(clone.data, h.data)
	return clone
}

func (h *BinaryHeap[T]) heapifyUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.data[i] >= h.data[parent] {
			break
		}
		h.data[i], h.data[parent] = h.data[parent], h.data[i]
		i = parent
	}
}

func (h *BinaryHeap[T]) heapifyDown(i int) {
	n := len(h.data)
	for {
		smallest := i
		left := 2*i + 1
		right := 2*i + 2

		if left < n && h.data[left] < h.data[smallest] {
			smallest = left
		}
		if right < n && h.data[right] < h.data[smallest] {
			smallest = right
		}
		if smallest == i {
			break
		}
		h.data[i], h.data[smallest] = h.data[smallest], h.data[i]
		i = smallest
	}
}
