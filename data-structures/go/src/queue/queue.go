package queue

import "errors"

var ErrEmptyQueue = errors.New("Queue: empty queue")

type Queue[T any] struct {
	data     []T
	head     int
	tail     int
	size     int
	capacity int
}

func New[T any]() *Queue[T] {
	return &Queue[T]{}
}

func (q *Queue[T]) Enqueue(value T) {
	if q.size == q.capacity {
		q.grow()
	}
	q.data[q.tail] = value
	q.tail = (q.tail + 1) % q.capacity
	q.size++
}

func (q *Queue[T]) Dequeue() (T, error) {
	var zero T
	if q.size == 0 {
		return zero, ErrEmptyQueue
	}
	value := q.data[q.head]
	q.data[q.head] = zero
	q.head = (q.head + 1) % q.capacity
	q.size--
	return value, nil
}

func (q *Queue[T]) Front() (T, error) {
	var zero T
	if q.size == 0 {
		return zero, ErrEmptyQueue
	}
	return q.data[q.head], nil
}

func (q *Queue[T]) Back() (T, error) {
	var zero T
	if q.size == 0 {
		return zero, ErrEmptyQueue
	}
	idx := (q.tail - 1 + q.capacity) % q.capacity
	return q.data[idx], nil
}

func (q *Queue[T]) Size() int {
	return q.size
}

func (q *Queue[T]) IsEmpty() bool {
	return q.size == 0
}

func (q *Queue[T]) Clear() {
	var zero T
	for i := 0; i < q.capacity; i++ {
		q.data[i] = zero
	}
	q.head = 0
	q.tail = 0
	q.size = 0
}

func (q *Queue[T]) Clone() *Queue[T] {
	clone := &Queue[T]{
		data:     make([]T, q.capacity),
		head:     0,
		tail:     q.size,
		size:     q.size,
		capacity: q.capacity,
	}
	for i := 0; i < q.size; i++ {
		clone.data[i] = q.data[(q.head+i)%q.capacity]
	}
	return clone
}

func (q *Queue[T]) grow() {
	newCap := 1
	if q.capacity > 0 {
		newCap = q.capacity * 2
	}
	newData := make([]T, newCap)
	for i := 0; i < q.size; i++ {
		newData[i] = q.data[(q.head+i)%q.capacity]
	}
	q.data = newData
	q.head = 0
	q.tail = q.size
	q.capacity = newCap
}
