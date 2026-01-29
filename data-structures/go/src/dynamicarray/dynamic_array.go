package dynamicarray

import "errors"

var ErrOutOfRange = errors.New("DynamicArray: index out of range")
var ErrEmpty = errors.New("DynamicArray: array is empty")

type DynamicArray[T any] struct {
	data     []T
	size     int
	capacity int
}

func New[T any]() *DynamicArray[T] {
	return &DynamicArray[T]{}
}

func NewWithSize[T any](size int) *DynamicArray[T] {
	if size <= 0 {
		return &DynamicArray[T]{}
	}
	return &DynamicArray[T]{
		data:     make([]T, size),
		size:     size,
		capacity: size,
	}
}

func (a *DynamicArray[T]) At(index int) (T, error) {
	var zero T
	if index < 0 || index >= a.size {
		return zero, ErrOutOfRange
	}
	return a.data[index], nil
}

func (a *DynamicArray[T]) SetAt(index int, value T) error {
	if index < 0 || index >= a.size {
		return ErrOutOfRange
	}
	a.data[index] = value
	return nil
}

func (a *DynamicArray[T]) Get(index int) T {
	return a.data[index]
}

func (a *DynamicArray[T]) Set(index int, value T) {
	a.data[index] = value
}

func (a *DynamicArray[T]) Front() (T, error) {
	var zero T
	if a.size == 0 {
		return zero, ErrEmpty
	}
	return a.data[0], nil
}

func (a *DynamicArray[T]) Back() (T, error) {
	var zero T
	if a.size == 0 {
		return zero, ErrEmpty
	}
	return a.data[a.size-1], nil
}

func (a *DynamicArray[T]) Data() []T {
	if a.size == 0 {
		return nil
	}
	return a.data[:a.size]
}

func (a *DynamicArray[T]) Size() int {
	return a.size
}

func (a *DynamicArray[T]) Capacity() int {
	return a.capacity
}

func (a *DynamicArray[T]) IsEmpty() bool {
	return a.size == 0
}

func (a *DynamicArray[T]) Reserve(newCap int) {
	if newCap <= a.capacity {
		return
	}
	newData := make([]T, newCap)
	copy(newData, a.data[:a.size])
	a.data = newData
	a.capacity = newCap
}

func (a *DynamicArray[T]) PushBack(value T) {
	if a.size == a.capacity {
		newCap := 1
		if a.capacity > 0 {
			newCap = a.capacity * 2
		}
		a.Reserve(newCap)
	}
	a.data[a.size] = value
	a.size++
}

func (a *DynamicArray[T]) PopBack() error {
	if a.size == 0 {
		return ErrEmpty
	}
	a.size--
	var zero T
	a.data[a.size] = zero
	return nil
}

func (a *DynamicArray[T]) Clear() {
	var zero T
	for i := 0; i < a.size; i++ {
		a.data[i] = zero
	}
	a.size = 0
}

func (a *DynamicArray[T]) Clone() *DynamicArray[T] {
	clone := &DynamicArray[T]{
		data:     make([]T, a.capacity),
		size:     a.size,
		capacity: a.capacity,
	}
	copy(clone.data, a.data[:a.size])
	return clone
}
