package staticarray

import "errors"

var ErrOutOfRange = errors.New("StaticArray: index out of range")

type StaticArray[T any] struct {
	data []T
	size int
}

func New[T any](size int) *StaticArray[T] {
	return &StaticArray[T]{
		data: make([]T, size),
		size: size,
	}
}

func (a *StaticArray[T]) At(index int) (T, error) {
	var zero T
	if index < 0 || index >= a.size {
		return zero, ErrOutOfRange
	}
	return a.data[index], nil
}

func (a *StaticArray[T]) SetAt(index int, value T) error {
	if index < 0 || index >= a.size {
		return ErrOutOfRange
	}
	a.data[index] = value
	return nil
}

func (a *StaticArray[T]) Get(index int) T {
	return a.data[index]
}

func (a *StaticArray[T]) Set(index int, value T) {
	a.data[index] = value
}

func (a *StaticArray[T]) Front() T {
	return a.data[0]
}

func (a *StaticArray[T]) Back() T {
	return a.data[a.size-1]
}

func (a *StaticArray[T]) Data() []T {
	if a.size == 0 {
		return nil
	}
	return a.data
}

func (a *StaticArray[T]) Size() int {
	return a.size
}

func (a *StaticArray[T]) IsEmpty() bool {
	return a.size == 0
}

func (a *StaticArray[T]) Fill(value T) {
	for i := range a.data {
		a.data[i] = value
	}
}
