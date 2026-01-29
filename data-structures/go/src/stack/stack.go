package stack

import "errors"

var ErrEmptyStack = errors.New("Stack: empty stack")

type Stack[T any] struct {
	data []T
}

func New[T any]() *Stack[T] {
	return &Stack[T]{}
}

func (s *Stack[T]) Push(value T) {
	s.data = append(s.data, value)
}

func (s *Stack[T]) Pop() (T, error) {
	var zero T
	if len(s.data) == 0 {
		return zero, ErrEmptyStack
	}
	idx := len(s.data) - 1
	value := s.data[idx]
	s.data[idx] = zero
	s.data = s.data[:idx]
	return value, nil
}

func (s *Stack[T]) Top() (T, error) {
	var zero T
	if len(s.data) == 0 {
		return zero, ErrEmptyStack
	}
	return s.data[len(s.data)-1], nil
}

func (s *Stack[T]) Size() int {
	return len(s.data)
}

func (s *Stack[T]) IsEmpty() bool {
	return len(s.data) == 0
}

func (s *Stack[T]) Clear() {
	var zero T
	for i := range s.data {
		s.data[i] = zero
	}
	s.data = s.data[:0]
}

func (s *Stack[T]) Clone() *Stack[T] {
	clone := &Stack[T]{
		data: make([]T, len(s.data)),
	}
	copy(clone.data, s.data)
	return clone
}
