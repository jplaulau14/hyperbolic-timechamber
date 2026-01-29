package linkedlist

import "errors"

var (
	ErrOutOfRange = errors.New("LinkedList: index out of range")
	ErrEmptyList  = errors.New("LinkedList: list is empty")
)

type node[T any] struct {
	value T
	next  *node[T]
}

type LinkedList[T any] struct {
	head *node[T]
	tail *node[T]
	size int
}

func New[T any]() *LinkedList[T] {
	return &LinkedList[T]{}
}

func (l *LinkedList[T]) Size() int {
	return l.size
}

func (l *LinkedList[T]) IsEmpty() bool {
	return l.size == 0
}

func (l *LinkedList[T]) PushFront(value T) {
	n := &node[T]{value: value, next: l.head}
	l.head = n
	if l.tail == nil {
		l.tail = n
	}
	l.size++
}

func (l *LinkedList[T]) PushBack(value T) {
	n := &node[T]{value: value}
	if l.tail == nil {
		l.head = n
		l.tail = n
	} else {
		l.tail.next = n
		l.tail = n
	}
	l.size++
}

func (l *LinkedList[T]) PopFront() (T, error) {
	var zero T
	if l.head == nil {
		return zero, ErrEmptyList
	}
	value := l.head.value
	l.head = l.head.next
	if l.head == nil {
		l.tail = nil
	}
	l.size--
	return value, nil
}

func (l *LinkedList[T]) PopBack() (T, error) {
	var zero T
	if l.head == nil {
		return zero, ErrEmptyList
	}
	if l.head == l.tail {
		value := l.head.value
		l.head = nil
		l.tail = nil
		l.size--
		return value, nil
	}
	curr := l.head
	for curr.next != l.tail {
		curr = curr.next
	}
	value := l.tail.value
	curr.next = nil
	l.tail = curr
	l.size--
	return value, nil
}

func (l *LinkedList[T]) Front() (T, error) {
	var zero T
	if l.head == nil {
		return zero, ErrEmptyList
	}
	return l.head.value, nil
}

func (l *LinkedList[T]) Back() (T, error) {
	var zero T
	if l.tail == nil {
		return zero, ErrEmptyList
	}
	return l.tail.value, nil
}

func (l *LinkedList[T]) At(index int) (T, error) {
	var zero T
	if index < 0 || index >= l.size {
		return zero, ErrOutOfRange
	}
	curr := l.head
	for i := 0; i < index; i++ {
		curr = curr.next
	}
	return curr.value, nil
}

func (l *LinkedList[T]) InsertAt(index int, value T) error {
	if index < 0 || index > l.size {
		return ErrOutOfRange
	}
	if index == 0 {
		l.PushFront(value)
		return nil
	}
	if index == l.size {
		l.PushBack(value)
		return nil
	}
	curr := l.head
	for i := 0; i < index-1; i++ {
		curr = curr.next
	}
	n := &node[T]{value: value, next: curr.next}
	curr.next = n
	l.size++
	return nil
}

func (l *LinkedList[T]) RemoveAt(index int) (T, error) {
	var zero T
	if index < 0 || index >= l.size {
		return zero, ErrOutOfRange
	}
	if index == 0 {
		return l.PopFront()
	}
	curr := l.head
	for i := 0; i < index-1; i++ {
		curr = curr.next
	}
	value := curr.next.value
	curr.next = curr.next.next
	if curr.next == nil {
		l.tail = curr
	}
	l.size--
	return value, nil
}

func (l *LinkedList[T]) Clear() {
	l.head = nil
	l.tail = nil
	l.size = 0
}

func (l *LinkedList[T]) Clone() *LinkedList[T] {
	clone := New[T]()
	curr := l.head
	for curr != nil {
		clone.PushBack(curr.value)
		curr = curr.next
	}
	return clone
}

func (l *LinkedList[T]) Values() []T {
	if l.size == 0 {
		return nil
	}
	result := make([]T, l.size)
	curr := l.head
	for i := 0; curr != nil; i++ {
		result[i] = curr.value
		curr = curr.next
	}
	return result
}
