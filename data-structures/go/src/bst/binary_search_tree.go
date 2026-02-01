package bst

import (
	"cmp"
	"errors"
)

var ErrEmptyTree = errors.New("BinarySearchTree: tree is empty")

type node[T cmp.Ordered] struct {
	value T
	left  *node[T]
	right *node[T]
}

type BinarySearchTree[T cmp.Ordered] struct {
	root *node[T]
	size int
}

func New[T cmp.Ordered]() *BinarySearchTree[T] {
	return &BinarySearchTree[T]{}
}

func (t *BinarySearchTree[T]) Insert(value T) {
	t.root = insertNode(t.root, value, &t.size)
}

func (t *BinarySearchTree[T]) Remove(value T) {
	t.root = removeNode(t.root, value, &t.size)
}

func (t *BinarySearchTree[T]) Contains(value T) bool {
	return findNode(t.root, value) != nil
}

func (t *BinarySearchTree[T]) Min() (T, error) {
	var zero T
	if t.root == nil {
		return zero, ErrEmptyTree
	}
	return findMin(t.root).value, nil
}

func (t *BinarySearchTree[T]) Max() (T, error) {
	var zero T
	if t.root == nil {
		return zero, ErrEmptyTree
	}
	return findMax(t.root).value, nil
}

func (t *BinarySearchTree[T]) Size() int {
	return t.size
}

func (t *BinarySearchTree[T]) IsEmpty() bool {
	return t.size == 0
}

func (t *BinarySearchTree[T]) Clear() {
	t.root = nil
	t.size = 0
}

func (t *BinarySearchTree[T]) InOrder() []T {
	result := make([]T, 0, t.size)
	inOrderTraverse(t.root, &result)
	return result
}

func (t *BinarySearchTree[T]) PreOrder() []T {
	result := make([]T, 0, t.size)
	preOrderTraverse(t.root, &result)
	return result
}

func (t *BinarySearchTree[T]) PostOrder() []T {
	result := make([]T, 0, t.size)
	postOrderTraverse(t.root, &result)
	return result
}

func (t *BinarySearchTree[T]) Clone() *BinarySearchTree[T] {
	clone := New[T]()
	cloneTree(t.root, clone)
	return clone
}

func insertNode[T cmp.Ordered](n *node[T], value T, size *int) *node[T] {
	if n == nil {
		*size++
		return &node[T]{value: value}
	}
	if value < n.value {
		n.left = insertNode(n.left, value, size)
	} else if value > n.value {
		n.right = insertNode(n.right, value, size)
	}
	return n
}

func removeNode[T cmp.Ordered](n *node[T], value T, size *int) *node[T] {
	if n == nil {
		return nil
	}

	if value < n.value {
		n.left = removeNode(n.left, value, size)
	} else if value > n.value {
		n.right = removeNode(n.right, value, size)
	} else {
		if n.left == nil {
			*size--
			return n.right
		}
		if n.right == nil {
			*size--
			return n.left
		}
		successor := findMin(n.right)
		n.value = successor.value
		n.right = removeNode(n.right, successor.value, size)
	}
	return n
}

func findNode[T cmp.Ordered](n *node[T], value T) *node[T] {
	for n != nil {
		if value < n.value {
			n = n.left
		} else if value > n.value {
			n = n.right
		} else {
			return n
		}
	}
	return nil
}

func findMin[T cmp.Ordered](n *node[T]) *node[T] {
	for n.left != nil {
		n = n.left
	}
	return n
}

func findMax[T cmp.Ordered](n *node[T]) *node[T] {
	for n.right != nil {
		n = n.right
	}
	return n
}

func inOrderTraverse[T cmp.Ordered](n *node[T], result *[]T) {
	if n == nil {
		return
	}
	inOrderTraverse(n.left, result)
	*result = append(*result, n.value)
	inOrderTraverse(n.right, result)
}

func preOrderTraverse[T cmp.Ordered](n *node[T], result *[]T) {
	if n == nil {
		return
	}
	*result = append(*result, n.value)
	preOrderTraverse(n.left, result)
	preOrderTraverse(n.right, result)
}

func postOrderTraverse[T cmp.Ordered](n *node[T], result *[]T) {
	if n == nil {
		return
	}
	postOrderTraverse(n.left, result)
	postOrderTraverse(n.right, result)
	*result = append(*result, n.value)
}

func cloneTree[T cmp.Ordered](n *node[T], clone *BinarySearchTree[T]) {
	if n == nil {
		return
	}
	clone.Insert(n.value)
	cloneTree(n.left, clone)
	cloneTree(n.right, clone)
}
