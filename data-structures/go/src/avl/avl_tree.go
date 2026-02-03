package avl

import (
	"cmp"
	"errors"
)

var ErrEmptyTree = errors.New("AVLTree: tree is empty")

type node[T cmp.Ordered] struct {
	value  T
	left   *node[T]
	right  *node[T]
	height int
}

type AVLTree[T cmp.Ordered] struct {
	root *node[T]
	size int
}

func New[T cmp.Ordered]() *AVLTree[T] {
	return &AVLTree[T]{}
}

func (t *AVLTree[T]) Insert(value T) {
	t.root = insertNode(t.root, value, &t.size)
}

func (t *AVLTree[T]) Remove(value T) {
	t.root = removeNode(t.root, value, &t.size)
}

func (t *AVLTree[T]) Contains(value T) bool {
	return findNode(t.root, value) != nil
}

func (t *AVLTree[T]) Min() (T, error) {
	var zero T
	if t.root == nil {
		return zero, ErrEmptyTree
	}
	return findMin(t.root).value, nil
}

func (t *AVLTree[T]) Max() (T, error) {
	var zero T
	if t.root == nil {
		return zero, ErrEmptyTree
	}
	return findMax(t.root).value, nil
}

func (t *AVLTree[T]) Size() int {
	return t.size
}

func (t *AVLTree[T]) IsEmpty() bool {
	return t.size == 0
}

func (t *AVLTree[T]) Clear() {
	t.root = nil
	t.size = 0
}

func (t *AVLTree[T]) Height() int {
	return getHeight(t.root)
}

func (t *AVLTree[T]) InOrder() []T {
	result := make([]T, 0, t.size)
	inOrderTraverse(t.root, &result)
	return result
}

func (t *AVLTree[T]) PreOrder() []T {
	result := make([]T, 0, t.size)
	preOrderTraverse(t.root, &result)
	return result
}

func (t *AVLTree[T]) PostOrder() []T {
	result := make([]T, 0, t.size)
	postOrderTraverse(t.root, &result)
	return result
}

func (t *AVLTree[T]) Clone() *AVLTree[T] {
	clone := New[T]()
	cloneTree(t.root, clone)
	return clone
}

func (t *AVLTree[T]) IsBalanced() bool {
	return checkBalance(t.root)
}

func getHeight[T cmp.Ordered](n *node[T]) int {
	if n == nil {
		return 0
	}
	return n.height
}

func updateHeight[T cmp.Ordered](n *node[T]) {
	n.height = 1 + max(getHeight(n.left), getHeight(n.right))
}

func balanceFactor[T cmp.Ordered](n *node[T]) int {
	if n == nil {
		return 0
	}
	return getHeight(n.left) - getHeight(n.right)
}

func rotateRight[T cmp.Ordered](y *node[T]) *node[T] {
	x := y.left
	b := x.right
	x.right = y
	y.left = b
	updateHeight(y)
	updateHeight(x)
	return x
}

func rotateLeft[T cmp.Ordered](x *node[T]) *node[T] {
	y := x.right
	b := y.left
	y.left = x
	x.right = b
	updateHeight(x)
	updateHeight(y)
	return y
}

func rebalance[T cmp.Ordered](n *node[T]) *node[T] {
	updateHeight(n)
	balance := balanceFactor(n)

	if balance > 1 {
		if balanceFactor(n.left) < 0 {
			n.left = rotateLeft(n.left)
		}
		return rotateRight(n)
	}

	if balance < -1 {
		if balanceFactor(n.right) > 0 {
			n.right = rotateRight(n.right)
		}
		return rotateLeft(n)
	}

	return n
}

func insertNode[T cmp.Ordered](n *node[T], value T, size *int) *node[T] {
	if n == nil {
		*size++
		return &node[T]{value: value, height: 1}
	}
	if value < n.value {
		n.left = insertNode(n.left, value, size)
	} else if value > n.value {
		n.right = insertNode(n.right, value, size)
	} else {
		return n
	}
	return rebalance(n)
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
		*size--
		if n.left == nil {
			return n.right
		}
		if n.right == nil {
			return n.left
		}
		*size++
		successor := findMin(n.right)
		n.value = successor.value
		n.right = removeNode(n.right, successor.value, size)
	}
	return rebalance(n)
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

func cloneTree[T cmp.Ordered](n *node[T], clone *AVLTree[T]) {
	if n == nil {
		return
	}
	clone.Insert(n.value)
	cloneTree(n.left, clone)
	cloneTree(n.right, clone)
}

func checkBalance[T cmp.Ordered](n *node[T]) bool {
	if n == nil {
		return true
	}
	balance := balanceFactor(n)
	if balance < -1 || balance > 1 {
		return false
	}
	return checkBalance(n.left) && checkBalance(n.right)
}
