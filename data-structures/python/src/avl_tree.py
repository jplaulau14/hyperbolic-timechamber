from typing import TypeVar, Generic, List, Iterator, Optional

T = TypeVar('T')


class AVLTree(Generic[T]):
    class Node:
        def __init__(self, value: T) -> None:
            self.value: T = value
            self.left: Optional['AVLTree.Node'] = None
            self.right: Optional['AVLTree.Node'] = None
            self.height: int = 1

    def __init__(self) -> None:
        self._root: Optional[AVLTree.Node] = None
        self._size: int = 0

    def _get_height(self, node: Optional[Node]) -> int:
        if node is None:
            return 0
        return node.height

    def _update_height(self, node: Node) -> None:
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

    def _get_balance(self, node: Optional[Node]) -> int:
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _right_rotate(self, y: Node) -> Node:
        x = y.left
        assert x is not None
        t2 = x.right

        x.right = y
        y.left = t2

        self._update_height(y)
        self._update_height(x)

        return x

    def _left_rotate(self, x: Node) -> Node:
        y = x.right
        assert y is not None
        t2 = y.left

        y.left = x
        x.right = t2

        self._update_height(x)
        self._update_height(y)

        return y

    def _rebalance(self, node: Node) -> Node:
        self._update_height(node)
        balance = self._get_balance(node)

        if balance > 1:
            if self._get_balance(node.left) < 0:
                assert node.left is not None
                node.left = self._left_rotate(node.left)
            return self._right_rotate(node)

        if balance < -1:
            if self._get_balance(node.right) > 0:
                assert node.right is not None
                node.right = self._right_rotate(node.right)
            return self._left_rotate(node)

        return node

    def _insert(self, node: Optional[Node], value: T) -> Node:
        if node is None:
            self._size += 1
            return AVLTree.Node(value)

        if value < node.value:
            node.left = self._insert(node.left, value)
        elif value > node.value:
            node.right = self._insert(node.right, value)
        else:
            return node

        return self._rebalance(node)

    def insert(self, value: T) -> None:
        self._root = self._insert(self._root, value)

    def _find_min_node(self, node: Node) -> Node:
        while node.left is not None:
            node = node.left
        return node

    def _remove(self, node: Optional[Node], value: T) -> Optional[Node]:
        if node is None:
            return None

        if value < node.value:
            node.left = self._remove(node.left, value)
        elif value > node.value:
            node.right = self._remove(node.right, value)
        else:
            if node.left is None:
                self._size -= 1
                return node.right
            elif node.right is None:
                self._size -= 1
                return node.left
            else:
                successor = self._find_min_node(node.right)
                node.value = successor.value
                node.right = self._remove(node.right, successor.value)

        return self._rebalance(node)

    def remove(self, value: T) -> None:
        self._root = self._remove(self._root, value)

    def contains(self, value: T) -> bool:
        node = self._root
        while node is not None:
            if value < node.value:
                node = node.left
            elif value > node.value:
                node = node.right
            else:
                return True
        return False

    def min(self) -> T:
        if self._root is None:
            raise ValueError("min from empty tree")
        node = self._root
        while node.left is not None:
            node = node.left
        return node.value

    def max(self) -> T:
        if self._root is None:
            raise ValueError("max from empty tree")
        node = self._root
        while node.right is not None:
            node = node.right
        return node.value

    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def clear(self) -> None:
        self._root = None
        self._size = 0

    def height(self) -> int:
        return self._get_height(self._root)

    def in_order(self) -> List[T]:
        result: List[T] = []
        stack: List[AVLTree.Node] = []
        node = self._root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            result.append(node.value)
            node = node.right
        return result

    def pre_order(self) -> List[T]:
        result: List[T] = []
        if self._root is None:
            return result
        stack: List[AVLTree.Node] = [self._root]
        while stack:
            node = stack.pop()
            result.append(node.value)
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)
        return result

    def post_order(self) -> List[T]:
        result: List[T] = []
        if self._root is None:
            return result
        stack: List[AVLTree.Node] = [self._root]
        while stack:
            node = stack.pop()
            result.append(node.value)
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
        result.reverse()
        return result

    def copy(self) -> 'AVLTree[T]':
        clone: AVLTree[T] = AVLTree()
        for value in self.pre_order():
            clone.insert(value)
        return clone

    def _is_balanced(self, node: Optional[Node]) -> bool:
        if node is None:
            return True
        balance = self._get_balance(node)
        if abs(balance) > 1:
            return False
        return self._is_balanced(node.left) and self._is_balanced(node.right)

    def is_balanced(self) -> bool:
        return self._is_balanced(self._root)

    def __len__(self) -> int:
        return self._size

    def __contains__(self, value: T) -> bool:
        return self.contains(value)

    def __iter__(self) -> Iterator[T]:
        return iter(self.in_order())

    def __repr__(self) -> str:
        return f"AVLTree({self.in_order()})"

    def __str__(self) -> str:
        return f"AVLTree(size={self._size}, height={self.height()})"
