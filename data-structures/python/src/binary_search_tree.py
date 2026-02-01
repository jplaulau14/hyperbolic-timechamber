from typing import TypeVar, Generic, List, Iterator, Optional

T = TypeVar('T')


class BinarySearchTree(Generic[T]):
    class Node:
        def __init__(self, value: T) -> None:
            self.value: T = value
            self.left: Optional['BinarySearchTree.Node'] = None
            self.right: Optional['BinarySearchTree.Node'] = None

    def __init__(self) -> None:
        self._root: Optional[BinarySearchTree.Node] = None
        self._size: int = 0

    def insert(self, value: T) -> None:
        if self._root is None:
            self._root = BinarySearchTree.Node(value)
            self._size += 1
            return

        node = self._root
        while True:
            if value < node.value:
                if node.left is None:
                    node.left = BinarySearchTree.Node(value)
                    self._size += 1
                    return
                node = node.left
            elif value > node.value:
                if node.right is None:
                    node.right = BinarySearchTree.Node(value)
                    self._size += 1
                    return
                node = node.right
            else:
                return

    def remove(self, value: T) -> None:
        parent: Optional[BinarySearchTree.Node] = None
        node = self._root
        is_left_child = False

        while node is not None and node.value != value:
            parent = node
            if value < node.value:
                node = node.left
                is_left_child = True
            else:
                node = node.right
                is_left_child = False

        if node is None:
            return

        if node.left is None:
            replacement = node.right
        elif node.right is None:
            replacement = node.left
        else:
            successor_parent = node
            successor = node.right
            while successor.left is not None:
                successor_parent = successor
                successor = successor.left
            node.value = successor.value
            if successor_parent == node:
                successor_parent.right = successor.right
            else:
                successor_parent.left = successor.right
            self._size -= 1
            return

        if parent is None:
            self._root = replacement
        elif is_left_child:
            parent.left = replacement
        else:
            parent.right = replacement
        self._size -= 1

    def contains(self, value: T) -> bool:
        return self._find_node(self._root, value) is not None

    def min(self) -> T:
        if self._root is None:
            raise ValueError("min from empty tree")
        return self._find_min(self._root).value

    def max(self) -> T:
        if self._root is None:
            raise ValueError("max from empty tree")
        return self._find_max(self._root).value

    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def clear(self) -> None:
        self._root = None
        self._size = 0

    def in_order(self) -> List[T]:
        result: List[T] = []
        stack: List[BinarySearchTree.Node] = []
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
        stack: List[BinarySearchTree.Node] = [self._root]
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
        stack: List[BinarySearchTree.Node] = [self._root]
        while stack:
            node = stack.pop()
            result.append(node.value)
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
        result.reverse()
        return result

    def copy(self) -> 'BinarySearchTree[T]':
        clone: BinarySearchTree[T] = BinarySearchTree()
        for value in self.pre_order():
            clone.insert(value)
        return clone

    def _find_node(self, node: Optional[Node], value: T) -> Optional[Node]:
        while node is not None:
            if value < node.value:
                node = node.left
            elif value > node.value:
                node = node.right
            else:
                return node
        return None

    def _find_min(self, node: Node) -> Node:
        while node.left is not None:
            node = node.left
        return node

    def _find_max(self, node: Node) -> Node:
        while node.right is not None:
            node = node.right
        return node

    def __len__(self) -> int:
        return self._size

    def __contains__(self, value: T) -> bool:
        return self.contains(value)

    def __iter__(self) -> Iterator[T]:
        return iter(self.in_order())

    def __repr__(self) -> str:
        return f"BinarySearchTree({self.in_order()})"

    def __str__(self) -> str:
        return f"BinarySearchTree(size={self._size})"
