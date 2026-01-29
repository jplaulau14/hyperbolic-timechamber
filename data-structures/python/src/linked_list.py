class LinkedList:
    class Node:
        def __init__(self, value):
            self.value = value
            self.next = None

    def __init__(self):
        self._head = None
        self._tail = None
        self._size = 0

    def push_front(self, value):
        node = self.Node(value)
        node.next = self._head
        self._head = node
        if self._tail is None:
            self._tail = node
        self._size += 1

    def push_back(self, value):
        node = self.Node(value)
        if self._tail is None:
            self._head = node
            self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._size += 1

    def pop_front(self):
        if self._head is None:
            raise IndexError("pop_front from empty list")
        value = self._head.value
        self._head = self._head.next
        if self._head is None:
            self._tail = None
        self._size -= 1
        return value

    def pop_back(self):
        if self._head is None:
            raise IndexError("pop_back from empty list")
        value = self._tail.value
        if self._head is self._tail:
            self._head = None
            self._tail = None
        else:
            current = self._head
            while current.next is not self._tail:
                current = current.next
            current.next = None
            self._tail = current
        self._size -= 1
        return value

    def front(self):
        if self._head is None:
            raise IndexError("front from empty list")
        return self._head.value

    def back(self):
        if self._tail is None:
            raise IndexError("back from empty list")
        return self._tail.value

    def at(self, index):
        if not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index >= self._size:
            raise IndexError("index out of range")
        current = self._head
        for _ in range(index):
            current = current.next
        return current.value

    def insert_at(self, index, value):
        if not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index > self._size:
            raise IndexError("index out of range")
        if index == 0:
            self.push_front(value)
        elif index == self._size:
            self.push_back(value)
        else:
            node = self.Node(value)
            current = self._head
            for _ in range(index - 1):
                current = current.next
            node.next = current.next
            current.next = node
            self._size += 1

    def remove_at(self, index):
        if not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index >= self._size:
            raise IndexError("index out of range")
        if index == 0:
            return self.pop_front()
        if index == self._size - 1:
            return self.pop_back()
        current = self._head
        for _ in range(index - 1):
            current = current.next
        value = current.next.value
        current.next = current.next.next
        self._size -= 1
        return value

    def size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def clear(self):
        self._head = None
        self._tail = None
        self._size = 0

    def copy(self):
        clone = LinkedList()
        current = self._head
        while current is not None:
            clone.push_back(current.value)
            current = current.next
        return clone

    def __iter__(self):
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    def __len__(self):
        return self._size
