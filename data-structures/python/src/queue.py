"""Circular queue implementation.

Note: This module is named 'queue' which shadows the stdlib queue module.
Import with care or rename to 'circular_queue' if stdlib queue is needed.
"""


class Queue:
    def __init__(self):
        self._data = [None] * 4
        self._head = 0
        self._tail = 0
        self._size = 0
        self._capacity = 4

    def enqueue(self, value):
        if self._size == self._capacity:
            self._grow()
        self._data[self._tail] = value
        self._tail = (self._tail + 1) % self._capacity
        self._size += 1

    def dequeue(self):
        if self._size == 0:
            raise IndexError("dequeue from empty queue")
        value = self._data[self._head]
        self._data[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._size -= 1
        return value

    def front(self):
        if self._size == 0:
            raise IndexError("front from empty queue")
        return self._data[self._head]

    def back(self):
        if self._size == 0:
            raise IndexError("back from empty queue")
        back_index = (self._tail - 1) % self._capacity
        return self._data[back_index]

    def size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def clear(self):
        self._data = [None] * self._capacity
        self._head = 0
        self._tail = 0
        self._size = 0

    def copy(self):
        """Return a shallow copy of the queue."""
        clone = Queue()
        clone._capacity = self._capacity
        clone._data = [None] * self._capacity
        clone._head = self._head
        clone._tail = self._tail
        clone._size = self._size
        for i in range(self._size):
            idx = (self._head + i) % self._capacity
            clone._data[idx] = self._data[idx]
        return clone

    def _grow(self):
        new_capacity = self._capacity * 2
        new_data = [None] * new_capacity
        for i in range(self._size):
            new_data[i] = self._data[(self._head + i) % self._capacity]
        self._data = new_data
        self._head = 0
        self._tail = self._size
        self._capacity = new_capacity

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0
