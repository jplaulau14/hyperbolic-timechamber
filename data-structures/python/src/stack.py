class Stack:
    def __init__(self):
        self._data = []

    def push(self, value):
        self._data.append(value)

    def pop(self):
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def top(self):
        if not self._data:
            raise IndexError("top from empty stack")
        return self._data[-1]

    def size(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0

    def clear(self):
        self._data.clear()

    def copy(self):
        clone = Stack()
        clone._data = self._data.copy()
        return clone

    def __len__(self):
        return len(self._data)

    def __bool__(self):
        return len(self._data) > 0
