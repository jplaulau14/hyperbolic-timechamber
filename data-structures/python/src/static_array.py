class StaticArray:
    def __init__(self, size, default=None):
        if not isinstance(size, int) or size < 0:
            raise ValueError("size must be a non-negative integer")
        self._size = size
        self._data = [default] * size

    def at(self, index):
        if index < 0 or index >= self._size:
            raise IndexError("StaticArray.at: index out of range")
        return self._data[index]

    def set_at(self, index, value):
        if index < 0 or index >= self._size:
            raise IndexError("StaticArray.at: index out of range")
        self._data[index] = value

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def front(self):
        return self._data[0]

    def back(self):
        return self._data[-1]

    def data(self):
        if self._size == 0:
            return None
        return self._data

    def size(self):
        return self._size

    def empty(self):
        return self._size == 0

    def fill(self, value):
        for i in range(self._size):
            self._data[i] = value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return self._size
