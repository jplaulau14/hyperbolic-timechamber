class HashMap:
    def __init__(self, capacity=16):
        self._capacity = capacity
        self._size = 0
        self._buckets = [[] for _ in range(capacity)]

    def _bucket_index(self, key):
        return hash(key) % self._capacity

    def _rehash(self):
        old_buckets = self._buckets
        self._capacity *= 2
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)

    def put(self, key, value):
        index = self._bucket_index(key)
        bucket = self._buckets[index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1
        if self._size / self._capacity > 0.75:
            self._rehash()

    def get(self, key):
        index = self._bucket_index(key)
        for k, v in self._buckets[index]:
            if k == key:
                return v
        raise KeyError(key)

    def get_or(self, key, default):
        try:
            return self.get(key)
        except KeyError:
            return default

    def remove(self, key):
        index = self._bucket_index(key)
        bucket = self._buckets[index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self._size -= 1
                return v
        return None

    def contains(self, key):
        index = self._bucket_index(key)
        for k, v in self._buckets[index]:
            if k == key:
                return True
        return False

    def size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def clear(self):
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0

    def keys(self):
        result = []
        for bucket in self._buckets:
            for k, v in bucket:
                result.append(k)
        return result

    def values(self):
        result = []
        for bucket in self._buckets:
            for k, v in bucket:
                result.append(v)
        return result

    def copy(self):
        """Create a copy of this HashMap.

        Note: This performs a shallow copy of values. If values are mutable
        objects, modifications to them will be visible in both the original
        and copied HashMap.
        """
        new_map = HashMap(self._capacity)
        for bucket in self._buckets:
            for k, v in bucket:
                new_map.put(k, v)
        return new_map

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self.contains(key)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        if not self.contains(key):
            raise KeyError(key)
        self.remove(key)

    def __iter__(self):
        for bucket in self._buckets:
            for k, v in bucket:
                yield k
