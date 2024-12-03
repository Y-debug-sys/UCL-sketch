import numpy as np

from Utils.common import calNextPrime
from Sketching.hash_function import GenHashSeed, AwareHash


class Csketch:

    def __init__(self, width: int, depth: int, KEY_T_SIZE=13):
        self.key_size = KEY_T_SIZE
        self.depth, self.width = depth, calNextPrime(width)
        self.h = [GenHashSeed(i) for i in range(depth)]
        self.s = [GenHashSeed(i) for i in range(depth)]
        self.n = [GenHashSeed(i) for i in range(depth)]
        self.i = [GenHashSeed(i) for i in range(depth)]
        self.j = [GenHashSeed(i) for i in range(depth)]
        self.k = [GenHashSeed(i) for i in range(depth)]
        self.matrix = np.zeros((self.depth, self.width), dtype=int)

    def hash(self, key, col):
        hash_value1 = AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col])
        hash_value2 = AwareHash(key, self.key_size, self.i[col], self.j[col], self.k[col])
        return hash_value1 % self.width, 1 - 2 * (hash_value2 % 2)
    
    def insert(self, key, val=1):
        for i in range(self.depth):
            pos, sign = self.hash(key, i)
            assert self.matrix[i, pos] != np.iinfo(self.matrix.dtype).max
            self.matrix[i, pos] += (val * sign)
        return 0
    
    def query(self, key):
        results = []
        for i in range(self.depth):
            pos, sign = self.hash(key, i)
            results.append(self.matrix[i, pos] * sign)
        return abs(np.median(results))
    
    def get_memory_usage(self):
        return self.depth * self.width * self.matrix.itemsize
