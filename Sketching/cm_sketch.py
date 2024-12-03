import numpy as np

from Utils.common import calNextPrime
from Sketching.hash_function import GenHashSeed, AwareHash


class CMsketch:

    def __init__(self, width: int, depth: int, KEY_T_SIZE=13):
        self.key_size = KEY_T_SIZE
        self.depth, self.width = depth, calNextPrime(width)
        self.h = [GenHashSeed(i) for i in range(depth)]
        self.s = [GenHashSeed(i) for i in range(depth)]
        self.n = [GenHashSeed(i) for i in range(depth)]
        self.matrix = np.zeros((self.depth, self.width), dtype=int)

    def hash(self, key, col):
        return AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col]) % self.width
    
    def insert(self, key, val=1):
        for i in range(self.depth):
            pos = self.hash(key, i)
            assert self.matrix[i, pos] != np.iinfo(self.matrix.dtype).max
            self.matrix[i, pos] += val
        return 0
    
    def query(self, key):
        result = np.iinfo(self.matrix.dtype).max
        for i in range(self.depth):
            pos = self.hash(key, i)
            result = min(result, self.matrix[i, pos])
        return result
    
    def get_memory_usage(self):
        return self.depth * self.width * self.matrix.itemsize
