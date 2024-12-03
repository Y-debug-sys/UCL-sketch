from Utils.common import calNextPrime
from Sketching.cs_sketch import Csketch
from Sketching.hash_function import GenHashSeed, AwareHash


class UnivMon:

    def __init__(self, log_n, width, depth, KEY_T_SIZE=13):
        self.key_size = KEY_T_SIZE
        self.depth = depth
        self.width = calNextPrime(width)
        self.logn = log_n
        self.h = [GenHashSeed(i) for i in range(log_n - 1)]
        self.s = [GenHashSeed(i) for i in range(log_n - 1)]
        self.n = [GenHashSeed(i) for i in range(log_n - 1)]
        self.sketch = [Csketch(width, depth, KEY_T_SIZE)]
        for i in range(1, log_n):
            width = max(1, width // 2)
            self.sketch.append(Csketch(width, depth, KEY_T_SIZE))
        self.flows = [None] * log_n

    def __del__(self):
        del self.sketch

    def hash(self, key, i):
        return AwareHash(key, self.key_size, self.h[i], self.s[i], self.n[i]) & 1

    def get_hash(self, flowkey, layer):
        if layer == 0:
            return 1
        return self.hash(flowkey, layer-1)

    def insert(self, flowkey, val=1):
        for i in range(self.logn):
            if self.get_hash(flowkey, i):
                self.sketch[i].insert(flowkey, val)
            else:
                break

    def query(self, flowkey):
        level = 0
        for level in range(self.logn):
            if not self.get_hash(flowkey, level):
                break
        level -= 1
        ret = self.sketch[level].query(flowkey)
        for i in range(level - 1, -1, -1):
            ret = 2 * ret - self.sketch[i].query(flowkey)
        return ret
    
    def get_memory_usage(self):
        total = 0
        for sketch in self.sketch:
            total += sketch.get_memory_usage()
        return total

    def clear(self):
        for sketch in self.sketch:
            sketch.matrix.fill(0)
