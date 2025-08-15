import numpy as np

from Utils.common import calNextPrime
from Sketching.hash_function import AwareHash, GenHashSeed

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from Sketching.bloom_filter import BloomFilter


class bloom_sketch:

    def __init__(self, n_length, n_hash, key_size):
        self.hash_num = n_hash
        self.key_size = key_size
        self.width = calNextPrime(n_length)
        self.h = [GenHashSeed(i) for i in range(n_hash)]
        self.s = [GenHashSeed(i) for i in range(n_hash)]
        self.n = [GenHashSeed(i) for i in range(n_hash)]
        self.vector = np.zeros((self.width ), dtype=int)
    
    def hash(self, key, col):
        return AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col]) % self.width
    
    def insert(self, key, val=1):
        for i in range(self.hash_num):
            pos = self.hash(key, i)
            assert self.vector[pos] != np.iinfo(self.vector.dtype).max
            self.vector[pos] += val
        return 0
    
    def get_memory_usage(self):
        return self.width * self.vector.itemsize


class PRSketch:

    def __init__(
        self,
        width: int, 
        depth: int, 
        bf_width: int, 
        bf_hash: int, 
        KEY_T_SIZE=8
    ):
        self.flowKeys = []
        self.sketchResult = {}
        self.sketch = bloom_sketch(width * depth, depth, KEY_T_SIZE)
        self.bf = BloomFilter(bf_width, bf_hash, KEY_T_SIZE)

    def insert(self, key, val=1):
        exist_or_not = self.bf.getbit(key)
        self.sketch.insert(key, val)
        if not exist_or_not:
            self.bf.setbit(key)
            self.flowKeys.append(key)

    def return_cs_components(self, M: int, N: int):
        A_data, A_rows, A_cols = [], [], []

        for i in range(self.sketch.hash_num):
            for j, key in enumerate(self.flowKeys):
                idx = self.sketch.hash(key, i)
                A_data.append(1)
                A_rows.append(idx)
                A_cols.append(j)
        
        A = csr_matrix((A_data, (A_rows, A_cols)), shape=(M, N))
        return A, self.sketch.vector

    def solve_equations(self):
        if self.sketchResult != {}:
            return
        
        M = self.sketch.width
        N = len(self.flowKeys)

        A, b = self.return_cs_components(M, N)
        x, i, *_ = lsmr(A, b)
        x[x<1] = 1
        x = np.ceil(np.abs(x)).astype(np.int32)

        for i, key in enumerate(self.flowKeys):
            self.sketchResult[key] = x[i]

    def query(self, key):
        self.solve_equations()
        exist_or_not = self.bf.getbit(key)

        if exist_or_not:
            try:
                ans = self.sketchResult[key]
            except:
                ans = 1
        else:
            ans = 0

        return ans
    
    def get_memory_usage(self):
        bf_size = self.bf.get_memory_usage()
        sketch_size = self.sketch.get_memory_usage()        
        return sketch_size + bf_size
    
    def test_linear(self, ground_truth):

        M = self.sketch.width
        N = len(self.flowKeys)

        b = self.sketch.vector
        x = np.zeros(N,)
        A_data, A_rows, A_cols = [], [], []

        for i in range(self.sketch.hash_num):
            for j, key in enumerate(self.flowKeys):
                idx = self.sketch.hash(key, i)
                A_data.append(1)
                A_rows.append(idx)
                A_cols.append(j)
                x[j] = ground_truth[key]
        
        A = csr_matrix((A_data, (A_rows, A_cols)), shape=(M, N)).A

        a = A @ x
        for i in range(M):
            print(a[i], b[i])
