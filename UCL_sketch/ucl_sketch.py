import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import OrthogonalMatchingPursuit

from Sketching.cm_sketch import CMsketch
from Sketching.bloom_filter import BloomFilter
from UCL_sketch.heavy_filter import heavyFilter


class UCLSketch:
    def __init__(
        self, 
        slot_num: int, 
        width: int, 
        depth: int, 
        bf_width: int, 
        bf_hash: int, 
        KEY_T_SIZE=8,
        decode_mode='ML'
    ):
        self.mode = decode_mode
        self.hTable = heavyFilter(slot_num, KEY_T_SIZE)
        self.cm = CMsketch(width, depth, KEY_T_SIZE)
        self.bf = BloomFilter(bf_width, bf_hash, KEY_T_SIZE)

        self.milestones = []
        self.evictKeys = []
        self.flowKeys = []
        self.cmResult = {}

    def get_keys(self):
        self.milestones.append((len(self.flowKeys), len(self.evictKeys)))
        keys, index = [], []
        s1 = s2 = 0
        for milestone in self.milestones:
            coldline, hotline = milestone[0], milestone[1]
            keys += (self.flowKeys[s1:coldline] + self.evictKeys[s2:hotline])
            index += [i for i in range(coldline+s2, coldline+hotline)]
            s1, s2 = coldline, hotline
        return keys, index
    
    def insert(self, key, val=1):
        evict_or_not, temp_key = self.hTable.insert(key, val)
        
        if evict_or_not != 0:
            exist_or_not = self.bf.getbit(temp_key.key)
            self.cm.insert(temp_key.key, temp_key.val)
            if not exist_or_not:
                self.bf.setbit(temp_key.key)
                if evict_or_not == 1 and temp_key.val > 1:
                    self.evictKeys.append(temp_key.key)
                else:
                    self.flowKeys.append(temp_key.key)
            elif evict_or_not == 1 and temp_key.val > 1:
                if temp_key.key not in self.evictKeys:
                    try:
                        self.flowKeys.remove(temp_key.key)
                        self.evictKeys.append(temp_key.key)
                    except:
                        pass

    def return_cs_components(self, M: int, N: int):
        b = np.zeros(M,)
        keys, index = self.get_keys()
        A_data, A_rows, A_cols = [], [], []

        for i in range(self.cm.depth):
            for j, key in enumerate(keys):
                idx = i * self.cm.width + self.cm.hash(key, i)
                A_data.append(1)
                A_rows.append(idx)
                A_cols.append(j)

            for j in range(self.cm.width):
                b[i * self.cm.width + j] = self.cm.matrix[i][j]
        
        A = csr_matrix((A_data, (A_rows, A_cols)), shape=(M, N))
        return A, b, index

    def solve_equations(self, x=None):
        # assert x and self.mode=='ML', 'results should not be None in Learning version.'
        if self.cmResult != {}:
            return
        M = self.cm.depth * self.cm.width
        keys = self.flowKeys + self.evictKeys
        N = len(keys)

        if self.mode=='OMP':
            A, b, _ = self.return_cs_components(M, N)
            omp = OrthogonalMatchingPursuit()
            x = omp.fit(A.toarray(), b).coef_
            x[x<1] = 1
        elif self.mode=='LSQR':
            A, b, _ = self.return_cs_components(M, N)
            x, *_ = lsqr(A, b)
            x[x<1] = 1

        for i, key in enumerate(keys):
            self.cmResult[key] = x[i]

    def query(self, key, results=None):
        table_ans = self.hTable.query(key)
        
        if self.mode != 'CM':
           self.solve_equations(results)
           exist_or_not = self.bf.getbit(key)

           if exist_or_not:
               try:
                   cm_ans = self.cmResult[key]
               except:
                   cm_ans = 1
           else:
               cm_ans = 0

        else:
            cm_ans = self.cm.query(key)

        return table_ans + cm_ans
    
    def get_current_state(self, return_A=True):
        M = self.cm.depth * self.cm.width
        keys = self.flowKeys + self.evictKeys
        N = len(keys)
        if return_A:
            A, _, index = self.return_cs_components(M, N)
            return A.A, index
        
        b = np.zeros(M,)
        for i in range(self.cm.depth):
            for j in range(self.cm.width):
                b[i * self.cm.width + j] = self.cm.matrix[i][j]
    
        return b.reshape(1, self.cm.depth, self.cm.width)
    
    def refresh(self):
        self.cmResult = {}
        self.milestones = []
        self.evictKeys = []
        self.flowKeys = []
        self.cmResult = {}
    
    def get_memory_usage(self):
        ht_size = self.hTable.get_memory_usage()
        bf_size = self.bf.get_memory_usage()
        cm_size = self.cm.get_memory_usage()
        
        print("----- Memory Usage -----")
        print(f"Hash Table Size(Byte): {ht_size} ({ht_size / 1024:.2f} KB)")
        print(f"Bloom Filter Size(Byte): {bf_size} ({bf_size / 1024:.2f} KB)")
        print(f"CM Sketch Size(Byte): {cm_size} ({cm_size / 1024:.2f} KB)")
        print(f"Total Memory(MB): {(ht_size + cm_size + bf_size) / 1024:.2f} KB")
        print("------------------------")
        
        return ht_size + cm_size + bf_size
