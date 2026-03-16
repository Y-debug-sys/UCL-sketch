import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import OrthogonalMatchingPursuit

from Sketching.cm_sketch import CMsketch
from Sketching.bloom_filter import BloomFilter
from UCL_sketch.heavy_filter import heavyFilter


class UCLSketch:
    """
    UCL-Sketch: Unsupervised Compressed Learning Sketch for frequency estimation in data streams.
    This class combines a heavy hitter filter, Count-Min sketch, and Bloom filter to achieve
    accurate frequency estimation without requiring labeled training data.
    
    The algorithm maintains two sets of keys: flow keys (light hitters) and evict keys (heavy hitters)
    to optimize estimation accuracy through a combination of different techniques for each type of key.
    """
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
        """
        Initialize the UCLSketch with specified parameters
        
        Args:
            slot_num: Number of slots in the heavy hitter filter
            width: Width of the Count-Min sketch (number of counters per row)
            depth: Depth of the Count-Min sketch (number of hash functions/rows)
            bf_width: Width of the Bloom filter
            bf_hash: Number of hash functions for the Bloom filter
            KEY_T_SIZE: Size of each key in bytes
            decode_mode: Decoding mode ('ML', 'OMP', 'LSQR', 'CM')
        """
        self.mode = decode_mode  # Decoding algorithm to use
        self.hTable = heavyFilter(slot_num, KEY_T_SIZE)  # Heavy hitter detection filter
        self.cm = CMsketch(width, depth, KEY_T_SIZE)     # Count-Min sketch for general counting
        self.bf = BloomFilter(bf_width, bf_hash, KEY_T_SIZE)  # Bloom filter to track seen items

        # Track insertion milestones and different types of keys
        self.milestones = []      # Record positions of key groups
        self.evictKeys = []       # Keys identified as heavy hitters
        self.flowKeys = []        # Keys considered as light/heavy flows
        self.cmResult = {}        # Cache for decoded Count-Min results

    def get_keys(self):
        """
        Retrieve all tracked keys (both flow and evicted) along with their indices
        
        Returns:
            Tuple of (keys, index) where keys is the list of all tracked keys
            and index indicates their positions in the combined list
        """
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
        """
        Insert a key-value pair into the sketch structure
        
        Args:
            key: The item/key to insert
            val: The value/frequency increment to add (default 1)
        """
        # Try to insert into heavy hitter filter
        evict_or_not, temp_key = self.hTable.insert(key, val)
        
        if evict_or_not != 0:
            # Key was processed by heavy filter, now handle in CM sketch and Bloom filter
            exist_or_not = self.bf.getbit(temp_key.key)
            self.cm.insert(temp_key.key, temp_key.val)
            if not exist_or_not:
                # Key hasn't been seen before
                self.bf.setbit(temp_key.key)
                if evict_or_not == 1 and temp_key.val > 1:
                    # Heavy hitter, add to evictKeys
                    self.evictKeys.append(temp_key.key)
                else:
                    # Light hitter, add to flowKeys
                    self.flowKeys.append(temp_key.key)
            elif evict_or_not == 1 and temp_key.val > 1:
                # Key became a heavy hitter after this insertion
                if temp_key.key not in self.evictKeys:
                    try:
                        # Move from flowKeys to evictKeys
                        self.flowKeys.remove(temp_key.key)
                        self.evictKeys.append(temp_key.key)
                    except:
                        pass

    def return_cs_components(self, M: int, N: int):
        """
        Return components for compressed sensing formulation (A matrix, b vector)
        Used for solving the underdetermined system to recover actual frequencies
        
        Args:
            M: Number of measurements (depth * width of CM sketch)
            N: Number of variables (number of tracked keys)
            
        Returns:
            Tuple of (A matrix, b vector, index list) for compressed sensing
        """
        b = np.zeros(M,)  # Right-hand side vector
        keys, index = self.get_keys()  # Get all tracked keys
        A_data, A_rows, A_cols = [], [], []  # Components for sparse matrix

        # Build constraint matrix A and observation vector b
        for i in range(self.cm.depth):
            for j, key in enumerate(keys):
                # Map each key to positions in the CM sketch using hash functions
                idx = i * self.cm.width + self.cm.hash(key, i)
                A_data.append(1)      # Value of 1 for presence
                A_rows.append(idx)    # Row index in A matrix
                A_cols.append(j)      # Column index in A matrix

            # Fill observation vector b with actual CM sketch counter values
            for j in range(self.cm.width):
                b[i * self.cm.width + j] = self.cm.matrix[i][j]
        
        # Create sparse matrix A from coordinates
        A = csr_matrix((A_data, (A_rows, A_cols)), shape=(M, N))
        return A, b, index

    def solve_equations(self, x=None):
        """
        Solve the compressed sensing problem to estimate actual frequencies from CM sketch
        
        Uses different algorithms based on the mode: OMP, LSQR, or keeps raw CM values
        Results are cached in self.cmResult to avoid recomputation
        """
        if self.cmResult != {}:
            # Already solved, skip computation
            return
        M = self.cm.depth * self.cm.width  # Total number of counters in CM sketch
        keys = self.flowKeys + self.evictKeys  # All tracked keys
        N = len(keys)  # Number of variables to solve for

        if self.mode=='OMP':
            # Orthogonal Matching Pursuit for sparse recovery
            A, b, _ = self.return_cs_components(M, N)
            omp = OrthogonalMatchingPursuit()
            x = omp.fit(A.toarray(), b).coef_
            x[x<0] = 0  # Ensure non-negative frequencies
        elif self.mode=='LSQR':
            # Least Squares QR algorithm for solving linear systems
            A, b, _ = self.return_cs_components(M, N)
            x, *_ = lsqr(A, b)
            x[x<0] = 0  # Ensure non-negative frequencies

        # Store results in dictionary for quick lookup
        for i, key in enumerate(keys):
            self.cmResult[key] = x[i]

    def query(self, key, results=None):
        """
        Query the frequency of a specific key
        
        Args:
            key: The key to query
            results: Pre-computed results to use (optional)
            
        Returns:
            Estimated frequency of the key
        """
        # Get result from heavy hitter table
        table_ans = self.hTable.query(key)
        
        if self.mode != 'CM':
           # Use compressed sensing decoding if not using raw CM
           self.solve_equations(results)
           exist_or_not = self.bf.getbit(key)

           if exist_or_not:
               # Key was seen before, use decoded result
               try:
                   cm_ans = self.cmResult[key]
               except:
                   # Fallback if key not in results
                   cm_ans = 1
           else:
               # Key never seen, return 0
               cm_ans = 0

        else:
            # Use raw CM sketch result
            cm_ans = self.cm.query(key)

        # Return sum of heavy hitter result and CM sketch result
        return table_ans + cm_ans
    
    def get_current_state(self, return_A=True):
        """
        Get the current state of the sketch for training purposes
        
        Args:
            return_A: Whether to return the constraint matrix (True) or just the CM values (False)
            
        Returns:
            Either (constraint matrix, index) or CM sketch values reshaped
        """
        M = self.cm.depth * self.cm.width
        keys = self.flowKeys + self.evictKeys
        N = len(keys)
        if return_A:
            A, _, index = self.return_cs_components(M, N)
            return A.A, index  # Return dense representation of A matrix and index
        
        # Just return the CM sketch values reshaped
        b = np.zeros(M,)
        for i in range(self.cm.depth):
            for j in range(self.cm.width):
                b[i * self.cm.width + j] = self.cm.matrix[i][j]
    
        return b.reshape(1, self.cm.depth, self.cm.width)
    
    def refresh(self):
        """
        Reset the sketch to its initial state, clearing all stored information
        """
        self.cmResult = {}
        self.milestones = []
        self.evictKeys = []
        self.flowKeys = []
        self.cmResult = {}
    
    def get_memory_usage(self):
        """
        Calculate and print the memory usage of each component
        
        Returns:
            Total memory usage in bytes
        """
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
