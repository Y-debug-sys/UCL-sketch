import numpy as np

from Utils.common import calNextPrime
from Sketching.hash_function import GenHashSeed, AwareHash


class CMsketch:
    """
    Count-Min Sketch implementation - a probabilistic data structure for estimating
    the frequency of events in a data stream using sub-linear space.
    It provides approximate counts with guarantees on the error bounds.
    """

    def __init__(self, width: int, depth: int, KEY_T_SIZE=13):
        """
        Initialize the Count-Min sketch with specified dimensions.
        
        Args:
            width: Width of the sketch matrix (number of counters per row)
            depth: Depth of the sketch matrix (number of hash functions/rows)
            KEY_T_SIZE: Size of keys in bytes
        """
        self.key_size = KEY_T_SIZE
        # Use next prime to reduce hash collisions
        self.depth, self.width = depth, calNextPrime(width)
        # Generate hash function parameters for each row
        self.h = [GenHashSeed(i) for i in range(depth)]  # Hash function parameter h
        self.s = [GenHashSeed(i) for i in range(depth)]  # Hash function parameter s
        self.n = [GenHashSeed(i) for i in range(depth)]  # Hash function parameter n
        # Initialize the counting matrix with zeros
        self.matrix = np.zeros((self.depth, self.width), dtype=int)

    def hash(self, key, col):
        """
        Compute the hash value for a key using the hash function for the specified column.
        
        Args:
            key: The key to hash
            col: The column/index of the hash function to use
            
        Returns:
            Hash value modulo the width of the sketch
        """
        return AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col]) % self.width
    
    def insert(self, key, val=1):
        """
        Insert a key-value pair into the sketch by incrementing counters in all rows.
        
        Args:
            key: The key to insert/update
            val: The value to add to the key's count (default 1)
            
        Returns:
            0 on successful insertion
        """
        for i in range(self.depth):
            pos = self.hash(key, i)
            # Check for potential overflow
            assert self.matrix[i, pos] != np.iinfo(self.matrix.dtype).max
            # Increment the counter at the hashed position in each row
            self.matrix[i, pos] += val
        return 0
    
    def query(self, key):
        """
        Query the estimated frequency of a key in the sketch.
        Returns the minimum value among all positions where the key hashes to.
        
        Args:
            key: The key to query
            
        Returns:
            Estimated frequency of the key
        """
        result = np.iinfo(self.matrix.dtype).max
        for i in range(self.depth):
            pos = self.hash(key, i)
            # Take the minimum value across all hash positions (conservative estimate)
            result = min(result, self.matrix[i, pos])
        return result
    
    def get_memory_usage(self):
        """
        Calculate the memory usage of the Count-Min sketch.
        
        Returns:
            Memory usage in bytes
        """
        return self.depth * self.width * self.matrix.itemsize