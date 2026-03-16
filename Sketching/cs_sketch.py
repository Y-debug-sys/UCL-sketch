import numpy as np

from Utils.common import calNextPrime
from Sketching.hash_function import GenHashSeed, AwareHash


class Csketch:
    """
    Count-Sketch implementation - a probabilistic data structure for estimating
    the frequency of events in a data stream. Unlike Count-Min Sketch, Count-Sketch
    can handle both positive and negative updates and provides unbiased estimates
    with strong concentration guarantees.
    """

    def __init__(self, width: int, depth: int, KEY_T_SIZE=13):
        """
        Initialize the Count-Sketch with specified dimensions.
        
        Args:
            width: Width of the sketch matrix (number of counters per row)
            depth: Depth of the sketch matrix (number of hash functions/rows)
            KEY_T_SIZE: Size of keys in bytes
        """
        self.key_size = KEY_T_SIZE
        # Use next prime to reduce hash collisions
        self.depth, self.width = depth, calNextPrime(width)
        # Generate first set of hash function parameters for position calculation
        self.h = [GenHashSeed(i) for i in range(depth)]  # Hash function parameter h
        self.s = [GenHashSeed(i) for i in range(depth)]  # Hash function parameter s
        self.n = [GenHashSeed(i) for i in range(depth)]  # Hash function parameter n
        # Generate second set of hash function parameters for sign calculation
        self.i = [GenHashSeed(i) for i in range(depth)]  # Hash function parameter i
        self.j = [GenHashSeed(i) for i in range(depth)]  # Hash function parameter j
        self.k = [GenHashSeed(i) for i in range(depth)]  # Hash function parameter k
        # Initialize the counting matrix with zeros
        self.matrix = np.zeros((self.depth, self.width), dtype=int)

    def hash(self, key, col):
        """
        Compute the hash value and sign for a key using two independent hash functions.
        
        Args:
            key: The key to hash
            col: The column/index of the hash function to use
            
        Returns:
            Tuple of (position, sign) where position is the index in the row and
            sign is either +1 or -1 for the update
        """
        hash_value1 = AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col])
        hash_value2 = AwareHash(key, self.key_size, self.i[col], self.j[col], self.k[col])
        return hash_value1 % self.width, 1 - 2 * (hash_value2 % 2)  # Position and sign (+1/-1)
    
    def insert(self, key, val=1):
        """
        Insert a key-value pair into the sketch by updating counters in all rows.
        
        Args:
            key: The key to insert/update
            val: The value to add to the key's count (default 1)
            
        Returns:
            0 on successful insertion
        """
        for i in range(self.depth):
            pos, sign = self.hash(key, i)
            # Check for potential overflow
            assert self.matrix[i, pos] != np.iinfo(self.matrix.dtype).max
            # Update the counter at the hashed position with signed value
            self.matrix[i, pos] += (val * sign)
        return 0
    
    def query(self, key):
        """
        Query the estimated frequency of a key in the sketch.
        Returns the median of the adjusted values across all positions where the key hashes to.
        
        Args:
            key: The key to query
            
        Returns:
            Estimated frequency of the key (absolute value of median estimate)
        """
        results = []
        for i in range(self.depth):
            pos, sign = self.hash(key, i)
            # Adjust the stored value by the sign to get the estimate
            results.append(self.matrix[i, pos] * sign)
        # Return the absolute value of the median estimate
        return abs(np.median(results))
    
    def get_memory_usage(self):
        """
        Calculate the memory usage of the Count-Sketch.
        
        Returns:
            Memory usage in bytes
        """
        return self.depth * self.width * self.matrix.itemsize
