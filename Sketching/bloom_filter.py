from bitarray import bitarray
from Utils.common import calNextPrime
from Sketching.hash_function import AwareHash, GenHashSeed


class BloomFilter:
    """
    Bloom Filter implementation for probabilistic membership testing.
    A Bloom filter is a space-efficient probabilistic data structure that answers
    whether an element is a member of a set. False positive matches are possible,
    but false negatives are not – it can return 'possibly in set' or 'definitely not in set'.
    """
    def __init__(self, w, hash_num, KEY_T_SIZE=8):
        """
        Initialize the Bloom filter with specified parameters.
        
        Args:
            w: Width of the Bloom filter (number of bits in the bit array)
            hash_num: Number of hash functions to use
            KEY_T_SIZE: Size of keys in bytes
        """
        self.key_size = KEY_T_SIZE
        # Round up to next prime to reduce hash collisions
        self.width = calNextPrime(w)
        # Calculate size in bytes (rounding up)
        self.size = (self.width >> 3) + ((self.width & 0x7) != 0)  # Equivalent to ceil(width/8)
        # Initialize bit array with all zeros
        self.bit_array = bitarray(self.size * 8)
        self.bit_array.setall(0)
        # Generate hash function parameters for multiple independent hash functions
        self.h = [GenHashSeed(i) for i in range(hash_num)]  # Hash function parameter h
        self.s = [GenHashSeed(i) for i in range(hash_num)]  # Hash function parameter s
        self.n = [GenHashSeed(i) for i in range(hash_num)]  # Hash function parameter n
        self.hash_num = hash_num  # Number of hash functions to use
    
    def getbit(self, k):
        """
        Check if a key is possibly in the set (membership test).
        
        Args:
            k: The key to check for membership
            
        Returns:
            True if the key is possibly in the set, False if definitely not in the set
        """
        for i in range(self.hash_num):
            # Compute hash position for the key using the i-th hash function
            pos = AwareHash(k, self.key_size, self.h[i], self.s[i], self.n[i]) % self.width
            # If any of the positions is 0, the key is definitely not in the set
            if not self.bit_array[pos]:
                return False
        # If all positions are 1, the key is possibly in the set
        return True
    
    def setbit(self, k):
        """
        Add a key to the set by setting the appropriate bits in the bit array.
        
        Args:
            k: The key to add to the set
        """
        for i in range(self.hash_num):
            # Compute hash position for the key using the i-th hash function
            pos = AwareHash(k, self.key_size, self.h[i], self.s[i], self.n[i]) % self.width
            # Set the bit at the computed position to 1
            self.bit_array[pos] = 1
    
    def reset(self):
        """
        Reset the Bloom filter by setting all bits back to 0.
        """
        self.bit_array.setall(0)
    
    def get_memory_usage(self):
        """
        Get the memory usage of the Bloom filter in bytes.
        
        Returns:
            Memory usage in bytes
        """
        return self.size
    
    def get_hash_num(self):
        """
        Get the number of hash functions used by the Bloom filter.
        
        Returns:
            Number of hash functions
        """
        return self.hash_num