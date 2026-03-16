from Utils.common import calNextPrime
from Sketching.hash_function import GenHashSeed, AwareHash


# Constants for the insertion result status
HIT, MISS_EVICT, MISS_INSERT = 0, 1, 2
# Threshold for evicting items based on negative counter ratio
EVICT_THRESHOLD = 1


class slot:
    """
    Represents a single slot in the heavy hitter filter to store key-value pairs.
    Each slot contains a key, its frequency value, and a negative counter to track
    how many times a different key has hashed to the same position.
    """
    def __init__(self, key, val=0, key_size=8):
        """
        Initialize a slot with a key and value
        
        Args:
            key: The key stored in this slot
            val: The frequency value associated with the key
            key_size: Size of the key in bytes
        """
        self.key = key
        self.val = val
        self.key_size = key_size
        self.negative_counter = 0  # Counts attempts to insert other keys at this position

    @classmethod
    def from_key(cls, key, key_size=8):
        """
        Create a slot instance from a key with default value
        
        Args:
            key: The key to store in the slot
            key_size: Size of the key in bytes
            
        Returns:
            A new slot instance with the given key
        """
        return cls(key, key_size=key_size)

    @classmethod
    def default(cls, key_size=8):
        """
        Create a default empty slot
        
        Args:
            key_size: Size of the key in bytes
            
        Returns:
            A new empty slot instance
        """
        return cls(None, key_size=key_size)

    def get_memory_usage(self):
        """
        Calculate the memory usage of this slot
        
        Returns:
            Memory usage in bytes (2 integers for val and negative_counter + key_size)
        """
        return 2 * 4 + self.key_size  # 2 integers (val, negative_counter) * 4 bytes + key_size


class heavyFilter:
    """
    Heavy hitter detection filter that identifies frequently occurring items in a data stream.
    This is a hash-based structure that tracks items likely to be heavy hitters and evicts
    items that are no longer frequent based on a negative counter threshold.
    """

    def __init__(self, slot_num, KEY_T_SIZE=13):
        """
        Initialize the heavy filter with specified capacity
        
        Args:
            slot_num: Number of slots to allocate (will be rounded up to next prime)
            KEY_T_SIZE: Size of keys in bytes
        """
        self.key_size = KEY_T_SIZE
        # Round up to next prime to reduce hash collisions
        self.size = calNextPrime(slot_num)
        # Initialize slots with default empty slots
        self.slots = [slot.default(key_size=KEY_T_SIZE) for _ in range(self.size)]
        # Generate three independent hash seeds for the AwareHash function
        self.h, self.s, self.n = [GenHashSeed(i) for i in range(3)]

    def insert(self, temp_key, val=1):
        """
        Insert a key-value pair into the heavy filter
        
        Args:
            temp_key: The key to insert
            val: The value to add to the key's frequency (default 1)
            
        Returns:
            Tuple of (status, slot) where status indicates insertion result and slot contains
            the evicted item if any (otherwise None)
        """
        temp_slot = None  # Placeholder for evicted item
        # Compute hash position for the key
        pos = AwareHash(temp_key, self.key_size, self.h, self.s, self.n) % self.size
        
        if self.slots[pos].key == temp_key:
            # Key already exists at this position, increment its value
            self.slots[pos].val += 1
            return HIT, temp_slot
        elif self.slots[pos].key is None:
            # Empty slot, insert the new key
            self.slots[pos].val = 1
            self.slots[pos].key = temp_key
            return HIT, temp_slot
        else:
            # Slot occupied by a different key, handle collision
            temp_slot = slot.default()
            # Increment negative counter since another key tried to use this slot
            self.slots[pos].negative_counter += 1
            # Check if the ratio of negative counter to value exceeds the eviction threshold
            if self.slots[pos].negative_counter / self.slots[pos].val >= EVICT_THRESHOLD:
                # Evict the current key and put the new key in this slot
                temp_slot.key = self.slots[pos].key
                temp_slot.val = self.slots[pos].val
                temp_slot.negative_counter = self.slots[pos].negative_counter
                self.slots[pos] = slot(temp_key, val, self.key_size)
                return MISS_EVICT, temp_slot  # Return that an eviction occurred
            else:
                # Keep the current key but return the new key info for further processing
                temp_slot.key = temp_key
                temp_slot.val = val
                temp_slot.negative_counter = 0
        
        return MISS_INSERT, temp_slot
    
    def query(self, temp_key):
        """
        Query the frequency of a key in the heavy filter
        
        Args:
            temp_key: The key to look up
            
        Returns:
            The frequency value of the key if found, otherwise 0
        """
        # Compute hash position for the key
        pos = AwareHash(temp_key, self.key_size, self.h, self.s, self.n) % self.size
        
        if self.slots[pos].key == temp_key:
            # Key found at the computed position
            return self.slots[pos].val
        
        # Key not found
        return 0
    
    def get_memory_usage(self):
        """
        Calculate the total memory usage of the heavy filter
        
        Returns:
            Total memory usage in bytes
        """
        return self.size * slot.default(key_size=self.key_size).get_memory_usage()