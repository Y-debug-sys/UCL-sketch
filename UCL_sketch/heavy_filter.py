from Utils.common import calNextPrime
from Sketching.hash_function import GenHashSeed, AwareHash


HIT, MISS_EVICT, MISS_INSERT = 0, 1, 2
EVICT_THRESHOLD = 1


class slot:
    def __init__(self, key, val=0, key_size=8):
        self.key = key
        self.val = val
        self.key_size = key_size
        self.negative_counter = 0

    @classmethod
    def from_key(cls, key, key_size=8):
        return cls(key, key_size=key_size)

    @classmethod
    def default(cls, key_size=8):
        return cls(None, key_size=key_size)

    def get_memory_usage(self):
        return 2 * 4 + self.key_size


class heavyFilter:

    def __init__(self, slot_num, KEY_T_SIZE=13):
        self.key_size = KEY_T_SIZE
        self.size = calNextPrime(slot_num)
        self.slots = [slot.default(key_size=KEY_T_SIZE) for _ in range(self.size)]
        self.h, self.s, self.n = [GenHashSeed(i) for i in range(3)]

    def insert(self, temp_key, val=1):
        temp_slot = None
        pos = AwareHash(temp_key, self.key_size, self.h, self.s, self.n) % self.size
        
        if self.slots[pos].key == temp_key:
            self.slots[pos].val += 1
            return HIT, temp_slot
        elif self.slots[pos].key is None:
            self.slots[pos].val = 1
            self.slots[pos].key = temp_key
            return HIT, temp_slot
        else:
            temp_slot = slot.default()
            self.slots[pos].negative_counter += 1
            if self.slots[pos].negative_counter / self.slots[pos].val >= EVICT_THRESHOLD:
                temp_slot.key = self.slots[pos].key
                temp_slot.val = self.slots[pos].val
                temp_slot.negative_counter = self.slots[pos].negative_counter
                self.slots[pos] = slot(temp_key, val, self.key_size)
                return MISS_EVICT, temp_slot
            else:
                temp_slot.key = temp_key
                temp_slot.val = val
                temp_slot.negative_counter = 0
        
        return MISS_INSERT, temp_slot
    
    def query(self, temp_key):
        pos = AwareHash(temp_key, self.key_size, self.h, self.s, self.n) % self.size
        
        if self.slots[pos].key == temp_key:
            return self.slots[pos].val
        
        return 0
    
    def get_memory_usage(self):
        return self.size * slot.default(key_size=self.key_size).get_memory_usage()