from bitarray import bitarray
from Utils.common import calNextPrime
from Sketching.hash_function import AwareHash, GenHashSeed


class BloomFilter:
    def __init__(self, w, hash_num, KEY_T_SIZE=8):
        self.key_size = KEY_T_SIZE
        self.width = calNextPrime(w)
        self.size = (self.width >> 3) + ((self.width & 0x7) != 0)
        self.bit_array = bitarray(self.size * 8)
        self.bit_array.setall(0)
        self.h = [GenHashSeed(i) for i in range(hash_num)]
        self.s = [GenHashSeed(i) for i in range(hash_num)]
        self.n = [GenHashSeed(i) for i in range(hash_num)]
        self.hash_num = hash_num
    
    def getbit(self, k):
        for i in range(self.hash_num):
            pos = AwareHash(k, self.key_size, self.h[i], self.s[i], self.n[i]) % self.width
            if not self.bit_array[pos]:
                return False
        return True
    
    def setbit(self, k):
        for i in range(self.hash_num):
            pos = AwareHash(k, self.key_size, self.h[i], self.s[i], self.n[i]) % self.width
            self.bit_array[pos] = 1
    
    def reset(self):
        self.bit_array.setall(0)
    
    def get_memory_usage(self):
        return self.size
    
    def get_hash_num(self):
        return self.hash_num