import math
import numpy as np

from scipy.stats import geom
from Utils.common import calNextPrime
from Sketching.hash_function import GenHashSeed, AwareHash
# from functools import cmp_to_key


class NitroSketch:

    update_probs = [(1.0 / (2**i)) for i in range(8)]

    def __init__(self, width: int, depth: int, KEY_T_SIZE=13):
        self.depth = depth
        self.width = calNextPrime(width)
        self.switch_thresh = (1.0 + math.sqrt(11.0 / self.width)) * self.width * self.width
        
        self.line_rate_enable = False
        self.update_prob = 1.0
        self.next_packet = 1
        self.next_bucket = 0
        self.key_size = KEY_T_SIZE
        self.square_sum = np.zeros(self.depth)
        self.array = np.zeros((self.depth, self.width), dtype=int)

        self.h = [GenHashSeed(i) for i in range(depth)]
        self.s = [GenHashSeed(i) for i in range(depth)]
        self.n = [GenHashSeed(i) for i in range(depth)]
        self.i = [GenHashSeed(i) for i in range(depth)]
        self.j = [GenHashSeed(i) for i in range(depth)]
        self.k = [GenHashSeed(i) for i in range(depth)]

    def hash(self, key, col):
        hash_value1 = AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col])
        hash_value2 = AwareHash(key, self.key_size, self.i[col], self.j[col], self.k[col])
        return hash_value1 % self.width, 1 - 2 * (hash_value2 % 2)

    def __del__(self):
        del self.array

    def always_line_rate_update(self, flowkey, value):
        self.__do_update(flowkey, value, self.update_prob)

    def always_correct_update(self, flowkey, value):
        if self.is_line_rate_update():
            self.__do_update(flowkey, value, self.update_prob)
        else:
            self.__do_update(flowkey, value, 1.0)

    def query(self, flowkey):
        values = np.zeros(self.depth, dtype=int)
        for i in range(self.depth):
            index, coeffi = self.hash(flowkey, i)
            values[i] = self.array[i][index] * coeffi
        return np.median(values)
    
    def get_memory_usage(self):
        return self.depth * self.width * self.array.itemsize +\
               self.depth * self.square_sum.itemsize

    def __do_update(self, flowkey, value, prob):
        self.next_packet -= 1
        if self.next_packet == 0:
            while True:
                i = self.next_bucket
                index, coeffi = self.hash(flowkey, i)
                delta = (1.0 * value / prob) * coeffi
                self.square_sum[i] += (2.0 * self.array[i][index] + delta) * delta
                self.array[i][index] += int(delta)
                self.get_next_update(prob)
                if self.next_packet > 0:
                    break

    def get_next_update(self, prob):
        sample = 1
        if prob < 1.0:
            sample = 1 + geom.rvs(prob)
        self.next_bucket += sample
        self.next_packet = self.next_bucket // self.depth
        self.next_bucket %= self.depth

    def is_line_rate_update(self):
        if self.line_rate_enable:
            return True
        values = self.square_sum.copy()
        values.sort()
        if self.depth % 2 == 1:
            median = values[self.depth // 2]
        else:
            median = (values[self.depth // 2 - 1] + values[self.depth // 2]) / 2
        if median >= self.switch_thresh:
            print("line rate update enable")
            self.line_rate_enable = True
        return self.line_rate_enable

    def adjust_update_prob(self, traffic_rate):
        log_rate = int(math.log2(traffic_rate))
        update_index = max(0, min(log_rate, 7))
        self.update_prob = self.update_probs[update_index]

    def insert(self, flowkey, value=1):
        self.__do_update(flowkey, value, self.update_prob)
