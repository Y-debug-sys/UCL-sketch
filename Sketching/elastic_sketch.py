from Utils.common import calNextPrime
from Sketching.cm_sketch import CMsketch
from Sketching.hash_function import GenHashSeed, AwareHash


def judge_if_swap(min_val, guard_val):
    return guard_val > (min_val << 3)


class Entry:
    def __init__(self, flowkey=None, val=0):
        self.flowkey = flowkey
        self.val = val
        self.flag = False
    
    def is_empty(self):
        return self.flowkey is None


class ElasticSketch:
    def __init__(self, num_buckets, num_per_bucket, width, depth, KEY_T_SIZE=13):
        self.key_size = KEY_T_SIZE
        self.num_buckets = calNextPrime(num_buckets)
        self.num_per_bucket = num_per_bucket
        self.cm = CMsketch(width, depth, KEY_T_SIZE)
        self.buckets = [[Entry() for _ in range(num_per_bucket)] for _ in range(self.num_buckets)]
        self.h, self.s, self.n = [GenHashSeed(i) for i in range(3)]

    def hash(self, key):
        return AwareHash(key, self.key_size, self.h, self.s, self.n) % self.num_buckets
    
    def get_memory_usage(self):
        return (len(self.buckets) * len(self.buckets[0]) * (4  + self.key_size + 0.125) + self.cm.get_memory_usage())

    def heavypart_insert(self, flowkey, val):
        # matched, empty = -1, -1
        index = self.hash(flowkey)
        bucket = self.buckets[index]
        
        for i, entry in enumerate(bucket[:-1]):
            
            if entry.flowkey == flowkey:
                # matched = i
                entry.val += val
                return 0

            if entry.is_empty():
                entry.flowkey = flowkey
                entry.val = val
                return 0

        min_entry = min(bucket[:-1], key=lambda e: e.val)
        guard_entry = bucket[-1]
        guard_entry.val += 1

        if not judge_if_swap(min_entry.val, guard_entry.val):
            return 2
        else:
            swap_key = min_entry.flowkey
            swap_val = min_entry.val
            guard_entry.val = 0
            min_entry.flowkey = flowkey
            min_entry.val = val
            min_entry.flag = True
            return 1, swap_key, swap_val

    def lightpart_insert(self, flowkey, val):
        self.cm.insert(flowkey, val)

    def insert(self, flowkey, val=1):
        result = self.heavypart_insert(flowkey, val)
        if result == 0:
            return
        elif result == 2:
            self.lightpart_insert(flowkey, val)
        elif result[0] == 1:
            swap_key, swap_val = result[1], result[2]
            self.lightpart_insert(swap_key, swap_val)

    def heavypart_query(self, flowkey):
        index = self.hash(flowkey)
        for entry in self.buckets[index][:-1]:
            if entry.flowkey == flowkey:
                return entry.val, entry.flag
        return 0, False

    def lightpart_query(self, flowkey):
        return self.cm.query(flowkey)

    def query(self, flowkey):
        heavy_result, flag = self.heavypart_query(flowkey)
        if heavy_result == 0 or flag:
            light_result = self.lightpart_query(flowkey)
        else:
            light_result = 0
        return heavy_result + light_result
