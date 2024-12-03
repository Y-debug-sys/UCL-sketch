from Utils.common import calNextPrime
from Sketching.bloom_filter import BloomFilter
from Sketching.hash_function import GenHashSeed, AwareHash


def bitwise_xor_bytes(a, b):
    result_int = int.from_bytes(a, byteorder="little") ^ int.from_bytes(b, byteorder="little")
    return result_int.to_bytes(max(len(a), len(b)), byteorder="little")


class CountTableEntry:
    def __init__(self, key_size):
        self.flow_count = 0
        self.packet_count = 0
        self.flowXOR = None
        self.key_size = key_size

    def get_memory_usage(self):
        return 4 * 2 + self.key_size


class FlowRadar:
    def __init__(
        self, 
        flow_filter_size, 
        flow_filter_hash, 
        count_table_size, 
        count_table_hash, 
        KEY_T_SIZE=8
    ):
        self.num_bitmap = calNextPrime(flow_filter_size)
        self.num_bit_hash = flow_filter_hash
        self.num_count_table = calNextPrime(count_table_size)
        self.num_count_hash = count_table_hash
        self.num_flows = 0
        self.key_size = KEY_T_SIZE
        self.flow_filter = BloomFilter(self.num_bitmap, self.num_bit_hash, KEY_T_SIZE)
        self.count_table = [CountTableEntry(KEY_T_SIZE) for _ in range(self.num_count_table)]

        self.h = [GenHashSeed(i) for i in range(count_table_hash)]
        self.s = [GenHashSeed(i) for i in range(count_table_hash)]
        self.n = [GenHashSeed(i) for i in range(count_table_hash)]

    def hash(self, key, col):
        return AwareHash(key, self.key_size, self.h[col], self.s[col], self.n[col]) % self.num_count_table

    def insert(self, flowkey, val=1):
        exist = self.flow_filter.getbit(flowkey)
        if not exist:
            self.flow_filter.setbit(flowkey)
            self.num_flows += 1

        for i in range(self.num_count_hash):
            index = self.hash(flowkey, i)
            if not exist:
                self.count_table[index].flow_count += 1
                self.count_table[index].flowXOR = bitwise_xor_bytes(self.count_table[index].flowXOR, flowkey) \
                if self.count_table[index].flowXOR else flowkey
            self.count_table[index].packet_count += val
    
    def decode(self):
        est = {}
        count_table_set = set(self.count_table)

        while count_table_set:
            print('cycle')
            index = self.count_table.index(min(count_table_set, key=lambda x: x.flow_count))
            value = self.count_table[index].flow_count
            if value > 1:
                break

            count_table_set.remove(self.count_table[index])
            if value == 0:
                continue

            flowkey = self.count_table[index].flowXOR
            size = self.count_table[index].packet_count
            for i in range(self.num_count_hash):
                l = self.hash(flowkey, i)
                count_table_set.remove(self.count_table[l])
                self.count_table[l].flow_count -= 1
                self.count_table[l].flowXOR = bitwise_xor_bytes(self.count_table[l].flowXOR, flowkey)
                count_table_set.add(self.count_table[l])

            est[flowkey] = size

        return est

    def get_memory_usage(self):
        return (sum(map(lambda x: x.flow_count + x.packet_count, self.count_table))
                + self.count_table[0].get_memory_usage() * self.num_count_table
                + self.flow_filter.get_memory_usage())

