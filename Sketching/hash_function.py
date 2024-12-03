import random


def AwareHash(data, n, hash_val, scale, hardener):
    while n:
        hash_val *= scale
        hash_val += data[0]
        data = data[1:]
        n -= 1
    return hash_val ^ hardener


def mangle(key, nbytes):
    new_key = 0
    for i in range(nbytes):
        new_key |= key[nbytes - i - 1] << (i * 8)
    new_key = (new_key * 2083697005) & 0xffffffff
    ret_key = [(new_key >> (i * 8)) & 0xff for i in range(nbytes)]
    return bytes(ret_key)


def GenHashSeed(index, seed=None):
    if seed is None:
        seed = random.randint(0, 2**64 - 1)
    y = seed + index
    # x = int.from_bytes(mangle(y.to_bytes(8, 'little'), 8), 'little')
    return AwareHash(y.to_bytes(8, 'little'), 8, 388650253, 388650319, 1176845762)


if __name__ == '__main__':
    for i in range(5):
        print(GenHashSeed(i))