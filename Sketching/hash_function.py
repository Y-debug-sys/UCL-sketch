import random


def AwareHash(data, n, hash_val, scale, hardener):
    """
    Custom hash function that processes data byte by byte to compute a hash value.
    This function creates a hash by multiplying the current hash value by a scale,
    adding the next byte of data, and continuing until all bytes are processed.
    
    Args:
        data: Byte sequence to hash
        n: Number of bytes to process
        hash_val: Initial hash value
        scale: Scaling factor for the hash computation
        hardener: Value XORed with final hash to make it more unpredictable
        
    Returns:
        Computed hash value
    """
    while n:
        hash_val *= scale
        hash_val += data[0]
        data = data[1:]
        n -= 1
    return hash_val ^ hardener


def mangle(key, nbytes):
    """
    Mangles a key by reversing byte order, applying multiplication with a constant,
    and then converting back to bytes with the original byte order.
    
    Args:
        key: Input key as a sequence of bytes
        nbytes: Number of bytes in the key
        
    Returns:
        Mangled key as bytes
    """
    new_key = 0
    # Reverse the byte order and construct a new integer
    for i in range(nbytes):
        new_key |= key[nbytes - i - 1] << (i * 8)
    # Apply multiplication with magic number and mask to 32 bits
    new_key = (new_key * 2083697005) & 0xffffffff
    # Convert back to bytes in original order
    ret_key = [(new_key >> (i * 8)) & 0xff for i in range(nbytes)]
    return bytes(ret_key)


def GenHashSeed(index, seed=None):
    """
    Generate a unique hash seed based on an index and optional seed value.
    This function creates a deterministic hash seed that can be used to
    generate independent hash functions for sketches.
    
    Args:
        index: Index to differentiate different hash functions
        seed: Optional base seed value (random if not provided)
        
    Returns:
        Generated hash seed value
    """
    if seed is None:
        seed = random.randint(0, 2**64 - 1)
    y = seed + index
    # x = int.from_bytes(mangle(y.to_bytes(8, 'little'), 8), 'little')
    return AwareHash(y.to_bytes(8, 'little'), 8, 388650253, 388650319, 1176845762)


if __name__ == '__main__':
    for i in range(5):
        print(GenHashSeed(i))