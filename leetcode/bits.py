def toBits(self, x):
    bits = []
    while x > 0:
        bits.append(x % 2)
        x = x >> 1
    return list(reversed(bits))

def fromBits(self, x):
    return int(x, 2)