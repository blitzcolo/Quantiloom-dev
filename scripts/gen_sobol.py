#!/usr/bin/env python3
"""Generate and VERIFY the Sobol' machinery before it goes into a shader.

The point of this script is the verification, not the table. A wrong direction
number or a broken scramble produces a sequence that still looks random and
still converges, just more slowly -- exactly the kind of bug that survives
review and quietly costs the speedup it was supposed to buy.

Design note. The first attempt used nine Sobol' dimensions, one per decision.
That is wrong: Sobol' is a (t,s)-sequence whose t degrades with dimension, and
the checks below show the pair (4,5) failing the elementary-interval property
at 2^-4 x 2^-4 while (0,1) passes. The 2D decisions would have been stratified
on each axis alone and not jointly, which is most of the benefit gone.

So this uses PADDED Sobol' instead (PBRT-v4's PaddedSobolSampler, Burley 2020):
only dimensions 0 and 1 -- the pair that is a genuine (0,2)-sequence -- with an
independent Owen scramble and index shuffle per decision. Every 2D decision then
gets a properly stratified 2D point set, decisions are decorrelated by their
scrambles, and the table shrinks to one 32-entry column.
"""

NBITS = 32
M32 = 0xFFFFFFFF


def dim1_matrix():
    """Sobol' dimension 1: polynomial x + 1, so m_i = m_{i-1} ^ (m_{i-1} << 1).

    Dimension 0 needs no table -- its v_i is 2^(31-i), which makes the XOR of
    the selected columns exactly the bit reversal of the index.
    """
    m = [1]
    for i in range(1, NBITS):
        m.append((m[i - 1] ^ (m[i - 1] << 1)) & M32)
    return [(m[i] << (NBITS - 1 - i)) & M32 for i in range(NBITS)]


V1 = dim1_matrix()


def sobol_dim0(index):
    return int(f"{index & M32:032b}"[::-1], 2)


def sobol_dim1(index):
    x = 0
    i, bit = index, 0
    while i:
        if i & 1:
            x ^= V1[bit]
        i >>= 1
        bit += 1
    return x


# --- Owen scrambling: Burley 2020, "Practical Hash-based Owen Scrambling" ---

def laine_karras(x, seed):
    x = (x + seed) & M32
    x ^= (x * 0x6C50B47C) & M32
    x ^= (x * 0xB82F1E52) & M32
    x ^= (x * 0xC7AFE638) & M32
    x ^= (x * 0x8D22F6E6) & M32
    return x & M32


def owen_scramble(x, seed):
    x = sobol_dim0(x)          # reverse bits
    x = laine_karras(x, seed)
    return sobol_dim0(x)       # reverse back


def hash_u32(x):
    """Jarzynski & Olano 2020 integer hash -- the seed mixer."""
    x = (x * 747796405 + 2891336453) & M32
    x = (((x >> ((x >> 28) + 4)) ^ x) * 277803737) & M32
    return ((x >> 22) ^ x) & M32


# --- checks ---

def check_matrix_diagonal():
    """v_i's LOWEST set bit must be exactly bit 31-i.

    m_i is odd and m_i < 2^(i+1), so v_i = m_i << (31-i) has bit 31-i set and
    nothing below it. Bits ABOVE are free -- that is the part the first version
    of this check got backwards, flagging a correct table.
    """
    for i in range(NBITS):
        v = V1[i]
        if not (v >> (NBITS - 1 - i)) & 1:
            return f"bit {i}: diagonal bit clear (v={v:#010x})"
        if v & ((1 << (NBITS - 1 - i)) - 1):
            return f"bit {i}: bits below the diagonal set (v={v:#010x})"
    return None


def check_1d_balance(gen, m):
    n = 1 << m
    seen = [0] * n
    for i in range(n):
        seen[gen(i) >> (NBITS - m)] += 1
    bad = [k for k, c in enumerate(seen) if c != 1]
    return None if not bad else f"{len(bad)} buckets off at 2^{m}"


def check_02_sequence(xgen, ygen, m):
    """Every elementary interval of volume 2^-m holds exactly one of the first
    2^m points. This is what makes a 2D decision jointly stratified rather than
    merely uniform on each axis."""
    n = 1 << m
    pts = [(xgen(i), ygen(i)) for i in range(n)]
    for p in range(m + 1):
        q = m - p
        grid = {}
        for (x, y) in pts:
            key = (x >> (NBITS - p) if p else 0, y >> (NBITS - q) if q else 0)
            grid[key] = grid.get(key, 0) + 1
        if len(grid) != n or any(c != 1 for c in grid.values()):
            return f"2^-{p} x 2^-{q} not one-per-cell"
    return None


def check_scrambled_02(trials, m):
    """The real test of the hash constants: Owen scrambling is only a valid
    randomization if it PRESERVES the net property. If laine_karras is not a
    proper base-2 nested permutation, the stratification silently dies here
    while the numbers still look uniform."""
    for t in range(trials):
        sx, sy = hash_u32(t * 2 + 1), hash_u32(t * 2 + 2)
        err = check_02_sequence(
            lambda i: owen_scramble(sobol_dim0(i), sx),
            lambda i: owen_scramble(sobol_dim1(i), sy), m)
        if err:
            return f"seed pair {t}: {err}"
    return None


def check_scramble_is_bijection(trials, m):
    """A permutation, not a hash: 2^m distinct inputs must give 2^m distinct
    outputs in their top m bits."""
    for t in range(trials):
        s = hash_u32(t + 12345)
        top = {owen_scramble(i << (NBITS - m), s) >> (NBITS - m)
               for i in range(1 << m)}
        if len(top) != (1 << m):
            return f"seed {t}: {1 << m} inputs -> {len(top)} outputs"
    return None


# --- the padded sampler, exactly as sampling.hlsli builds it ---

SHIFT = 8


def next_seed(s):
    s = (s * 747796405 + 2891336453) & M32
    s = (((s >> ((s >> 28) + 4)) ^ s) * 277803737) & M32
    return ((s >> 22) ^ s) & M32


def shuffle_index(index, seed):
    return (owen_scramble((index & 0xFFFFFF) << SHIFT, seed) >> SHIFT) & 0xFFFFFF


def slot_1d(index, seed):
    return owen_scramble(sobol_dim0(shuffle_index(index, next_seed(seed))), seed)


def slot_2d(index, seed):
    seed_y = next_seed(seed)
    idx = shuffle_index(index, next_seed(seed_y))
    return owen_scramble(sobol_dim0(idx), seed), owen_scramble(sobol_dim1(idx), seed_y)


def check_slots_are_independent(trials, m, cells=16):
    """Two slots drawn at the same sample index must be JOINTLY uniform.

    This is the check that the first version of sampling.hlsli would have
    failed, and the reason it exists. Owen scrambling randomizes a sequence but
    does not decorrelate two copies of it: the leading digit is flipped by a
    seed-dependent CONSTANT, so two differently-scrambled copies of the same
    point have leading bits that agree for every index or disagree for every
    index. Without a per-slot index shuffle this reports ~240 of 256 empty
    cells and leading-bit agreement of exactly 100% or exactly 0%.

    Why it matters in a renderer: a threshold test on one slot (pick a lobe,
    pick an emitter) then partitions samples by that leading bit, and on the
    partition every other slot's leading bit is constant -- so the specular
    branch samples half the hemisphere and never the other half. It reads as
    ordinary noise and cost 3.6x on a scene with mixed lobes.
    """
    n = 1 << m
    for t in range(trials):
        sa, sb = hash_u32(t * 7 + 1), hash_u32(t * 7 + 2)
        hist = {}
        agree = 0
        for i in range(n):
            a = slot_1d(i, sa) / 2**32
            b = slot_1d(i, sb) / 2**32
            hist[(int(a * cells), int(b * cells))] = 1
            agree += (a < 0.5) == (b < 0.5)
        empty = cells * cells - len(hist)
        frac = agree / n
        # Uniform random leaves ~exp(-n/cells^2) of the cells empty and agrees
        # half the time. Locked slots leave almost all of them empty.
        if empty > cells * cells // 2:
            return f"seeds {t}: {empty}/{cells*cells} joint cells empty -- slots are locked"
        if not 0.35 < frac < 0.65:
            return f"seeds {t}: leading bits agree {frac:.0%} of the time"
    return None


def check_progressive_prefixes(trials, kmax):
    """Every 2^k prefix of a slot must still be stratified, after shuffling.

    An interactive render is stopped wherever the user stops dragging, so the
    stratification has to hold at the count actually reached and not only at
    the total. The index shuffle is itself an Owen scramble, which is what
    preserves this -- an arbitrary permutation would not.
    """
    for t in range(trials):
        seed = hash_u32(t + 31)
        for k in range(2, kmax + 1):
            n = 1 << k
            vals = sorted(slot_1d(i, seed) / 2**32 for i in range(n))
            if any(int(v * n) != q for q, v in enumerate(vals)):
                return f"seed {t}, prefix 2^{k}: not one sample per stratum"
            if k % 2 == 0:                      # square grid available
                c = 1 << (k // 2)
                cellset = set()
                for i in range(n):
                    x, y = slot_2d(i, seed)
                    cellset.add((int(x / 2**32 * c), int(y / 2**32 * c)))
                if len(cellset) != n:
                    return f"seed {t}, prefix 2^{k}: {n - len(cellset)} 2D cells doubled up"
    return None


def main():
    print("verifying ...")
    checks = [
        ("dim1 matrix diagonal", check_matrix_diagonal()),
        ("dim0 1D balance at 2^14", check_1d_balance(sobol_dim0, 14)),
        ("dim1 1D balance at 2^14", check_1d_balance(sobol_dim1, 14)),
        ("(dim0,dim1) is a (0,2)-sequence at 2^10",
         check_02_sequence(sobol_dim0, sobol_dim1, 10)),
        ("Owen scramble is a bijection", check_scramble_is_bijection(8, 10)),
        ("scrambling PRESERVES the (0,2) property (32 seed pairs)",
         check_scrambled_02(32, 8)),
        ("two slots are jointly uniform (padding actually pads)",
         check_slots_are_independent(6, 8)),
        ("every 2^k prefix of a slot is still stratified",
         check_progressive_prefixes(4, 10)),
    ]
    failed = False
    for name, err in checks:
        print(f"  {'FAIL  ' if err else 'ok    '}{name}{': ' + err if err else ''}")
        failed |= bool(err)
    if failed:
        raise SystemExit("verification failed -- not emitting")

    print()
    print("static const uint SOBOL_DIM1[32] = {")
    for i in range(0, NBITS, 4):
        print("    " + " ".join(f"0x{V1[i + k]:08x}u," for k in range(4)))
    print("};")


if __name__ == "__main__":
    main()
