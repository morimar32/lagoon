"""FNV-1a 64-bit hash implementation."""

FNV1A_OFFSET: int = 14695981039346656037
FNV1A_PRIME: int = 1099511628211
_MASK64: int = 0xFFFFFFFFFFFFFFFF


def fnv1a_u64(s: str) -> int:
    """Compute FNV-1a 64-bit hash of a string (UTF-8 bytes)."""
    h = FNV1A_OFFSET
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * FNV1A_PRIME) & _MASK64
    return h
