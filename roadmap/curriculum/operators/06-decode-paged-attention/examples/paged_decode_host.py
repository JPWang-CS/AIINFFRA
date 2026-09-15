"""CPU-testable metadata and byte-range contracts for the decode wrapper."""


def byte_ranges_overlap(left_begin, left_bytes, right_begin, right_bytes):
    left_end = left_begin + left_bytes
    right_end = right_begin + right_bytes
    return left_begin < right_end and right_begin < left_end


def _overlap(left, right):
    return byte_ranges_overlap(left.data_ptr(), left.numel() * left.element_size(),
                               right.data_ptr(), right.numel() * right.element_size())


def validate_head_counts(q_heads, kv_heads):
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("q_heads and kv_heads must be positive and divisible")


def validate_trusted_capacity(max_length, max_pages, page_size):
    if not isinstance(max_length, int) or isinstance(max_length, bool):
        raise ValueError("trusted max_length must be an integer")
    if max_length <= 0 or max_length > max_pages * page_size:
        raise ValueError("trusted max_length exceeds unchanged table capacity")


def validate_memory_contract(q, cache_k, cache_v, lengths, table, new_k, new_v, out):
    """Check byte ownership before append/read launches; accepts fake tensors."""
    tensors = [q, cache_k, cache_v, lengths, table]
    if new_k is not None:
        tensors.extend([new_k, new_v])
    if any(_overlap(out, tensor) for tensor in tensors):
        raise ValueError("out must not overlap q/cache/new KV/metadata")
    if _overlap(cache_k, cache_v):
        raise ValueError("cache_k and cache_v must be separate pools")
    if any(_overlap(cache, tensor) for cache in (cache_k, cache_v) for tensor in (q, lengths, table)):
        raise ValueError("writeable cache must not overlap q or metadata")
    if new_k is not None and any(_overlap(new, tensor)
                                 for new in (new_k, new_v)
                                 for tensor in (q, cache_k, cache_v, lengths, table)):
        raise ValueError("new KV must not overlap q/cache/metadata")
    if new_k is not None and _overlap(new_k, new_v):
        raise ValueError("new_k and new_v must be separate pools")


def validate_private_append_targets(lengths, table, num_pages, page_size):
    """Validate only references visible in this call's batch.

    A serving allocator must additionally guarantee that no external batch,
    prefix view, reader stream, or copy-on-write owner references these pages.
    """
    max_pages = len(table[0])
    owners = {}
    for batch, length in enumerate(lengths):
        if length <= 0 or length > max_pages * page_size:
            raise ValueError("baseline requires 1 <= length <= max_pages * page_size")
        needed = (length + page_size - 1) // page_size
        for logical_page in range(needed):
            physical = int(table[batch][logical_page])
            if not 0 <= physical < num_pages:
                raise ValueError("logical page maps outside the physical page pool")
            owners.setdefault(physical, []).append((batch, logical_page))
    targets = set()
    for batch, length in enumerate(lengths):
        logical_page, offset = divmod(length - 1, page_size)
        physical = int(table[batch][logical_page])
        if owners[physical] != [(batch, logical_page)]:
            raise ValueError("append target page is shared; privateize/COW it first")
        target = (physical, offset)
        if target in targets:
            raise ValueError("append target slot is duplicated")
        targets.add(target)
