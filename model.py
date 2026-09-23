"""
Prefix Caching and Cache-Aware Routing

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - PrefixCache
def block_hashes(tokens, block_size):
    # Compute chained hashes for all complete blocks.
    # A trailing partial block is intentionally ignored.
    hashes = []
    prev_hash = 0

    for i in range(0, len(tokens) - block_size + 1, block_size):
        block = tuple(tokens[i:i + block_size])
        current_hash = hash((prev_hash, block))
        hashes.append(current_hash)
        prev_hash = current_hash

    return hashes


class PrefixCache:
    def __init__(self, capacity_blocks, block_size):
        self.capacity_blocks, self.block_size = capacity_blocks, block_size
        self.blocks = {}
        self.tick = 0
        self.hits = self.misses = self.evictions = 0

    def lookup(self, tokens):
        # Generate the chained hashes for all complete blocks.
        hashes = block_hashes(tokens, self.block_size)

        n_present = 0

        # Only a leading prefix can be reused. Stop at the first miss.
        for h in hashes:
            if h not in self.blocks:
                break

            self.tick += 1
            self.blocks[h] = self.tick
            n_present += 1

        # Every full block not present in the leading prefix is a miss.
        self.hits += n_present
        self.misses += len(hashes) - n_present

        return n_present * self.block_size

    def insert(self, tokens):
        # Generate the chained hashes for all complete blocks.
        hashes = block_hashes(tokens, self.block_size)

        # Add missing blocks while maintaining the LRU block budget.
        for h in hashes:
            if h not in self.blocks:
                # Evict the least recently used block when the cache is full.
                if len(self.blocks) >= self.capacity_blocks:
                    lru_hash = min(self.blocks, key=self.blocks.get)
                    del self.blocks[lru_hash]
                    self.evictions += 1

                self.tick += 1
                self.blocks[h] = self.tick
            else:
                # Touch existing blocks as well.
                self.tick += 1
                self.blocks[h] = self.tick

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

# Step 2 - ttft_with_cache
def ttft_with_cache(prompt_len, cached_tokens, prefill_tps, overhead_s):
    # Cache lookup/handling overhead plus prefill time for uncached tokens.
    ttft = overhead_s + (prompt_len - cached_tokens) / prefill_tps
    return round(ttft, 6)


def prefill_saved_s(prompt_len, cached_tokens, prefill_tps):
    # Time saved by skipping prefill for cached tokens.
    return cached_tokens / prefill_tps


def kv_bytes(tokens, kv_bytes_per_token):
    # Total KV-cache memory required for the given number of tokens.
    return tokens * kv_bytes_per_token


def blocks_for(memory_bytes, block_size, kv_bytes_per_token):
    # Number of complete blocks that fit in the available memory.
    return int(memory_bytes // (block_size * kv_bytes_per_token))


def savings_report(prompt_lens, cached_lens, prefill_tps, overhead_s):
    # TTFT with caching for each request.
    ttfts = [
        ttft_with_cache(prompt_len, cached_tokens, prefill_tps, overhead_s)
        for prompt_len, cached_tokens in zip(prompt_lens, cached_lens)
    ]

    # TTFT without caching: no cached tokens, but the same cache overhead
    # is still part of the measured TTFT.
    ttfts_nocache = [
        ttft_with_cache(prompt_len, 0, prefill_tps, overhead_s)
        for prompt_len in prompt_lens
    ]

    total_prompt_tokens = sum(prompt_lens)
    total_cached_tokens = sum(cached_lens)

    mean_ttft = sum(ttfts) / len(ttfts)
    mean_ttft_nocache = sum(ttfts_nocache) / len(ttfts_nocache)

    # Fractional TTFT reduction relative to the no-cache workload.
    if mean_ttft_nocache != 0:
        ttft_reduction = 1 - mean_ttft / mean_ttft_nocache
    else:
        ttft_reduction = 0.0

    # Fraction of all prompt tokens that were served from cache.
    if total_prompt_tokens != 0:
        fraction_saved = total_cached_tokens / total_prompt_tokens
    else:
        fraction_saved = 0.0

    return {
        "mean_ttft": round(mean_ttft, 6),
        "mean_ttft_nocache": round(mean_ttft_nocache, 6),
        "ttft_reduction": round(ttft_reduction, 6),
        "tokens_saved": total_cached_tokens,
        "fraction_saved": round(fraction_saved, 6),
    }

