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

