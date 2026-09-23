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

# Step 3 - workload
import numpy as np

def workload(kind, n, rng, vocab=1000):
    # Generate prompts for the requested workload type.
    if kind == "chat":
        # Draw the shared system prompt once and reuse it for every request.
        system_prompt = rng.integers(0, vocab, size=256).tolist()

        prompts = []
        for _ in range(n):
            user_turn = rng.integers(20, 61)
            prompt = system_prompt + rng.integers(
                0, vocab, size=user_turn
            ).tolist()
            prompts.append(prompt)

        return prompts

    elif kind == "multi_turn":
        # Create 8 conversations up front, each with its own 64-token opener.
        conversations = [
            rng.integers(0, vocab, size=64).tolist()
            for _ in range(8)
        ]

        prompts = []
        for _ in range(n):
            # Select a conversation and append a new user turn to its history.
            conversation_idx = rng.integers(0, 8)
            new_turn_len = rng.integers(20, 61)
            new_turn = rng.integers(
                0, vocab, size=new_turn_len
            ).tolist()

            conversations[conversation_idx].extend(new_turn)

            # Send the entire conversation history.
            prompts.append(conversations[conversation_idx].copy())

        return prompts

    elif kind == "rag":
        # Build the document pool once and reuse documents across prompts.
        documents = [
            rng.integers(0, vocab, size=200).tolist()
            for _ in range(50)
        ]

        # Draw the fixed instruction once for all RAG requests.
        instruction = rng.integers(0, vocab, size=16).tolist()

        prompts = []
        for _ in range(n):
            # Choose 2 distinct documents and preserve the sampled order.
            doc_indices = rng.choice(50, size=2, replace=False)

            question_len = rng.integers(10, 31)
            question = rng.integers(
                0, vocab, size=question_len
            ).tolist()

            prompt = (
                instruction
                + documents[doc_indices[0]]
                + documents[doc_indices[1]]
                + question
            )
            prompts.append(prompt)

        return prompts

    elif kind == "unique":
        # Generate a completely random-length prompt for each request.
        prompts = []
        for _ in range(n):
            prompt_len = rng.integers(200, 401)
            prompt = rng.integers(
                0, vocab, size=prompt_len
            ).tolist()
            prompts.append(prompt)

        return prompts

    else:
        raise ValueError(
            "kind must be one of: 'chat', 'multi_turn', 'rag', 'unique'"
        )


def simulate_cache(prompts, cache, prefill_tps, overhead_s):
    # Record the number of tokens reused from the cache for each prompt.
    cached_lens = []

    for prompt in prompts:
        # Lookup must happen before insertion so the current request
        # receives credit only for blocks already cached.
        cached_tokens = cache.lookup(prompt)
        cached_lens.append(cached_tokens)

        # Insert the prompt after the lookup.
        cache.insert(prompt)

    # Reuse the savings_report implementation from Step 2.
    prompt_lens = [len(prompt) for prompt in prompts]
    report = savings_report(
        prompt_lens,
        cached_lens,
        prefill_tps,
        overhead_s,
    )

    # Add the cache hit rate to the workload-level report.
    report["hit_rate"] = cache.hit_rate()

    return report

# Step 4 - Router
class Router:
    def __init__(self, n_replicas, policy, capacity_blocks, block_size, window=32):
        self.n, self.policy, self.block_size, self.window = (
            n_replicas,
            policy,
            block_size,
            window,
        )
        self.caches = [
            PrefixCache(capacity_blocks, block_size)
            for _ in range(n_replicas)
        ]
        self.assigned = [0] * n_replicas
        self.recent = []
        self.rr = 0
        self.last_cached = 0

    def load(self, i):
        # Load is the number of recent requests assigned to replica i.
        return sum(replica == i for replica in self.recent)

    def route(self, prompt):
        if self.policy == "round_robin":
            # Cycle through replicas in order.
            replica = self.rr % self.n
            self.rr += 1

        elif self.policy == "least_loaded":
            # Use the recent load window; lowest index wins ties.
            loads = [self.load(i) for i in range(self.n)]
            replica = min(range(self.n), key=lambda i: loads[i])

        elif self.policy == "prefix_hash":
            # Route using the first full block's chained hash.
            hashes = block_hashes(prompt, self.block_size)
            replica = hashes[0] % self.n if hashes else 0

        elif self.policy == "cache_aware":
            # Determine the longest cached prefix on every replica
            # without updating cache statistics or LRU timestamps.
            hashes = block_hashes(prompt, self.block_size)
            cached_prefixes = []

            for cache in self.caches:
                match = 0

                for h in hashes:
                    if h not in cache.blocks:
                        break
                    match += 1

                cached_prefixes.append(match)

            # Choose the replica with the longest cached prefix.
            # Lowest index wins ties.
            replica = min(
                range(self.n),
                key=lambda i: (-cached_prefixes[i], i),
            )

            # Apply the load safeguard to the selected replica.
            loads = [self.load(i) for i in range(self.n)]
            mean_load = sum(loads) / self.n

            if (
                cached_prefixes[replica] == 0
                or (
                    mean_load > 0
                    and loads[replica] > 2 * mean_load
                )
            ):
                replica = min(range(self.n), key=lambda i: loads[i])

        else:
            raise ValueError(
                "policy must be one of: "
                "'round_robin', 'least_loaded', "
                "'prefix_hash', 'cache_aware'"
            )

        # Record the routing decision in the recent load window.
        self.recent.append(replica)

        if len(self.recent) > self.window:
            self.recent.pop(0)

        # Track total assignments for imbalance measurement.
        self.assigned[replica] += 1

        # Lookup before insertion so the current request only receives
        # credit for blocks that were already cached.
        self.last_cached = self.caches[replica].lookup(prompt)

        # Populate the selected replica's cache after lookup.
        self.caches[replica].insert(prompt)

        return replica


def simulate_routing(
    prompts,
    n_replicas,
    policy,
    capacity_blocks,
    block_size,
    prefill_tps,
    overhead_s,
):
    router = Router(
        n_replicas,
        policy,
        capacity_blocks,
        block_size,
    )

    ttfts = []

    for prompt in prompts:
        router.route(prompt)

        # Use the cached-token count produced by the chosen replica.
        ttfts.append(
            ttft_with_cache(
                len(prompt),
                router.last_cached,
                prefill_tps,
                overhead_s,
            )
        )

    # Combine hit/miss accounting across all replicas.
    total_hits = sum(cache.hits for cache in router.caches)
    total_misses = sum(cache.misses for cache in router.caches)
    total_accesses = total_hits + total_misses

    hit_rate = (
        total_hits / total_accesses
        if total_accesses > 0
        else 0.0
    )

    # Compute assignment imbalance from total requests assigned
    # to each replica.
    mean_assigned = sum(router.assigned) / n_replicas

    imbalance = (
        max(router.assigned) / mean_assigned
        if mean_assigned > 0
        else 0.0
    )

    mean_ttft = (
        sum(ttfts) / len(ttfts)
        if ttfts
        else 0.0
    )

    return {
        "hit_rate": round(hit_rate, 6),
        "imbalance": round(imbalance, 6),
        "mean_ttft": round(mean_ttft, 6),
    }

