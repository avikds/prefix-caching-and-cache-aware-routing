# Prefix Caching and Cache-Aware Routing

Chapter 5.3 of Inference Engineering as a working system. Build the block-hash prefix cache that inference engines use to reuse KV state across requests: chained hashes over fixed-size token blocks, longest-prefix lookup, LRU eviction under a block budget, and hit-rate accounting. Price a hit in time to first token and in bytes, then generate the workloads that decide whether caching matters at all, a chat application with a shared system prompt, multi-turn conversations that resend their history, retrieval prompts that shuffle documents, and unrelated traffic, and measure what each one gets. Scale out: give every replica its own cache and compare round-robin, least-loaded, prefix-hash and cache-aware routing on hit rate, load balance and latency, the routing problem chapter 4.4's orchestration layer exists to solve. Add a CPU tier that keeps evicted blocks restorable at a bandwidth cost, and finish with the hit-rate-versus-capacity curve that tells you how much GPU memory to give the cache.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** PrefixCache
- [x] **2.** ttft_with_cache

---

Built on Deep-ML.
