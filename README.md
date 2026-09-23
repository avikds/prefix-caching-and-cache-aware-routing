# Prefix Caching and Cache-Aware Routing

Chapter 5.3 of Inference Engineering as a working system. Build the block-hash prefix cache that inference engines use to reuse KV state across requests: chained hashes over fixed-size token blocks, longest-prefix lookup, LRU eviction under a block budget, and hit-rate accounting. Price a hit in time to first token and in bytes, then generate the workloads that decide whether caching matters at all, a chat application with a shared system prompt, multi-turn conversations that resend their history, retrieval prompts that shuffle documents, and unrelated traffic, and measure what each one gets. Scale out: give every replica its own cache and compare round-robin, least-loaded, prefix-hash and cache-aware routing on hit rate, load balance and latency, the routing problem chapter 4.4's orchestration layer exists to solve. Add a CPU tier that keeps evicted blocks restorable at a bandwidth cost, and finish with the hit-rate-versus-capacity curve that tells you how much GPU memory to give the cache.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** PrefixCache
- [x] **2.** ttft_with_cache
- [x] **3.** workload
- [x] **4.** Router
- [x] **5.** TieredCache
- [x] **6.** hit_rate_vs_capacity

## Results

```
KV cache: 128 KB per token, 16-token blocks; 20 GB holds 9536 blocks

one replica, 4096-block cache, 300 requests each:
  chat       hit rate 88.4%  tokens saved 86.1%  TTFT   34.8 ms ->   22.1 ms
  multi_turn hit rate 95.0%  tokens saved 94.2%  TTFT   62.1 ms ->   22.5 ms
  rag        hit rate 44.3%  tokens saved 43.4%  TTFT   41.8 ms ->   32.3 ms
  unique     hit rate  0.0%  tokens saved  0.0%  TTFT   35.1 ms ->   35.1 ms

eight replicas on multi-turn traffic, 2048 blocks each:
  round_robin   hit rate 73.5%  imbalance 1.00  mean TTFT   34.5 ms
  least_loaded  hit rate 73.3%  imbalance 1.22  mean TTFT   34.6 ms
  prefix_hash   hit rate 96.2%  imbalance 3.10  mean TTFT   22.4 ms
  cache_aware   hit rate 95.4%  imbalance 1.16  mean TTFT   22.8 ms

64 GPU blocks under mixed traffic: plain cache 2768 hits; with a 4096-block CPU tier 2768 GPU + 416 CPU hits; total TTFT 11.79 s -> 11.50 s

hit rate vs capacity, multi-turn traffic:
      16 blocks    0.03 GB hit rate   0.3%
      64 blocks    0.13 GB hit rate   9.4%
     256 blocks    0.54 GB hit rate  45.5%
    1024 blocks    2.15 GB hit rate  95.1%
    4096 blocks    8.59 GB hit rate  95.1%
  knee: 1024 blocks
```
