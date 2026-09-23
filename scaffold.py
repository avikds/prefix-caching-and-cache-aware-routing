"""
Prefix Caching and Cache-Aware Routing scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Prefix Caching and Cache-Aware Routing (Inference Engineering, chapter 5.3).

Story: build a block-hash prefix cache with LRU eviction; price a hit in time to
first token; run four workloads through it; route across replicas four ways;
add a CPU tier; and size the cache from the hit-rate-versus-capacity curve.
"""
import numpy as np


def main() -> None:
    block_size, kv_per_token, prefill_tps, overhead = 16, 131072, 20000, 0.02
    print(f"KV cache: {kv_per_token / 1024:.0f} KB per token, {block_size}-token blocks; 20 GB holds {blocks_for(20e9, block_size, kv_per_token)} blocks")

    # ---- 1. One replica, four workloads ----
    print("\none replica, 4096-block cache, 300 requests each:")
    for kind in ("chat", "multi_turn", "rag", "unique"):
        prompts = workload(kind, 300, np.random.default_rng(0))
        r = simulate_cache(prompts, PrefixCache(4096, block_size), prefill_tps, overhead)
        print(f"  {kind:10s} hit rate {r['hit_rate']:5.1%}  tokens saved {r['fraction_saved']:5.1%}  TTFT {r['mean_ttft_nocache'] * 1000:6.1f} ms -> {r['mean_ttft'] * 1000:6.1f} ms")

    # ---- 2. Four replicas, four policies ----
    prompts = workload("multi_turn", 400, np.random.default_rng(1))
    print("\neight replicas on multi-turn traffic, 2048 blocks each:")
    for policy in ("round_robin", "least_loaded", "prefix_hash", "cache_aware"):
        r = simulate_routing(prompts, 8, policy, 2048, block_size, prefill_tps, overhead)
        print(f"  {policy:13s} hit rate {r['hit_rate']:5.1%}  imbalance {r['imbalance']:.2f}  mean TTFT {r['mean_ttft'] * 1000:6.1f} ms")

    # ---- 3. A CPU tier under memory pressure ----
    mixed = workload("chat", 200, np.random.default_rng(2)) + workload("unique", 200, np.random.default_rng(3))
    order = np.random.default_rng(4).permutation(len(mixed))
    plain, tiered = PrefixCache(64, block_size), TieredCache(64, 4096, block_size, block_size * kv_per_token, 20e9)
    t_plain = t_tiered = 0.0
    for k in order:
        p = mixed[int(k)]
        c = plain.lookup(p); plain.insert(p); t_plain += ttft_with_cache(len(p), c, prefill_tps, overhead)
        c2, rs = tiered.lookup(p); tiered.insert(p); t_tiered += ttft_tiered(len(p), c2, rs, prefill_tps, overhead)
    st = tier_stats(tiered)
    print(f"\n64 GPU blocks under mixed traffic: plain cache {plain.hits} hits; with a 4096-block CPU tier {st['gpu_hits']} GPU + {st['cpu_hits']} CPU hits; "
          f"total TTFT {t_plain:.2f} s -> {t_tiered:.2f} s")

    # ---- 4. Sizing ----
    print("\nhit rate vs capacity, multi-turn traffic:")
    for line in cache_report(workload("multi_turn", 300, np.random.default_rng(5)), [16, 64, 256, 1024, 4096], block_size, kv_per_token):
        print("  " + line)


if __name__ == "__main__":
    main()

