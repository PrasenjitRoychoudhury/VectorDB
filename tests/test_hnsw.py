import numpy as np
import pytest

from vectordb.index.hnsw import HnswIndex


def _unit(rng, n, dim):
    v = rng.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_single_insert_sets_entry_point():
    idx = HnswIndex(dim=8, rng_seed=42)
    v = _unit(np.random.default_rng(1), 1, 8)
    idx.add([1], v)
    assert idx.entry_point != -1
    assert idx.count == 1
    assert idx.slot_to_id[idx.entry_point] == 1


def test_duplicate_id_raises():
    idx = HnswIndex(dim=8, rng_seed=42)
    v = _unit(np.random.default_rng(1), 1, 8)
    idx.add([1], v)
    with pytest.raises(ValueError):
        idx.add([1], v)


def test_bulk_insert_layer0_degree_bound():
    rng = np.random.default_rng(7)
    idx = HnswIndex(dim=16, m=16, m_l0=32, ef_construction=100, rng_seed=42)
    n = 300
    vecs = _unit(rng, n, 16)
    idx.add(list(range(n)), vecs)

    assert idx.count == n
    assert idx.entry_point != -1
    # layer 0 must exist and hold every inserted node
    assert 0 in range(len(idx.layers))
    assert len(idx.layers[0]) == n
    # degree bound at layer 0 is m_l0 (allowing the diversification
    # heuristic's temporary over-connect before a neighbour's own prune runs
    # is not expected here since _connect prunes on every insert)
    for slot, neighbours in idx.layers[0].items():
        assert len(neighbours) <= idx.m_l0, f"slot {slot} has {len(neighbours)} > m_l0"


def test_graph_is_connected_at_layer0():
    """Every node must be reachable from the entry point via layer-0 edges —
    a disconnected graph silently returns wrong/empty search results."""
    rng = np.random.default_rng(11)
    idx = HnswIndex(dim=16, m=8, m_l0=16, ef_construction=64, rng_seed=42)
    n = 150
    vecs = _unit(rng, n, 16)
    idx.add(list(range(n)), vecs)

    # BFS over layer 0 from the entry point
    layer0 = idx.layers[0]
    seen = {idx.entry_point}
    frontier = [idx.entry_point]
    while frontier:
        nxt = []
        for node in frontier:
            for nb in layer0.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    nxt.append(nb)
        frontier = nxt

    assert len(seen) == n, f"only {len(seen)}/{n} nodes reachable from entry point"


def test_recall_against_brute_force():
    """Sanity check ahead of the real Day 9 gate — not the formal recall@10
    test (that needs the query-time search() this doc's Day 9 note defers),
    but confirms _search_layer at layer 0 alone finds true nearest neighbours
    reasonably well on a small set, which is what insert-time connectivity
    depends on."""
    from vectordb.index.brute_force import BruteForceIndex

    rng = np.random.default_rng(99)
    n, dim = 500, 32
    vecs = _unit(rng, n, dim)

    idx = HnswIndex(dim=dim, m=16, m_l0=32, ef_construction=200, rng_seed=42)
    idx.add(list(range(n)), vecs)

    bf = BruteForceIndex(dim=dim)
    bf.add(list(range(n)), vecs)

    queries = _unit(rng, 20, dim)
    hits = 0
    total = 0
    for q in queries:
        true_top10 = {i for i, _ in bf.search(q, k=10, ef=0)}
        approx = idx._search_layer(q, [idx.entry_point], ef=64, layer=0)
        approx_top10 = {s for _, s in approx[:10]}
        hits += len(true_top10 & approx_top10)
        total += 10

    recall = hits / total
    assert recall > 0.7, f"recall {recall:.2f} too low for layer-0 search alone (expected, not the Day 9 gate)"
