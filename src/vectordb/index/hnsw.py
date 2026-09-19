"""HNSW index. Design doc, Section 2/3 — insert implemented Day 8, search
implemented Day 9. Not wired into api/ yet: BruteForceIndex still serves
/search until search() is real (see hnsw.search below).
"""
import heapq
from pathlib import Path

import numpy as np

from vectordb.config import settings


class HnswIndex:
    def __init__(self, dim: int, m: int = settings.m, m_l0: int = settings.m_l0,
                 ef_construction: int = settings.ef_construction,
                 rng_seed: int = settings.rng_seed):
        self.dim = dim
        self.m = m
        self.m_l0 = m_l0
        self.ef_construction = ef_construction
        self.ml = 1.0 / np.log(m)
        self.rng = np.random.default_rng(rng_seed)

        self._capacity = 0
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.id_to_slot: dict[int, int] = {}
        self.slot_to_id: list[int] = []
        self.free_slots: list[int] = []
        self.tombstoned: np.ndarray = np.empty(0, dtype=bool)
        self.levels: list[int] = []
        self.layers: list[dict[int, list[int]]] = []  # layers[lc][slot] -> neighbour slots
        self.entry_point: int = -1

    @property
    def count(self) -> int:
        return len(self.id_to_slot)

    # ---------- capacity / slot management ----------

    def _ensure_capacity(self, min_size: int) -> None:
        if min_size <= self._capacity:
            return
        new_capacity = max(min_size, max(64, self._capacity * 2))
        new_vectors = np.zeros((new_capacity, self.dim), dtype=np.float32)
        new_vectors[: len(self.vectors)] = self.vectors
        self.vectors = new_vectors
        new_tombstoned = np.zeros(new_capacity, dtype=bool)
        new_tombstoned[: len(self.tombstoned)] = self.tombstoned
        self.tombstoned = new_tombstoned
        self._capacity = new_capacity

    def _alloc_slot(self) -> int:
        if self.free_slots:
            return self.free_slots.pop()
        slot = len(self.slot_to_id)
        self._ensure_capacity(slot + 1)
        self.slot_to_id.append(-1)  # placeholder, set by caller
        self.levels.append(0)
        return slot

    # ---------- distance helpers ----------

    def _distance_to_query(self, query: np.ndarray, slot: int) -> float:
        return 1.0 - float(np.dot(query, self.vectors[slot]))

    def _distance_between(self, a: int, b: int) -> float:
        return 1.0 - float(np.dot(self.vectors[a], self.vectors[b]))

    # ---------- core HNSW search (used by insert AND, from Day 9, by query search) ----------

    def _search_layer(self, query: np.ndarray, entry_points: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """Best-first search within one layer. Returns up to `ef` (distance, slot)
        pairs ascending, excluding tombstoned slots from the RESULT set — but
        still traversing through tombstoned nodes' edges for connectivity
        (design doc's tombstone rule, applied uniformly to insert-time and
        query-time search since they share this routine).
        """
        visited: set[int] = set(entry_points)
        candidates: list[tuple[float, int]] = []  # min-heap
        results: list[tuple[float, int]] = []  # max-heap via negated distance

        for ep in entry_points:
            d = self._distance_to_query(query, ep)
            heapq.heappush(candidates, (d, ep))
            if not self.tombstoned[ep]:
                heapq.heappush(results, (-d, ep))

        layer_graph = self.layers[layer]

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            if results and dist_c > -results[0][0] and len(results) >= ef:
                break
            for neighbor in layer_graph.get(c, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                dist_n = self._distance_to_query(query, neighbor)
                furthest = -results[0][0] if results else float("inf")
                if len(results) < ef or dist_n < furthest:
                    heapq.heappush(candidates, (dist_n, neighbor))
                    if not self.tombstoned[neighbor]:
                        heapq.heappush(results, (-dist_n, neighbor))
                        if len(results) > ef:
                            heapq.heappop(results)

        return sorted([(-d, s) for d, s in results])

    def _select_neighbors_heuristic(self, candidates: list[tuple[float, int]], m: int) -> list[tuple[float, int]]:
        """Algorithm 4 (Malkov & Yashunin): prefer diverse neighbours over
        naive closest-M — a candidate is kept only if it's closer to the new
        node than to every neighbour already selected, otherwise it's parked
        and only used to fill remaining slots if the diverse set runs short.
        """
        working = sorted(candidates)
        selected: list[tuple[float, int]] = []
        discarded: list[tuple[float, int]] = []

        for dist_e, e in working:
            if len(selected) >= m:
                break
            good = True
            for _, r in selected:
                if self._distance_between(e, r) < dist_e:
                    good = False
                    break
            if good:
                selected.append((dist_e, e))
            else:
                discarded.append((dist_e, e))

        for dist_e, e in discarded:
            if len(selected) >= m:
                break
            selected.append((dist_e, e))

        return selected

    def _connect(self, slot: int, layer: int, neighbours: list[tuple[float, int]]) -> None:
        max_degree = self.m_l0 if layer == 0 else self.m
        self.layers[layer].setdefault(slot, [])
        self.layers[layer][slot] = [s for _, s in neighbours]

        for _, nb in neighbours:
            nb_list = self.layers[layer].setdefault(nb, [])
            if slot not in nb_list:
                nb_list.append(slot)
            if len(nb_list) > max_degree:
                cands = [(self._distance_between(nb, other), other) for other in nb_list]
                pruned = self._select_neighbors_heuristic(cands, max_degree)
                self.layers[layer][nb] = [s for _, s in pruned]

    # ---------- public interface ----------

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        for i, id_ in enumerate(ids):
            if id_ in self.id_to_slot:
                raise ValueError(f"id {id_} already exists")
            self._add_one(id_, vectors[i].astype(np.float32))

    def _add_one(self, id_: int, vector: np.ndarray) -> None:
        slot = self._alloc_slot()
        self.vectors[slot] = vector
        self.slot_to_id[slot] = id_
        self.id_to_slot[id_] = slot
        self.tombstoned[slot] = False

        level = int(-np.log(self.rng.uniform()) * self.ml)
        self.levels[slot] = level

        if self.entry_point == -1:
            self.entry_point = slot
            for lc in range(level + 1):
                if lc >= len(self.layers):
                    self.layers.append({})
                self.layers[lc].setdefault(slot, [])
            return

        cur_top = len(self.layers) - 1
        ep = [self.entry_point]

        # greedy single-nearest descent from cur_top down to level+1
        for lc in range(cur_top, level, -1):
            nearest = self._search_layer(vector, ep, ef=1, layer=lc)
            if nearest:
                ep = [nearest[0][1]]

        # insert into layers min(level, cur_top) down to 0
        for lc in range(min(level, cur_top), -1, -1):
            if lc >= len(self.layers):
                self.layers.append({})
            candidates = self._search_layer(vector, ep, ef=self.ef_construction, layer=lc)
            max_degree = self.m_l0 if lc == 0 else self.m
            selected = self._select_neighbors_heuristic(candidates, max_degree)
            self._connect(slot, lc, selected)
            ep = [s for _, s in selected] if selected else ep

        if level > cur_top:
            for lc in range(cur_top + 1, level + 1):
                if lc >= len(self.layers):
                    self.layers.append({})
                self.layers[lc].setdefault(slot, [])
            self.entry_point = slot

    def search(self, query: np.ndarray, k: int, ef: int) -> list[tuple[int, float]]:
        if self.entry_point == -1:
            return []

        cur_top = len(self.layers) - 1
        ep = [self.entry_point]

        # greedy single-nearest descent from the top layer down to layer 1 —
        # identical to insert's descent phase, same routine.
        for lc in range(cur_top, 0, -1):
            nearest = self._search_layer(query, ep, ef=1, layer=lc)
            if nearest:
                ep = [nearest[0][1]]

        candidates = self._search_layer(query, ep, ef=max(ef, k), layer=0)
        top_k = candidates[:k]
        return [(self.slot_to_id[slot], dist) for dist, slot in top_k]

    def delete(self, ids: list[int]) -> None:
        for id_ in ids:
            slot = self.id_to_slot.pop(id_, None)
            if slot is None:
                continue  # unknown id — ignored silently, per contract
            self.tombstoned[slot] = True
            self.free_slots.append(slot)

            if slot == self.entry_point:
                # Walk layer-0 neighbours for the first live one. None found
                # (last node) -> entry_point = -1, reclaimed on next add().
                new_entry = -1
                for neighbour in self.layers[0].get(slot, []):
                    if not self.tombstoned[neighbour]:
                        new_entry = neighbour
                        break
                self.entry_point = new_entry

    def to_arrays(self) -> dict:
        raise NotImplementedError("Day 11")

    @classmethod
    def from_arrays(cls, data: dict) -> "HnswIndex":
        raise NotImplementedError("Day 11")

    def save(self, path: Path) -> None:
        raise NotImplementedError("delegates to storage/snapshot.py — Day 11")

    def load(self, path: Path) -> None:
        raise NotImplementedError("delegates to storage/snapshot.py — Day 11")
