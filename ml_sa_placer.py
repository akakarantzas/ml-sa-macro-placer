"""
ML-SA Macro Placer  v1
======================
Apostolos Kakarantzas - Partcl x Hudson River Trading Macro Placement Challenge 2026

Pipeline
--------
  1. Extract connectivity graph via plc.nets  (same API as will_seed)
  2. Graph Laplacian eigenvectors → spectral init (connectivity-aware positions)
  3. Legalisation → zero hard-macro overlaps
  4. Simulated Annealing:
       - SHIFT   : move one macro, O(N) overlap check, delta-WL accept/reject
       - SWAP    : exchange two macros (neighbour-biased 70% of the time)
       - ATTRACT : nudge macro toward a connected neighbour

Usage
-----
  uv run evaluate submissions/ml_sa_placer.py -b ibm01
  uv run evaluate submissions/ml_sa_placer.py --all
"""

import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Load PlacementCost object  (mirrors will_seed helper)
# ═══════════════════════════════════════════════════════════════════════════

def _load_plc(name: str):
    """Return PlacementCost for benchmark `name`, or None on failure."""
    from macro_place.loader import load_benchmark_from_dir, load_benchmark

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root).replace("\\", "/"))
        return plc

    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45":     "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    design = ng45.get(name)
    if design:
        base = (Path("external/MacroPlacement/Flows/NanGate45")
                / design / "netlist" / "output_CT_Grouping")
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(
                str(base / "netlist.pb.txt"),
                str(base / "initial.plc"),
            )
            return plc
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Edge extraction  (directly from plc.nets — proven API)
# ═══════════════════════════════════════════════════════════════════════════

def extract_edges(
    benchmark: Benchmark,
    plc,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    Build a weighted undirected graph over hard macros.

    Returns
    -------
    edges        : (E, 2) int32 — pairs of hard-macro tensor indices
    edge_weights : (E,)   float32 — connection strength
    neighbors    : list[list[int]] — adjacency list (tensor indices)
    """
    n_hard = benchmark.num_hard_macros

    name_to_bidx: Dict[str, int] = {}
    for bidx, plc_idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[plc_idx].get_name()] = bidx

    edge_dict: Dict[Tuple[int, int], float] = {}

    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) < 2:
            continue
        ml = sorted(macros)
        w = 1.0 / (len(ml) - 1)
        for i in range(len(ml)):
            for j in range(i + 1, len(ml)):
                pair = (ml[i], ml[j])
                edge_dict[pair] = edge_dict.get(pair, 0) + w

    if not edge_dict:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.float32),
            [[] for _ in range(n_hard)],
        )

    edges_np   = np.array(list(edge_dict.keys()),   dtype=np.int32)
    weights_np = np.array(list(edge_dict.values()), dtype=np.float32)

    neighbors: List[List[int]] = [[] for _ in range(n_hard)]
    for (i, j) in edge_dict:
        neighbors[i].append(j)
        neighbors[j].append(i)

    print(f"  {len(edges_np)} edges extracted from plc.nets")
    return edges_np, weights_np, neighbors


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Spectral initialisation
# ═══════════════════════════════════════════════════════════════════════════

def spectral_init(
    edges: np.ndarray,
    weights: np.ndarray,
    n: int,
    canvas_w: float,
    canvas_h: float,
    margin: float = 0.05,
) -> Optional[np.ndarray]:
    """
    Place macros using Fiedler vector + next eigenvector of graph Laplacian.
    Returns (n, 2) float32, or None on failure.
    """
    if len(edges) == 0:
        return None

    adj = np.zeros((n, n), dtype=np.float64)
    for (i, j), w in zip(edges, weights):
        adj[i, j] += w
        adj[j, i] += w

    degree = adj.sum(axis=1)
    if degree.sum() < 1e-9:
        return None

    L = np.diag(degree) - adj

    try:
        if n <= 700:
            from scipy.linalg import eigh
            _, vecs = eigh(L, subset_by_index=[0, 2])
            x_raw, y_raw = vecs[:, 1], vecs[:, 2]
        else:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            vals, vecs = eigsh(csr_matrix(L), k=3, which="SM", tol=1e-4)
            order = np.argsort(vals)
            x_raw, y_raw = vecs[:, order[1]], vecs[:, order[2]]

        def _scale(v: np.ndarray, dim: float) -> np.ndarray:
            lo, hi = v.min(), v.max()
            if abs(hi - lo) < 1e-9:
                return np.full_like(v, dim / 2)
            return ((v - lo) / (hi - lo)) * dim * (1 - 2 * margin) + dim * margin

        return np.stack(
            [_scale(x_raw, canvas_w), _scale(y_raw, canvas_h)], axis=1
        ).astype(np.float32)

    except Exception as exc:
        print(f"  [spectral] Failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Legalisation  (spiral search, from will_seed)
# ═══════════════════════════════════════════════════════════════════════════

def legalize(
    pos: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    n: int,
) -> np.ndarray:
    sep_x  = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y  = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    order  = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
    placed = np.zeros(n, dtype=bool)
    legal  = pos.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            conflict = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
            conflict[idx] = False
            if not conflict.any():
                placed[idx] = True
                continue

        step   = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        best_p = legal[idx].copy()
        best_d = float("inf")

        for r in range(1, 150):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = np.clip(pos[idx, 0] + dxm * step, half_w[idx], canvas_w - half_w[idx])
                    cy = np.clip(pos[idx, 1] + dym * step, half_h[idx], canvas_h - half_h[idx])
                    if placed.any():
                        dx = np.abs(cx - legal[:, 0])
                        dy = np.abs(cy - legal[:, 1])
                        conflict = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
                        conflict[idx] = False
                        if conflict.any():
                            continue
                    d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                    if d < best_d:
                        best_d, best_p, found = d, np.array([cx, cy]), True
            if found:
                break

        legal[idx] = best_p
        placed[idx] = True

    return legal


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Simulated Annealing
# ═══════════════════════════════════════════════════════════════════════════

def simulated_annealing(
    pos: np.ndarray,
    edges: np.ndarray,
    edge_weights: np.ndarray,
    neighbors: List[List[int]],
    movable: np.ndarray,
    sizes: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    steps: int,
    seed: int,
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)

    pos         = pos.copy()
    movable_idx = np.where(movable)[0]
    n_mov       = len(movable_idx)

    if n_mov == 0 or len(edges) == 0:
        return pos

    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

    def wl() -> float:
        dx = np.abs(pos[edges[:, 0], 0] - pos[edges[:, 1], 0])
        dy = np.abs(pos[edges[:, 0], 1] - pos[edges[:, 1], 1])
        return float((edge_weights * (dx + dy)).sum())

    def no_overlap(idx: int) -> bool:
        gap  = 0.05
        dx   = np.abs(pos[idx, 0] - pos[:, 0])
        dy   = np.abs(pos[idx, 1] - pos[:, 1])
        hits = (dx < sep_x[idx] + gap) & (dy < sep_y[idx] + gap)
        hits[idx] = False
        return not hits.any()

    cur_cost  = wl()
    best_cost = cur_cost
    best_pos  = pos.copy()

    T_start  = max(canvas_w, canvas_h) * 0.15
    T_end    = max(canvas_w, canvas_h) * 0.001
    accepted = 0

    for step in range(steps):
        frac = step / steps
        T    = T_start * (T_end / T_start) ** frac
        move_type = random.random()

        i      = int(random.choice(movable_idx))
        ox, oy = pos[i, 0], pos[i, 1]

        if move_type < 0.5:
            # ── SHIFT ─────────────────────────────────────────────────
            shift     = T * (0.3 + 0.7 * (1 - frac))
            pos[i, 0] = np.clip(ox + random.gauss(0, shift), half_w[i], canvas_w - half_w[i])
            pos[i, 1] = np.clip(oy + random.gauss(0, shift), half_h[i], canvas_h - half_h[i])

            if not no_overlap(i):
                pos[i, 0] = ox; pos[i, 1] = oy
                continue

            new_cost = wl()
            delta    = new_cost - cur_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                cur_cost = new_cost; accepted += 1
                if cur_cost < best_cost:
                    best_cost = cur_cost; best_pos = pos.copy()
            else:
                pos[i, 0] = ox; pos[i, 1] = oy

        elif move_type < 0.8:
            # ── SWAP ──────────────────────────────────────────────────
            nb = [j for j in neighbors[i] if movable[j]]
            j  = int(random.choice(nb if nb and random.random() < 0.7 else movable_idx))
            if i == j:
                continue

            ojx, ojy = pos[j, 0], pos[j, 1]
            pos[i, 0] = np.clip(ojx, half_w[i], canvas_w - half_w[i])
            pos[i, 1] = np.clip(ojy, half_h[i], canvas_h - half_h[i])
            pos[j, 0] = np.clip(ox,  half_w[j], canvas_w - half_w[j])
            pos[j, 1] = np.clip(oy,  half_h[j], canvas_h - half_h[j])

            if not no_overlap(i) or not no_overlap(j):
                pos[i, 0] = ox;  pos[i, 1] = oy
                pos[j, 0] = ojx; pos[j, 1] = ojy
                continue

            new_cost = wl()
            delta    = new_cost - cur_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                cur_cost = new_cost; accepted += 1
                if cur_cost < best_cost:
                    best_cost = cur_cost; best_pos = pos.copy()
            else:
                pos[i, 0] = ox;  pos[i, 1] = oy
                pos[j, 0] = ojx; pos[j, 1] = ojy

        else:
            # ── ATTRACT ───────────────────────────────────────────────
            if not neighbors[i]:
                continue
            j         = random.choice(neighbors[i])
            alpha     = random.uniform(0.05, 0.3)
            pos[i, 0] = np.clip(ox + alpha * (pos[j, 0] - ox), half_w[i], canvas_w - half_w[i])
            pos[i, 1] = np.clip(oy + alpha * (pos[j, 1] - oy), half_h[i], canvas_h - half_h[i])

            if not no_overlap(i):
                pos[i, 0] = ox; pos[i, 1] = oy
                continue

            new_cost = wl()
            delta    = new_cost - cur_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                cur_cost = new_cost; accepted += 1
                if cur_cost < best_cost:
                    best_cost = cur_cost; best_pos = pos.copy()
            else:
                pos[i, 0] = ox; pos[i, 1] = oy

        if (step + 1) % 2000 == 0:
            pct = accepted / (step + 1) * 100
            print(
                f"    step {step+1:6d}/{steps}"
                f"  cost={cur_cost:10.2f}  best={best_cost:10.2f}"
                f"  T={T:.4f}  acc={pct:.1f}%"
            )

    print(f"    SA done — best={best_cost:.2f}  ({accepted}/{steps} = {accepted/steps*100:.1f}% acc)")
    return best_pos


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Public Placer class
# ═══════════════════════════════════════════════════════════════════════════

class Placer:
    """
    Spectral-Init + SA macro placer.

    Parameters
    ----------
    sa_steps : SA iterations (10 000 ≈ 10–15 s per benchmark)
    seed     : random seed
    """

    def __init__(self, sa_steps: int = 10_000, seed: int = 42):
        self.sa_steps = sa_steps
        self.seed     = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        try:
            t0 = time.time()
            random.seed(self.seed)
            np.random.seed(self.seed)

            name     = benchmark.name
            n_hard   = benchmark.num_hard_macros
            canvas_w = float(benchmark.canvas_width)
            canvas_h = float(benchmark.canvas_height)

            print(f"\n{'═'*60}")
            print(f"  {name}  |  {n_hard} hard macros  |  {canvas_w:.1f}×{canvas_h:.1f} μm")
            print(f"{'═'*60}")

            pos_np   = benchmark.macro_positions.numpy().copy().astype(np.float64)
            sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
            fixed_np = benchmark.macro_fixed.numpy()

            hard_pos   = pos_np[:n_hard].copy()
            hard_sizes = sizes_np[:n_hard]
            hard_fixed = fixed_np[:n_hard]
            hard_mov   = ~hard_fixed
            half_w     = hard_sizes[:, 0] / 2
            half_h     = hard_sizes[:, 1] / 2

            # ── 1. Connectivity ──────────────────────────────────────────
            print(f"\n[1/4] Loading connectivity …")
            plc       = _load_plc(name)
            edges     = np.zeros((0, 2), dtype=np.int32)
            weights   = np.zeros(0,      dtype=np.float32)
            neighbors = [[] for _ in range(n_hard)]

            if plc is not None:
                edges, weights, neighbors = extract_edges(benchmark, plc)
            else:
                print("  WARNING: could not load plc — spectral init and SA disabled.")

            # ── 2. Spectral init ─────────────────────────────────────────
            print(f"\n[2/4] Spectral initialisation …")
            if len(edges) > 0:
                spec = spectral_init(edges, weights, n_hard, canvas_w, canvas_h)
                if spec is not None:
                    hard_pos[hard_mov] = spec[hard_mov].astype(np.float64)
                    hard_pos[:, 0] = np.clip(hard_pos[:, 0], half_w, canvas_w - half_w)
                    hard_pos[:, 1] = np.clip(hard_pos[:, 1], half_h, canvas_h - half_h)
                    print(f"  Applied to {int(hard_mov.sum())} movable macros.")
                else:
                    print("  Spectral failed — keeping initial positions.")
            else:
                print("  Skipped (no edges).")

            # ── 3. Legalisation ──────────────────────────────────────────
            print(f"\n[3/4] Legalisation …")
            hard_pos = legalize(
                hard_pos, hard_mov, hard_sizes, half_w, half_h,
                canvas_w, canvas_h, n_hard,
            )

            sep_x = (hard_sizes[:, 0:1] + hard_sizes[:, 0:1].T) / 2
            sep_y = (hard_sizes[:, 1:2] + hard_sizes[:, 1:2].T) / 2

            def count_overlaps(p):
                total = 0
                for a in range(len(p)):
                    dx   = np.abs(p[a, 0] - p[:, 0])
                    dy   = np.abs(p[a, 1] - p[:, 1])
                    hits = (dx < sep_x[a]) & (dy < sep_y[a])
                    hits[a] = False
                    total   += hits.sum()
                return total // 2

            print(f"  Overlaps after legalisation: {count_overlaps(hard_pos)}")

# ── 4. SA ────────────────────────────────────────────────────
            time_budget = 45.0
            steps_done  = 0
            sa_start    = time.time()
            chunk       = 2000
            print(f"\n[4/4] Simulated Annealing (budget: {time_budget}s) …")
            if len(edges) > 0:
                while (time.time() - sa_start) < time_budget:
                    hard_pos = simulated_annealing(
                        pos          = hard_pos,
                        edges        = edges,
                        edge_weights = weights,
                        neighbors    = neighbors,
                        movable      = hard_mov,
                        sizes        = hard_sizes,
                        half_w       = half_w,
                        half_h       = half_h,
                        canvas_w     = canvas_w,
                        canvas_h     = canvas_h,
                        steps        = chunk,
                        seed         = self.seed + steps_done,
                    )
                    steps_done += chunk
                    print(f"    {steps_done} steps  {time.time()-sa_start:.1f}s elapsed")
            else:
                print("  Skipped (no edges).")

            # ── 5. Final legalisation (removes any residual overlaps) ────
            print(f"\n[5/5] Final legalisation …")
            hard_pos = legalize(
                hard_pos, hard_mov, hard_sizes, half_w, half_h,
                canvas_w, canvas_h, n_hard,
            )
            ov_final = count_overlaps(hard_pos)
            print(f"  Final overlaps: {ov_final}")

            # ── 6. Return final positions ───────────────────────────────
            pos_np[:n_hard] = hard_pos
            elapsed = time.time() - t0
            print(f"\n  Done in {elapsed:.1f}s\n{'═'*60}\n")
            return torch.tensor(pos_np, dtype=torch.float32)
        except Exception as e:
            import traceback
            print(f"\n  ERROR in place(): {e}")
            traceback.print_exc()
            print("  Returning initial positions as fallback.")
            return benchmark.macro_positions.clone()
MLSAPlacer = Placer