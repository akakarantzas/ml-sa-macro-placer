[README.md](https://github.com/user-attachments/files/26523647/README.md)
# ML-SA Macro Placer - Partcl x Hudson River Trading Macro Placement Challenge 2026

A macro placement engine built for the **Partcl x Hudson River Trading Macro Placement Challenge 2026**.

---

## What It Does

The placer positions hard macros on a chip floorplan to minimise the proxy cost:

```
Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion
```

It runs on all 17 IBM ICCAD04 benchmarks and produces placements with 0 overlaps within the 1-hour runtime limit.

---

## Algorithm

The approach combines two techniques: spectral graph initialisation and simulated annealing.

**1. Spectral Graph Initialisation**

Instead of starting from a random or greedy layout, I extract the circuit's connectivity graph from `plc.nets` and compute the graph Laplacian. The Fiedler vector (second eigenvector) and the next eigenvector encode the graph's structure and are used to assign macros initial positions that respect connectivity - highly connected macros start closer together. This gives simulated annealing a much better starting point.

**2. Simulated Annealing**

After spectral initialization, I used Simulated Annealing (SA) to improve the placement. The algorithm applies three types of moves:
- **SHIFT** - moves a single macro to a nearby position, scaled by current temperature
- **SWAP** - swaps two macros with 70% chance of targeting a connected neighbour
- **ATTRACT** - slightly moves a macro closer to one of its connected neighbours

Moves are accepted or rejected using the standard Metropolis criterion. Temperature decays exponentially from `T_start = canvas × 0.15` to `T_end = canvas × 0.001` over a configurable time budget.

**3. Legalisation**

After SA, a two-stage legalisation pass removes any remaining overlaps:

*Spiral search* - greedily places macros (largest first) by searching outward for the nearest valid position
*Push-apart fallback* - a simple method inspired by physics that resolves remaining overlaps by treating macros as repelling objects

---

## Results

Evaluated on all 17 IBM ICCAD04 benchmarks with a 45-second SA time budget per benchmark.

| Benchmark | Proxy | vs SA Baseline | vs RePlAce | Overlaps |
|-----------|-------|----------------|------------|----------|
| ibm01 | 1.6261 | -23.5% | -63.0% | 0 |
| ibm02 | 1.7794 | +6.7% | +3.1% | 0 |
| ibm03 | 1.9216 | -10.4% | -45.3% | 0 |
| ibm04 | 1.8487 | -22.9% | -41.9% | 0 |
| ibm06 | 2.1944 | +12.4% | -35.6% | 0 |
| ibm07 | 1.9310 | +4.5% | -32.0% | 0 |
| ibm08 | 2.3128 | -20.2% | -61.9% | 0 |
| ibm09 | 1.5445 | -11.3% | -38.0% | 0 |
| ibm10 | 1.9364 | +8.3% | -29.0% | 0 |
| ibm11 | 1.7657 | -3.2% | -50.0% | 0 |
| ibm12 | 2.5448 | +10.0% | -47.4% | 0 |
| ibm13 | 1.9350 | -1.1% | -44.9% | 0 |
| ibm14 | 2.0693 | +9.0% | -34.1% | 0 |
| ibm15 | 1.7885 | +22.2% | -18.0% | 0 |
| ibm16 | 2.2106 | +1.0% | -49.6% | 0 |
| ibm17 | 2.3111 | +37.1% | -40.5% | 0 |
| ibm18 | 1.9393 | +30.1% | -9.4% | 0 |
| **AVG** | **1.9800** | **+6.8%** | **-35.8%** | **0** |

- Beats the SA baseline (2.1251) on **11 out of 17 benchmarks**
- Zero overlaps across all benchmarks
- Total runtime: ~1816 seconds for all 17 benchmarks (~107 seconds per benchmark average)

---

## Setup

```bash
# Clone the competition repository
git clone https://github.com/partcleda/macro-place-challenge-2026.git
cd macro-place-challenge-2026

# Initialize submodules
git submodule update --init external/MacroPlacement

# Install dependencies
uv sync

# Copy the placer
cp ml_sa_placer.py submissions/ml_sa_placer.py

# Run on a single benchmark
uv run evaluate submissions/ml_sa_placer.py -b ibm01

# Run on all 17 benchmarks
uv run evaluate submissions/ml_sa_placer.py --all
```

---

## Dependencies

- Python 3.10+, NumPy, SciPy, PyTorch, TILOS MacroPlacement evaluator (via competition submodule)

---

## File Structure

```
ml_sa_placer.py
├── Section 1 - PlacementCost loader (_load_plc)
├── Section 2 - Edge extraction from plc.nets
├── Section 3 - Spectral initialisation
├── Section 4 - Legalisation (spiral search + push-apart)
├── Section 5 - Simulated annealing (SHIFT / SWAP / ATTRACT)
└── Section 6 - Placer class (public entry point)
```

---

## Notes

This was my first time working on an Electronic Design Automation problem. I had no prior knowledge of macro placement or chip design before this competition. The spectral initialisation idea came from reading about graph partitioning and thinking it could give SA a better starting point than random placement, which happened to be true in most cases.

---

## Author

**[Apostolos Kakarantzas](https://www.linkedin.com/in/akakarantzas/)**
