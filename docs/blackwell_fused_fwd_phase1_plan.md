# Phase 1 Plan: Eliminate the H-pipeline tax

> **Background**: see `blackwell_fused_fwd_perf_investigation.md` §11b. Cons-O NOOP measurement showed the H pipeline (cons-S + cons-V + producer) takes **3.37 µs/chunk** vs FLA's **1.46 µs/chunk** — a **1.91 µs structural tax** that is independent of cons-O.
>
> **Phase 1 target**: bring `per_chunk(cons-O NOOP) = 3.37 → ~2.0 µs`. With cons-O still in place, full per_chunk should drop from 5.04 → ~3.5 µs (gdr ~1.8 ms).

---

## 1. Why merging cons-S and cons-V is required

The 1.91 µs tax has four candidate sources:

a. Multi-WG mbarrier chain (4 barriers per chunk: data_is_ready, bar_1, bar_3, bar_5).
b. SMEM round-trip for `v_new` (cons-V writes `v_shared` in-place; cons-S reads `vn_shared`).
c. SMEM round-trip for `vn_shared` (cons-V writes; cons-S reads).
d. SMEM round-trip for `h_shared` (cons-S writes; cons-V reads).

**Item (a) cannot be removed without merging WGs.** Any pair of WGs must rendezvous on a barrier whenever there is RAW or WAW data dependency between them.

**Items (b)/(c)/(d) cannot be removed by passing fragments**, because TileLang's tcgen05 GEMM on SM100 only supports SS (SMEM×SMEM) and TS (TMEM×SMEM) variants — fragment-B is not implemented for tcgen05 on Blackwell. Cross-WG fragment sharing is also impossible by hardware definition.

**Conclusion**: All four can only be eliminated by collapsing cons-S + cons-V into a single warp group.

---

## 2. Current data flow (cons-S ↔ cons-V per chunk)

```
producer        cons-V (tx 128-256)              cons-S (tx 0-128)
────────        ─────────────────────             ──────────────────
load v_shared
load k_shared
load a_shared
load g_shared
data_is_ready ─→ wait
                 g_exp = exp2(g)
                 g_inv = exp2(-g)
                 g_rev = exp2(g[-1]-g)
                 ───── bar_1 ─────────  ←─────  T.copy(h_fragment → h_shared)
                                                arrive bar_1
                 wait bar_1
                 u = K^T @ h_shared          ← needs h_shared from cons-S
                 wait bar_3                  ← cons-O signals a_shared ready
                 v_shared -= g_exp * u       ← in-place SMEM write!
                 vd_shared = a_shared @ v_shared  (for cons-O)
                 v_shared(=vd) arrive bar_4
                 vn_shared = vd * g_rev      (for cons-S)
                 ───── bar_5 ─────────  ──→  wait bar_5
                                              h_fragment *= g_last
                                              arrive bar_5
                                              wait bar_5
                                              h_fragment += k^T @ vn_shared
                                              arrive data_is_free
```

**Key SMEM hops to eliminate**:
- `h_shared`: cons-S writes → cons-V reads (in `K @ h_shared`)
- `v_shared` in-place: cons-V reads raw v, writes `v - g_exp*u` back
- `vn_shared`: cons-V writes → cons-S reads (in `K^T @ vn_shared`)

`vd_shared` (for cons-O) must remain as long as cons-O is a separate WG.

---

## 3. Target data flow (after merge — single "cons-H" WG)

```
producer        cons-H (tx 0-128)                 cons-O (tx 128-256)
────────        ─────────────────────             ──────────────────────
load v_shared
load k_shared
load a_shared
load g_shared
data_is_ready ─→ wait
                 g_exp / g_inv / g_rev (in fragments, no SMEM)
                 u_fragment = K^T @ h_fragment   ← fully-on-fragment
                 wait bar_3                       ← cons-O signals a_shared ready
                 v_new_fragment = v_shared - g_exp * u_fragment
                 vd_fragment = a_shared @ v_new_fragment
                 T.copy(vd_fragment → vd_shared)  ← only kept hop, for cons-O
                 arrive bar_4
                 h_fragment *= g_last
                 vn_fragment = v_new_fragment * g_rev
                 h_fragment += k^T @ vn_fragment  ← fragment-fragment
                 arrive data_is_free
```

Eliminated:
- `h_shared` write from cons-S, `h_shared` read in cons-V's `K @ h_shared`.
- `v_shared` in-place write (we read raw v_shared and produce v_new on the fly).
- `vn_shared` write/read.
- Two barriers: `bar_1`, `bar_5` (no longer cross-WG dependency).

Kept:
- `vd_shared` for cons-O (TS GEMM `p @ vd` requires SMEM B).
- `bar_3` (cons-O signals a_shared writes done before cons-H reads in vd path).
- `bar_4` (cons-H signals vd_shared done for cons-O).

Open question: can the `K @ h_fragment` GEMM be done with h in fragment, given tcgen05 doesn't support RS? Two options:
1. **Use cuda.mma instead of tcgen05** for this specific GEMM — block_S=64, DK=128, block_DV=32 is small enough that mma.16x8x16 may be efficient.
2. **Keep h in TMEM**, use TS GEMM with h_tmem as A. This means the h state lives in TMEM, not in registers.

Decide empirically: try option 2 first (lower implementation cost; reuse the existing TMEM infrastructure from commit `n52xnmsioygtezfnrfmg`).

---

## 4. Implementation plan (3 sub-commits)

### Sub-commit 1: Move cons-V's body into cons-S, leave cons-V as an empty WG

The minimum step to validate that 128 threads can carry the work:
- cons-S (tx<128) does **all** of the H computation: g_exp/g_inv/g_rev, u, v_new, vd to SMEM, vn (fragment), h *= g_last, h += k^T @ vn.
- cons-V (tx 128-256) becomes an empty loop that arrives/waits on existing barriers (so cons-O's barrier protocol still works).
- Validate against `FLASHQLA_CONS_O_NOOP=1` per_chunk.

**Risk**: 128 threads might not be enough for the full GEMM workload.
**Pre-check**: tcgen05 atom on SM100 is 128×128 (one warp group). Our biggest GEMM is K^T @ vn = (DK=128) × (block_DV=32). That fits a single tcgen05.mma call cleanly.

### Sub-commit 2: Delete cons-V WG, drop dead barriers

Now that cons-S does all the work:
- Reduce thread count 512 → 384 (cons-S + cons-O + producer).
- Delete `bar_1`, `bar_5` (cons-V no longer participates).
- Delete `vn_shared` allocation.
- Delete `v_shared` in-place write — read from `v_shared`, materialize v_new in a fragment.

**SMEM saved**: vn_shared (8 KB). **Reg pressure**: more reg per cons-S thread (was 168, may need to grow to 200+).

### Sub-commit 3: Optimise the merged cons-H

Now we can iterate on:
- LDS swizzle for k_shared/v_shared (current LDS waste 1.57×).
- TMEM placement for h_fragment (option 2 from §3).
- Reduce num_stages if SMEM is the new bottleneck.

---

## 5. Validation methodology

After each sub-commit, run:

```bash
# Correctness
FLASHQLA_DISABLE_WG_REG=1 \
FLASHQLA_AUTOCP=0 \
FLASHQLA_BLACKWELL_NATIVE=1 \
FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE=1 \
python3 tests/test_gdr.py --set profile --nkh 8 --nvh 32 --skip-bwd

# H-pipeline timing (cons-O NOOP)
FLASHQLA_DISABLE_WG_REG=1 FLASHQLA_CONS_O_NOOP=1 \
FLASHQLA_AUTOCP=0 \
FLASHQLA_BLACKWELL_NATIVE=1 \
FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE=1 \
python3 -c "<T-scan script>"
```

**Pass criteria**:
- Sub-commit 1: cons-O NOOP per_chunk should be ≤ 3.37 µs (no regression). Full per_chunk may move ±10%.
- Sub-commit 2: cons-O NOOP per_chunk → ~2.5 µs (drop barriers + SMEM round-trip).
- Sub-commit 3: cons-O NOOP per_chunk → ~2.0 µs (target).

---

## 6. Risks and rollback

- **Single WG insufficient parallelism**: If 128 threads can't hide tensor-pipe latency without two WGs to ping-pong, sub-commit 1 itself will already show a slowdown. Rollback and consider keeping 2 WGs but eliminate just the SMEM hops via TMEM (more complex).
- **TMEM pressure**: if h, u, v all live in TMEM simultaneously, may exceed TMEM allocation. Only one of u/v can be in TMEM at a time, since they are produced and consumed within the same chunk loop body.
- **Regression to numerical correctness**: any change to v_shared in-place semantics needs a unit test. Start by adding a 64-token correctness regression that doesn't go through `--set profile`.

---

## 7. Out of scope for Phase 1

These are deferred to Phase 2 / 3:
- Extracting cons-O into a separate kernel (that's Phase 2).
- Changing the chunk-loop grid (would require stream-K).
- TMEM block-scaled MX-FP8 paths (orthogonal).
- bwd kernel (a separate effort entirely).

---

*Drafted 2026-05-20 — ready for execution in next thread session with fresh context.*

---

## 8. Execution log (2026-05-20 evening session)

### Sub-commit 1: cons-V body merged into cons-S (cons-V kept as empty barrier stub)

- Done: cons-S (tx<128) now owns the full H pipeline (`g_*_shared` compute, `u = K^T @ h_shared`, `v_new` in-place SMEM, `vd = a @ v_new`, `vn_shared = vd * g_rev`, `h *= g_last`, `h += K^T @ vn_shared`).
- cons-V (128≤tx<256) reduced to an empty stub that still arrives `bar_1` / `bar_5` / `data_is_free` and waits `bar_5` / `data_is_ready` to keep all `arrive_count`s valid.
- ⚠️ Deviation from §3 plan: `vn_shared` was **not** moved to fragment. Tcgen05 SS gemm requires SMEM B operand, and cons-S's `K^T @ vn` is SS gemm. `vn_shared` stays.
- ⚠️ Deviation from §3 plan: `v_new` was **not** kept in fragment. The in-place `v_shared` SMEM RAW write was preserved (as it was in cons-V); a previous attempt at fragment-B for the `a @ v_new` GEMM "broke correctness" per existing comments, so the safe SMEM round-trip stays.
- Result T=32k no-CP: gdr 2.347 → 2.367ms (≈0%, expected — sub1 is a scaffolding step, not a perf win on its own).
- Correctness: ✓ all four sequence lengths (4k/8k/16k/32k) pass.

### Sub-commit 2: cons-V WG deleted (512 → 384 threads, barrier counts tightened)

- Done: cons-V `elif` branch deleted. Layout now `tx<128` cons-S, `tx<256` cons-O, `tx<288/320/352/384` four 32-thread producer sub-groups.
- `bar_1` count 256→128 / `bar_3` 128 (unchanged) / `bar_4` 128 (unchanged) / `bar_5` 416→288 / `data_is_free` 384→256.
- ✅ Result T=32k:
  - **CP path: 1.408 → 1.275ms (-9.4%, speedup 0.71× → 0.75×)**
  - no-CP path: 2.367 → 2.367ms (~0%, untouched — bottleneck has moved out of H pipeline for no-CP)
  - **CONS_O_NOOP path: per_chunk 3.37 → 2.41 µs (-28%, speedup 1.03× FLA total)**
- Correctness: ✓ all four sequence lengths pass.
- **Sub-commit 2 was the big win for this phase**.

### Sub-commit 3: deferred / partially executed

§4 listed three follow-ups; here is what actually happened:

- **3-A (drop cons-S `wait bar_5`)**: ❌ Cannot do safely. `h_shared` is unstaged, and cons-S's next-iter `T.copy(h, h_shared)` would race with cons-O's cur-iter `Q @ h_shared`. The `data_is_free → data_is_ready` chain only fences the round-robin'd q/k/v/a buffers, not `h_shared`. Reverted before commit.
- **3-B (fuse `vd_shared` and `vn_shared` SMEM stores)**: ❌ Pseudo-optimization. Merging the two stores delays `arrive bar_4` by one pass, costing the cons-O P@Vd early-start window. Reverted before commit.
- **3-C (fuse three `g_*_shared` compute passes into one `T.Parallel(block_S)`)**: ⚠️ Applied. ±0% measured (within noise). TileLang likely already fused them at codegen time. Kept the change for code locality, no rollback needed.
- **LDS swizzle / h-TMEM / num_stages tuning**: not done.

### Phase C investigation: `num_stages` ∈ {3, 4}

Tested as a separate experiment to validate "producer stall is not a bottleneck":

| config | T=32k gdr | speedup | vs sub2 |
|---|---|---|---|
| CP, stages=2 (sub2 baseline) | 1.297ms | 0.75× | — |
| CP, stages=3 | 1.287ms | 0.75× | -0.8% (noise) |
| CP, stages=4 | hang | — | SMEM cap exceeded |
| no-CP, stages=3, block_DV=128 | 2.360ms | 0.61× | ~0% (noise) |
| no-CP, stages=3, block_DV=64 | 2.830ms | 0.53× | **+20% regression** |

- **`num_stages=3` produces no measurable benefit at any setting.** Producer DMA is not the bottleneck.
- **`num_stages=4` exceeds the SMEM-per-block cap and hangs.** Confirms the 147KB/block cap from NCU.
- **`block_DV=64` is strictly worse for no-CP** (CTA count doubles, fixed overhead per CTA dominates). This validates the existing `_select_block_dv` defaults.

### Net Phase 1 outcome

Starting from the previous "current best" (sub1 baseline: T=32k CP gdr 1.408ms / 0.71×), phase 1 landed:
- **CP path: 1.408 → 1.275ms (speedup 0.71× → 0.75×, -9%)**
- **CONS_O_NOOP path: 1.03× FLA total — H-only path no longer the bottleneck**
- no-CP path: untouched (the bottleneck there is now cons-O itself)

### Bottleneck has shifted (important)

After sub2, the dominant gap is no longer "H-pipeline tax". CONS_O_NOOP measurement shows cons-O alone contributes:
```
T=32k no-CP: full 2.36ms − NOOP 1.24ms = 1.12ms attributable to cons-O work
            per_chunk: 4.62µs − 2.41µs = 2.21µs/chunk attributable to cons-O
```
Phase 2 must address cons-O internals, not the H pipeline.

---

## 9. Phase 2 candidates (drafted at end of this session)

Three candidate directions, each independently worth a sub-commit:

### Candidate A: `h_shared` double-buffering (drop wait `bar_5`)
- Allocate `h_shared` as `(2, DK, block_DV)`. cons-S writes `h_shared[i_s%2]`, cons-O reads `h_shared[i_s%2]`.
- Drops the iter-end serialization: cons-S can start next iter's `K^T @ h` immediately after `arrive bar_5`, overlapping with cons-O's `P @ Vd`.
- SMEM cost: +16KB (block_DV=128) / +8KB (block_DV=64). Need to confirm we stay under the 147KB cap.
- Expected: T=32k no-CP 2.36 → ~2.0ms, speedup 0.62× → ~0.73×.
- **Risk**: SMEM cap. Must verify with NCU before committing.

### Candidate B: cons-O P-path uses RS-gemm (eliminate `p_shared`)
- Currently `T.gemm(p_shared, vd_shared, o_fragment)` is SS gemm.
- If TileLang supports RS-tcgen05 on SM100, can use `T.gemm(p_fragment, vd_shared, o_fragment)` directly, eliminating the `T.copy(p_fragment, p_shared)` round-trip.
- SMEM cost: -8KB (`p_shared` deleted).
- Expected: cons-O per_chunk 2.21 → ~1.5µs, full T=32k no-CP gdr 2.36 → ~2.0ms.
- **Risk**: TileLang RS-gemm support on SM100 unverified.

### Candidate C: `vd_shared` double-buffering
- Stage `vd_shared` so cons-S can write iter `i_s+1`'s vd while cons-O still reads iter `i_s`'s vd.
- Drops the `bar_4` per-iter rendezvous to a 2-iter-deep pipeline.
- SMEM cost: +16KB (block_DV=128) / +8KB (block_DV=64).
- Expected: 5–8% on the H path. Lower-leverage than A/B.

**Suggested ordering**: B first (least risky, smallest SMEM footprint, biggest leverage on the no-CP critical path), then A (if SMEM permits), then C only if 5% still matters.

---

## 10. NCU diagnostic methodology (HOW to decide what to do next)

Before committing any of A/B/C, validate the hypothesis with NCU. Below are the exact commands and what to look for. **All future optimization decisions must be backed by an NCU section.**

### 10.1 Capture a single-kernel profile

```bash
# pick a representative shape; T=32k no-CP gives the biggest signal because
# H-only path is already at 1.03×, so all remaining gap is cons-O / serial path.
export NCU_OUT=/tmp/flashqla_blackwell_native_T32k.ncu-rep

FLASHQLA_DISABLE_WG_REG=1 \
FLASHQLA_AUTOCP=0 \
FLASHQLA_BLACKWELL_NATIVE=1 \
FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE=1 \
ncu \
  --target-processes all \
  --kernel-name regex:'tilelang_fused_chunk_gdr_fwd_blackwell' \
  --launch-skip 5 --launch-count 1 \
  --set full \
  --import-source yes \
  -o ${NCU_OUT%.ncu-rep} \
  python3 tests/test_gdr.py --set profile --nkh 8 --nvh 32 --skip-bwd \
    --seqlen 32768 --hide-acc --hide-lat
```

`--launch-skip 5 --launch-count 1` skips the first 5 launches (warm-up / autotune) and captures one steady-state launch only. `--set full` enables every section we need (Memory Workload Analysis, Source Counters, Warp State, Scheduler).

### 10.2 Sections to read (in order of priority)

After opening the report (`ncu-ui ${NCU_OUT}` or `ncu --import ${NCU_OUT} -p ...`):

1. **GPU Speed of Light (Throughput section)**
   - "SM Throughput" vs "Memory Throughput": if SM<50% AND Memory<50%, kernel is barrier-bound (our hypothesis). If SM>80%, we are compute-bound and need algorithmic changes, not pipeline changes.
   - Tensor-core utilization: previously NCU showed ~8%. **Each new sub-commit should track this number.** A pipeline win without a TC% increase is suspicious.

2. **Memory Workload Analysis → Shared Memory Bank Conflicts** (validate Candidate A/B/C SMEM impact)
   - "Shared Store Bank Conflicts" + "Shared Load Bank Conflicts": if these are >20%, swizzle/layout work is the win, not buffering.
   - "Shared Memory Used per Block": confirms our 147KB hypothesis. If we're at 144KB and Candidate A adds 16KB → cap exceeded → A is dead.

3. **Warp State Statistics → Stall Reasons** (THIS IS THE ONE THAT CHOOSES A vs B vs C)
   - **`Stall Barrier`**: time waiting on `bar_*` and `data_is_*`. If this is the top stall reason, Candidate A wins (cons-S/O serialization).
   - **`Stall Long Scoreboard`**: time waiting on memory operations. If top, Candidate B (RS gemm killing `p_shared` round-trip) or partial staging wins.
   - **`Stall MIO Throttle` / `Stall Tex Throttle`**: pipeline pressure on LSU/TEX. SMEM bandwidth saturated → swizzle/widen loads.
   - **`Stall Wait`**: warp scheduler can't issue. If top, occupancy or instruction-mix issue, not pipeline.

4. **Source Counters (per-line breakdown, requires `--import-source yes`)**
   - Find the single hottest line in `fused_fwd_native.py`. If it's the `T.gemm` calls, we're compute-bound. If it's `T.barrier_wait`, we're sync-bound. If it's `T.copy`, we're memory-bound.

5. **Scheduler Statistics**
   - "Issued Warp Per Scheduler" vs "Eligible Warps Per Scheduler": gap = stall ratio. Confirms (3).

### 10.3 Cheap targeted runs (if `--set full` is too slow)

```bash
# stall reasons only — fast (~5x speed-up over --set full)
ncu --set roofline --section WarpStateStats --section LaunchStats \
    --kernel-name regex:'tilelang_fused_chunk_gdr_fwd_blackwell' \
    --launch-skip 5 --launch-count 1 \
    -o /tmp/flashqla_warp_state \
    <same-env-and-cmd-as-above>

# SMEM only — confirm the 147KB cap before allocating a new staged buffer
ncu --section MemoryWorkloadAnalysis --section LaunchStats \
    --kernel-name regex:'tilelang_fused_chunk_gdr_fwd_blackwell' \
    --launch-skip 5 --launch-count 1 \
    -o /tmp/flashqla_smem \
    <same-env-and-cmd-as-above>
```

### 10.4 Decision rule for Phase 2 sub-commits

After capturing 10.1, look at `Stall Barrier` and `Shared Memory Used per Block`:

- If `Stall Barrier` > 30% of total stall time → **do Candidate A** (h_shared double buffer).
- Else if `Stall Long Scoreboard` > 30% AND `Shared Memory Used per Block` ≤ 130KB → **do Candidate B** (RS gemm); the extra 8KB headroom from removing `p_shared` is gravy.
- Else if `Shared Memory Bank Conflicts` > 20% → swizzle work first (orthogonal sub-commit).
- Else (no clear winner in NCU) → re-examine assumptions. Do not commit blindly.

The previous sessions skipped this step and hit two dead-ends (sub-commit 3-A and 3-B). All future optimization rounds must produce an NCU snapshot before code changes, and that snapshot lives in this file or a sibling `phase2_plan.md`.

