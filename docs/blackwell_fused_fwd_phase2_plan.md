# Phase 2 Plan: Beyond the H-pipeline tax

> **Predecessor**: see `blackwell_fused_fwd_phase1_plan.md` §8–§10 for the Phase 1 execution log and NCU-driven decision methodology.
>
> **Current state (end of Phase 1, branch `hyq/blackwell`)**:
> - Layout: `tx<128` cons-S (full H pipeline), `tx<256` cons-O, `tx<288/320/352/384` four 32-thread producer sub-WGs. Total 384 threads.
> - T=32k results: CP gdr 1.275ms / 0.75× FLA, no-CP gdr 2.36ms / 0.62× FLA, NOOP no-CP gdr 1.24ms / **1.03× FLA total** (H-only path is no longer the bottleneck).
> - Bottleneck has shifted to cons-O internals on the no-CP path.
>
> **Phase 2 target**: bring no-CP T=32k from 0.62× → 0.85×+ FLA total by attacking cons-O directly.

---

## 1. Confirm bottleneck with NCU before touching code

**Hard rule established at the end of Phase 1**: no optimization is committed without an NCU section that justifies it. See `blackwell_fused_fwd_phase1_plan.md` §10 for full methodology.

### Step 1: capture a baseline profile

**NCU command (corrected for Blackwell metric naming, 2026-05-20)**:

```bash
# Throughput / tensor / bank conflicts (verified working on SM100)
FLASHQLA_DISABLE_WG_REG=1 FLASHQLA_AUTOCP=0 \
FLASHQLA_BLACKWELL_NATIVE=1 FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE=1 \
ncu \
  --target-processes all \
  --kernel-name 'regex:tilelang_fused_chunk_gdr_fwd_blackwell.*' \
  --launch-count 3 \
  --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
  python3 tests/test_gdr.py --set profile --nkh 8 --nvh 32 --skip-bwd \
    --seqlen 32768 --hide-acc --hide-lat

# Stall reasons (TODO next session: verify these metric names work on local NCU
#   version; on this machine the older _per_active_cycle suffix returned n/a).
# If the metrics below still return n/a, fall back to:
#   ncu --section WarpStateStats --section LaunchStats ... 2>&1 | grep -E 'smsp__|launch__|stall' | head -50
FLASHQLA_DISABLE_WG_REG=1 FLASHQLA_AUTOCP=0 \
FLASHQLA_BLACKWELL_NATIVE=1 FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE=1 \
ncu \
  --target-processes all \
  --kernel-name 'regex:tilelang_fused_chunk_gdr_fwd_blackwell.*' \
  --launch-count 3 \
  --metrics \
smsp__average_warps_issue_stalled_barrier.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard.ratio,\
smsp__average_warps_issue_stalled_mio_throttle.ratio,\
smsp__average_warps_issue_stalled_membar.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard.ratio,\
smsp__average_warps_issue_stalled_wait.ratio,\
smsp__average_warps_issue_stalled_no_instruction.ratio,\
launch__shared_mem_config_size.value,\
launch__registers_per_thread.value \
  python3 tests/test_gdr.py --set profile --nkh 8 --nvh 32 --skip-bwd \
    --seqlen 32768 --hide-acc --hide-lat
```

Save the output as plain text in `docs/ncu_reports/phase2_baseline_T32k_no_cp.txt` (no `.ncu-rep` binary needed; metric-only output is small enough to commit).

**Note (2026-05-20)**: The first command was run successfully. Three steady-state launches:

| metric | launch1 | launch2 | launch3 |
|---|---|---|---|
| sm__throughput | 14.72% | 14.85% | 14.97% |
| compute_memory_throughput | 36.99% | 37.34% | 37.59% |
| pipe_tensor_cycles_active | 14.68% | 14.85% | 14.97% |
| l1tex bank_conflicts (cumulative) | 725,832 | 1,462,525 | 2,929,165 |

Per-launch bank conflicts: ~1.4M. SM throughput ~15% means the kernel is **stall-dominated** (not compute-bound, not HBM-bound).

**Stall breakdown (from `--section WarpStateStats` fallback)**:
- Warp Cycles Per Issued Instruction: **17.19 cycles**
- "On average, each warp spends 8.5 cycles being stalled waiting for a scoreboard dependency"
- "This stall type represents about **49.6% of the total average of 17.2 cycles**"

**Conclusion**: Long Scoreboard (L1TEX data access) stall is the **single dominant** bottleneck at ~50%. Combined with the 1.4M bank conflicts and 15% SM throughput, the kernel is unambiguously **SMEM-subsystem-bound**.

**Decision rule trigger**: `Stall Long Scoreboard > 30%` ✓ + SMEM headroom (B reduces, doesn't increase) ✓ → **Candidate B (RS-gemm to delete `p_shared`) wins as first move.** Candidate D (swizzle) is the natural follow-up. Candidate A is on hold (no barrier-stall data, but with 50% spent on long scoreboard, A's headroom is at most 50%).

### Step 2: extract the four numbers that decide everything

Open the report and record the following table in this file under §3:

| metric | value | source section |
|---|---|---|
| SM throughput (%) | ? | GPU SoL |
| Memory throughput (%) | ? | GPU SoL |
| Tensor Core utilization (%) | ? | GPU SoL → Pipeline Utilization |
| Stall Barrier (%) | ? | Warp State Statistics |
| Stall Long Scoreboard (%) | ? | Warp State Statistics |
| Stall MIO Throttle (%) | ? | Warp State Statistics |
| Shared Memory Used per Block (KB) | ? | Launch Statistics |
| Shared Store Bank Conflicts (%) | ? | Memory Workload Analysis |
| Hottest source line | ? | Source Counters |

### Step 3: pick the candidate using the decision rule (Phase 1 §10.4)

| condition | candidate | doc link |
|---|---|---|
| `Stall Barrier > 30%` | **A: `h_shared` double buffer** | §2 below |
| `Stall Long Scoreboard > 30%` AND `SMEM ≤ 130KB` | **B: cons-O RS-gemm** | §3 below |
| `Shared Bank Conflicts > 20%` (or absolute >1M/launch) | **D: swizzle layout** | §5 below (new candidate) |
| no clear signal | re-investigate, do not commit | — |

**Preliminary signal from this session's NCU pass**: 
- Long Scoreboard stall = **49.6%** (single dominant bottleneck)
- L1tex bank conflicts = **1.4M / launch**
- SM throughput 15%, HBM 37%
- → Kernel is **SMEM-subsystem-bound**.

**This decisively favours Candidate B first** (RS-gemm deletes `p_shared`, removing one full round-trip from the long-scoreboard hot path). Candidate D (swizzle) is the second move on the same root cause. Candidate A is on hold.

The previous session committed sub-commit 3-A and 3-B without NCU evidence; both turned out to be wrong. **Do not skip step 3.**

---

## 2. Candidate A: `h_shared` double-buffering

### Hypothesis
Cons-S waits `bar_5` at iter end purely to fence its own next-iter `T.copy(h_fragment, h_shared)` against cons-O's cur-iter `T.gemm(q_shared, h_shared, o_fragment)` read. With a staged `h_shared`, the WAR vanishes, and `K^T @ vn_shared` (cons-S, end of cur iter) overlaps with `P @ Vd` (cons-O, end of cur iter).

### Code change sketch
```python
# alloc
h_shared = T.alloc_shared((2, DK, block_DV), dtype=qkva_dtype)

# cons-S
T.copy(h_fragment, h_shared[i_s % 2])
T.barrier_arrive(bar_1)            # cons-O wait bar_1 reads h_shared[i_s%2]
...
# at iter end:
T.barrier_arrive(bar_5)            # still fence prod-output o_shared
# DROP wait bar_5 — h_shared next-iter slot is i_s%2 != cur cons-O reader
T.gemm(k_shared[stage,:,:], vn_shared, h_fragment, transpose_A=True, clear_accum=False)
T.barrier_arrive(data_is_free[stage])

# cons-O  
T.gemm(q_shared[stage,:,:], h_shared[i_s % 2], o_fragment, clear_accum=True)
```

### SMEM cost (must verify with NCU §1 before committing!)
- block_DV=128: +16KB. If Phase 1 baseline reports ≥132KB used, **A is dead**.
- block_DV=64: +8KB. Easier to fit.

### Validation
- Correctness: T=4k/8k/16k/32k all four shapes.
- NOOP per_chunk: should drop from 2.41 → ~2.0 µs.
- Full T=32k no-CP gdr: should drop from 2.36 → ~2.0ms.

---

## 3. Candidate B: cons-O P uses RS-gemm (delete `p_shared`)

### Hypothesis
Currently:
```python
T.gemm(q_shared, k_shared, p_fragment, transpose_B=True, clear_accum=True)  # ← in fragment
... # P-postprocess in fragment
T.copy(p_fragment, p_shared)  # ← round-trip!
T.gemm(p_shared, vd_shared, o_fragment, clear_accum=False)  # ← SS gemm
```

The `T.copy(p_fragment, p_shared)` is a pure round-trip: the same warp group writes and immediately reads it via `T.gemm`. If TileLang exposes RS (register × shared) tcgen05 gemm on SM100, we can drop the round-trip:
```python
# unchanged
T.gemm(q_shared, k_shared, p_fragment, transpose_B=True, clear_accum=True)
# postprocess in p_fragment
T.gemm(p_fragment, vd_shared, o_fragment, clear_accum=False)  # ← RS gemm
```

### Pre-check (do this FIRST in next session)
```bash
# minimal RS-gemm probe; create a 64×64×64 test that uses register A, shared B
cat > /tmp/probe_rs_gemm.py <<'EOF'
import tilelang
import tilelang.language as T

@tilelang.jit(target='cuda')
def probe(M=64, N=64, K=64, dtype="bfloat16", accum_dtype="float32"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), accum_dtype)):
        with T.Kernel(1, threads=128) as bx:
            A_frag = T.alloc_fragment((M, K), dtype)
            B_shared = T.alloc_shared((K, N), dtype)
            C_frag = T.alloc_fragment((M, N), accum_dtype)
            T.copy(A, A_frag)
            T.copy(B, B_shared)
            T.gemm(A_frag, B_shared, C_frag, clear_accum=True)
            T.copy(C_frag, C)
    return main

# compile and check it doesn't error
print(probe()())
EOF
python3 /tmp/probe_rs_gemm.py
```

If TileLang errors out with "tcgen05 RS not supported", **B is dead**, fall back to A.

### Code change
Two-line change in cons-O `if not _cons_o_noop:` block: delete the `T.copy(p_fragment, p_shared)`, change `p_shared` to `p_fragment` in the second `T.gemm`.

### SMEM saving
- `p_shared`: -8KB (block_S × block_S × 2B = 64×64×2).
- Frees headroom for Candidate A or num_stages exploration later.

### Validation
- Correctness: all four shapes.
- NCU: `Stall Long Scoreboard` should drop noticeably; `Shared Store Bank Conflicts` should drop too (one fewer SMEM store stream).
- Full T=32k no-CP gdr: should drop from 2.36 → ~2.0ms.

---

## 4. Candidate C: `vd_shared` double-buffering (lower priority)

Defer unless A/B both fail or only land 5–10%.

---

## 5. Candidate D: SMEM swizzle pass (NEW — added based on Phase 1 NCU note)

### Hypothesis (now backed by NCU 2026-05-20)
Phase 1 §11 note in `blackwell_fused_fwd_perf_investigation.md` mentioned LDS waste 1.57×. **NCU on this branch shows 1.4M shared-memory bank conflicts per kernel launch** while SM throughput is only 15% — bank conflicts are a measurable contributor.

If a follow-up NCU run shows `Shared Bank Conflicts > 20%` of total SMEM accesses (or stall-MIO-throttle is high in the WarpStateStats section), the win is to add `T.use_swizzle` directives or change SMEM layouts (e.g., `swizzle="128B"` on `q/k/v_shared`).

### Concrete code points to investigate (in priority order)

1. `k_shared` — used as A in three SS gemms per chunk: `K^T @ h_shared` (cons-S), `Q @ K^T` (cons-O), `K^T @ vn_shared` (cons-S). It is the most-read SMEM tile in the kernel.
2. `q_shared` — used as A in cons-O `Q @ K^T` and `Q @ H`.
3. `v_shared` — used as B in `a @ v_new` AND read+written in-place for `v_new = v - g_exp * u`. The in-place write is the most likely conflict source.
4. `vd_shared` / `vn_shared` — used as B in cons-O `P @ Vd` and cons-S `K^T @ vn`.
5. `a_shared` — used as A in `a @ v_new`; also rewritten by cons-O scalar postprocess.

### Pre-check NCU command (run this FIRST in next session)

```bash
# Per-buffer bank conflict attribution: requires --set full + Source Counters
FLASHQLA_DISABLE_WG_REG=1 FLASHQLA_AUTOCP=0 \
FLASHQLA_BLACKWELL_NATIVE=1 FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE=1 \
ncu \
  --target-processes all \
  --kernel-name 'regex:tilelang_fused_chunk_gdr_fwd_blackwell.*' \
  --launch-count 1 \
  --section MemoryWorkloadAnalysis_Tables \
  --section SourceCounters \
  --import-source yes \
  -o /tmp/phase2_swizzle_diag \
  python3 tests/test_gdr.py --set profile --nkh 8 --nvh 32 --skip-bwd \
    --seqlen 32768 --hide-acc --hide-lat
ncu-ui /tmp/phase2_swizzle_diag.ncu-rep  # open Source page, sort by "Shared Conflict Cycles"
```

The Source page maps each conflict count to a specific `T.copy` / `T.gemm` / `for j_s,j_v in T.Parallel` line in `fused_fwd_native.py`. Whichever line tops the list is where the swizzle change goes.

### Code change sketch (illustrative — refine after NCU pinpoints the buffer)

```python
# Current
k_shared = T.alloc_shared((num_stages, block_S, DK), dtype=qkva_dtype)
# Candidate D: explicit swizzle on the K/V SMEM tiles
k_shared = T.alloc_shared((num_stages, block_S, DK), dtype=qkva_dtype, swizzle="128B")
v_shared = T.alloc_shared((num_stages, block_S, block_DV), dtype=qkva_dtype, swizzle="128B")
```

If TileLang's allocator does not accept a `swizzle` kwarg on SM100, the alternative is to insert `T.annotate_layout` calls or padded-leading-dim allocations.

### Validation

- Correctness: T=4k/8k/16k/32k all pass.
- NCU re-run: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` should drop from ~1.4M/launch to <300K/launch.
- Full T=32k no-CP gdr: target 2.36 → 2.0–2.1ms.
- If both bank conflicts AND gdr improve → keep. If conflicts drop but gdr unchanged → SMEM bandwidth was not the binding constraint after all → revert and pivot to A or B.

---

## 6. Numerical correctness regression test (mandatory addition)

Phase 1 had two close calls with numerical correctness (sub-commit 3-A WAR violation, prior fragment-B attempt). Before any Phase 2 sub-commit lands, add a fast unit test that runs without `--set profile`:

```python
# tests/test_gdr_smoke.py (new file)
def test_blackwell_native_smoke():
    """Tiny shape that runs in <1s; catches numerical regressions early."""
    # T=64 (1 chunk), 1 batch, 1 head, K=V=128
    # compare against ref_gdr.chunk_gated_delta_rule_fwd
    # tol=1e-3 on output, 1e-3 on final state
    ...
```

Add this to `pytest tests/test_gdr_smoke.py -x` and run it as the first step of every sub-commit's validation.

---

## 7. Suggested execution order (next session)

1. **Re-run NCU stall-reasons command (§1 step 1, second snippet)** — get the missing barrier/long_scoreboard/mio_throttle ratios. The metric names in §1 may need adjustment for the local NCU version; if so, fall back to `--section WarpStateStats` and grep the output.
2. **Source-level bank conflict attribution (§5 pre-check NCU command)** — open `ncu-ui`, find which `T.copy` / `T.gemm` line tops the "Shared Conflict Cycles" column. **This pinpoints which buffer to swizzle.**
3. **Add `tests/test_gdr_smoke.py`** (§6) — 5 minutes, prevents future regressions.
4. **RS-gemm probe (§3 pre-check)** — 5 minutes, decides if B is alive (regardless of whether D is also done; B is independent).
5. **Pick D and/or B** based on §1 step 3 decision rule. Initial signal favours D first; barrier-stall data may bring A back into contention.
6. Implement, validate correctness, NCU again, measure perf delta. Update the table in §1 with post-change metrics.

---

## 8. Out of scope for Phase 2

- TMEM placement of `h_fragment` (was option 2 in Phase 1 §3; deferred unless A/B both fail).
- bwd kernel optimization.
- Stream-K / persistent kernel restructuring.
- MX-FP8 / block-scaled paths.

---

*Drafted 2026-05-20 evening — ready for execution in next thread session. **Do NOT skip §1 NCU baseline capture.***
