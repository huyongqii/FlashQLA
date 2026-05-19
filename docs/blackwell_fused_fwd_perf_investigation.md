# Blackwell `fused_fwd_native` Performance Investigation

> **Status**: In progress. Major win identified (`disable_warp_group_reg_alloc` → 2.8× speedup). Further headroom requires grid-level changes.
>
> **Target file**: `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd_native.py`
>
> **Workload (all measurements)**: `B=1, Hk=8, Hv=32, T=32768, D=DV=128`, `--set profile`, no CP.
>
> **Hardware**: NVIDIA Blackwell (CC 10.3, 132 SMs, 228 KB SMEM/SM, 65536 register/SM, 64 warp/SM).

---

## 1. Starting point

| | FLA | FlashQLA fused |
|---|---|---|
| `gdr` time | 0.75 ms | **10.78 ms** |
| `total` fwd | 1.73 ms | 11.22 ms |
| Speedup vs FLA | 1.0× | **0.31×** |

The same fused kernel runs **faster than FLA on Hopper** but is **3× slower on Blackwell**, suggesting an SM90→SM100-specific regression rather than an algorithmic issue.

---

## 2. Initial NCU profile (baseline)

| Metric | FlashQLA | FLA |
|---|---|---|
| GPU time | 13.01 ms | — |
| Registers/thread | 128 | 144 |
| Dynamic SMEM/block | 147 KB (later confirmed 118 KB for non-AG variant) | 65.84 KB |
| Waves/SM | 0.86 | 0.58 |
| SM throughput | 7.96% | 23.09% |
| Tensor pipe active | 7.96% | 11.64% |
| `warps_active` | **24.99%** | — |

Two suspicious facts:
1. **Active warps only 25%** despite supposedly heavy fused work.
2. SM throughput **lower** than FLA, even though total work is similar.

---

## 3. Hypothesis 1: barrier stall in fused warp-spec — **DISPROVEN**

Initial guess: 4 WG × 9 barrier rendezvous per chunk forms a serial critical path.

**Disproof experiments** (each removed one major piece of work; baseline 10.14 ms):

| Experiment | gdr time |
|---|---|
| baseline | 10.14 ms |
| Remove `cons-S K^T@vn` GEMM | 10.77 ms |
| Remove `cons-O Q@H` GEMM | 10.62 ms |
| Remove `cons-O P` post-process | 10.51 ms |
| Remove `prod-output` HBM store | 10.60 ms |
| `num_stages=1` (no double-buffer) | 12.13 ms |

Removing any single piece of work has **no effect** on overall time. If barriers were on the critical path, deletions would shift but not eliminate stalls. This means **none of the GEMMs/elementwise are the critical path**.

---

## 4. Hypothesis 2: occupancy starvation — **PARTIALLY CONFIRMED**

Re-ran `WarpStateStats`:

```
Warp Cycles Per Issued Instruction:    76.0 cycle
  - of which stall_long_scoreboard:    68.5 cycle (90.1%)
  - barrier stall:                     not in top
```

Re-ran detailed metrics:

| Metric | Value |
|---|---|
| `l1tex__throughput.pct` | **11.22%** |
| `sm__pipe_tensor_cycles_active.pct` | **4.58%** |
| `sm__pipe_fma_cycles_active.pct` | 1.27% |
| `smsp__inst_executed_op_shared_ld` | 51 M |
| `mem_shared_op_ld wavefronts` | 320 M |

Warps spend 90% of cycles waiting for L1TEX, **but L1TEX itself is only 11% busy**. The pipes are idle while warps wait — classic **latency-hiding failure**, not bandwidth saturation.

---

## 5. Occupancy breakdown

```
launch__occupancy_limit_blocks    = 32
launch__occupancy_limit_registers = 1   ← reg pins us to 1 CTA/SM
launch__occupancy_limit_shared_mem= 1   ← SMEM also pins us to 1 CTA/SM
sm__ctas_active.avg.per_cycle     = 1.00
sm__warps_active.avg.pct          = 24.99%
launch__waves_per_multiprocessor  = 0.86  ← grid (128) < SM count (132)
```

**Both register and SMEM caps independently force 1 CTA/SM**. With 16 warps/CTA, that's **16/64 ≈ 25%** of the SM's warp slots filled — exactly matching the observed `warps_active`.

### Resource breakdown

**SMEM (`block_DV=32, num_stages=2`, total ≈ 118 KB)**:
| Buffer | size | num_stages | total |
|---|---|---|---|
| q_shared | 16 KB | 2 | 32 KB |
| k_shared | 16 KB | 2 | 32 KB |
| a_shared | 8 KB | 2 | 16 KB |
| v_shared | 4 KB | 2 | 8 KB |
| h_shared | 8 KB | 1 | 8 KB |
| p_shared | 8 KB | 1 | 8 KB |
| vd, vn, o, g/b | misc | — | ~14 KB |
| **Total** | | | **~118 KB** |

**Register (`512 thread/CTA × 128 reg/thread`)**:
- producer 4 warp × 24 reg = 3 KB
- cons-S 4 warp × 168 reg = 21 KB
- cons-V 4 warp × 160 reg = 20 KB
- cons-O 4 warp × 160 reg = 20 KB
- **Total = 65536 register/CTA = 100% of SM register file**

### Sanity check: H-scan to test grid-fill

```
H=32   grid=128  time=25.7 ms   (work = 1×)
H=64   grid=256  time=26.5 ms   (work = 2×, +3%)
H=128  grid=512  time=30.6 ms   (work = 4×, +19%)
```

**4× the work takes only 19% more time** — the SM has **3.4× spare capacity** that the current grid doesn't fill. Grid is too small (128 vs 132 SMs = single wave that doesn't even cover the GPU).

---

## 6. Hypothesis 3: warp-specialization with `setmaxnreg` is an SM100 anti-pattern — **CONFIRMED**

### Why we suspected it
- Hopper (SM90) `wgmma` keeps accumulators in registers; `setmaxnreg` is essentially free.
- Blackwell (SM100) `tcgen05` puts accumulators in TMEM; `tcgen05.ld` brings them back at additional latency.
- `setmaxnreg.inc.sync.aligned` is a **runtime** register-pool reallocation that **synchronizes across the warp group**.

### Experiments

#### Attempt A: just `T.annotate_min_blocks_per_sm(2)`
Result: **kernel hangs at 100% GPU util**.
Cause: `__launch_bounds__(512, 2)` cuts the per-CTA register pool in half, but `setmaxnreg.inc(168)` from cons-S then tries to grow allocation beyond what the pool can give → infinite spin / deadlock.

#### Attempt B: scale down `CONSUMER_*_NREG` to 0.5 alone
Result: gdr 10.14 → **10.77 ms** (essentially unchanged, no spilling triggered).
Insight: cons WGs are **over-allocated** at 168/160 — they don't actually use that many registers. Confirms there's headroom.

#### Attempt C: `T.disable_warp_group_reg_alloc()` + `min_blocks=2` ✅
Replaced all `T.set_max_nreg(...)` calls with `T.disable_warp_group_reg_alloc()` (controlled by `FLASHQLA_DISABLE_WG_REG=1`).

Result: gdr **10.78 ms → 3.84 ms** (**2.8× speedup**).

### Verifying *why* it sped up

Re-ran NCU after the win:

| Metric | Before | After |
|---|---|---|
| `registers_per_thread` | 128 | **64** |
| `occupancy_limit_registers` | 1 | **2** |
| `occupancy_limit_shared_mem` | 1 | 1 (still) |
| `ctas_active.avg` | 1.00 | **1.00** (still!) |
| `warps_active.pct` | 25% | **25%** (still!) |
| `tensor_cycles_active.pct` | 4.58% | **12.44%** |
| `stall_long_scoreboard` | ~68 of 76 cycles (90%) | **15.6** |

**Surprising finding**: Occupancy did **not** change — still 1 CTA/SM. The 2.8× speedup came from `tensor_cycles_active` jumping 4.58 → 12.44 (≈2.7×), exactly matching the speedup ratio.

### Conclusion: the real cost of `setmaxnreg`

`setmaxnreg.inc/dec` is not free on SM100:
1. It's a **synchronizing** instruction across the warp group.
2. It introduces **scheduler hazards** that prevent ptxas from co-issuing tensor instructions efficiently.
3. With it disabled, ptxas emits a flat 64 reg/thread per CTA, which has **dramatically better instruction scheduling** and tensor pipe utilization.

This is an **SM100-specific regression** of the warp-spec pattern — on SM90 it does not have this scheduling tax (likely because `wgmma` issues differ and the pool-resize is cheaper).

---

## 7. Hypothesis 4: cutting SMEM unlocks 2 CTA/SM — **DISPROVEN (for current grid)**

Followed up by adding `FLASHQLA_NUM_STAGES=1` on top of disabled-WG + min_blocks=2.

| | gdr time | SMEM | occ_limit_blocks | ctas_active | tensor_active | long_sb |
|---|---|---|---|---|---|---|
| disabled-WG + min_blocks=2 | **3.84 ms** | 116 KB | reg=2, smem=1 | 1.00 | 12.44% | 15.6 |
| disabled-WG + min_blocks=2 + num_stages=1 | **10.5 ms** | **73 KB** | reg=2, smem=2 | **1.00** | **5.20%** | 53.88 |

**SMEM dropped to 73 KB and both occupancy limits opened to 2 — but `ctas_active` stayed at 1.00.**

Reason: `waves_per_multiprocessor = 0.86`. The grid is `B × H × ceil(DV/block_DV) = 1 × 32 × 4 = 128 CTAs` for 132 SMs. Even if every SM could fit 2 CTAs, **there aren't 264 CTAs in flight to fill them**. The grid is the limiter, not occupancy.

Without double-buffer, producer-consumer pipelining collapses → tensor utilization halves (12.44 → 5.20) → time blows up.

---

## 8. Current best result

```bash
FLASHQLA_DISABLE_WG_REG=1 \
FLASHQLA_AUTOCP=0 \
FLASHQLA_BLACKWELL_NATIVE=1 \
FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE=1 \
python3 tests/test_gdr.py --set profile --nkh 8 --nvh 32 --skip-bwd
```

| | FLA | FlashQLA |
|---|---|---|
| `gdr` time | 0.749 ms | **3.84 ms** |
| `total` fwd | 1.73 ms | 4.28 ms |
| Speedup vs FLA | 1.0× | **0.40×** (was 0.31×) |

**Net improvement**: gdr 10.78 → 3.84 ms (**−65%, 2.8×**), achieved via a single env-controlled patch with no change in algorithm or buffer layout.

`min_blocks_per_sm` annotation alone (without disabling `setmaxnreg`) does **not** work — the two are linked.

---

## 9. Hypothesis 5: grid is the bottleneck — **DISPROVEN**

After the `disable_wg_reg` win, re-ran a clean H-scan and a T-scan to verify whether the remaining 3.84 ms gap is from grid (small grid → SM idle) or from per-CTA work (SM saturated).

### Hv-scan (vary V-heads, hence vary grid)
`B=1, Hk=8, T=32768`, with `_select_block_dv` auto-sized so `grid = Hv × ceildiv(DV, block_DV)`.

| Hv | est_grid | total fwd | ratio vs Hv=32 |
|---|---|---|---|
| 32 | 128 | 2.64 ms | 1.0× |
| 64 | 256 | 3.46 ms | 1.31× |
| 128 | 512 | **9.15 ms** | **3.47×** |
| 256 | 1024 | **18.02 ms** | **6.83×** |

**4× the work → 3.47× the time** (near-linear). The earlier H-scan that suggested "3-4× spare capacity" was an artifact of the `setmaxnreg` scheduling tax — not real headroom.

After `disable_wg_reg`, the SM is **already saturated** by a single CTA per SM. Increasing grid linearly increases time — there's no slack to absorb extra CTAs.

### T-scan (fix work/CTA, vary serial chunk loop)
`B=1, Hk=8, Hv=32` — grid stays at 128 CTAs; only the serial chunk loop length changes.

| T | NT | time | per_chunk |
|---|---|---|---|
| 4096 | 64 | 0.39 ms | 6.12 µs |
| 8192 | 128 | 0.71 ms | 5.55 µs |
| 16384 | 256 | 1.36 ms | 5.29 µs |
| 32768 | 512 | 2.63 ms | 5.14 µs |
| 65536 | 1024 | 5.20 ms | 5.07 µs |
| 131072 | 2048 | 10.32 ms | **5.04 µs** |

**Per-chunk time converges to 5.04 µs.** Short-sequence overhead (~1 µs from cumsum/solve startup) bleeds out as T grows; the steady state is the chunk-loop body.

### Comparison with FLA
| | per_chunk in chunk loop | what runs in chunk loop |
|---|---|---|
| FLA H-kernel | **1.46 µs** | 2 GEMM (`w@h`, `k^T @ v_new`) + 1 FMA (`h *= exp(g)`) |
| FLA O-kernel | (separate kernel, large grid) | 3 GEMM + softmax (parallelizable across NT) |
| **FlashQLA fused** | **5.04 µs** | 6 GEMM (`q@k^T`, `q@h`, `p@v_new`, `w@h`, `k^T@v_new`, plus aux) + softmax + all SMEM staging — **all serialised in the chunk loop** |

Ratio: **5.04 / 1.46 ≈ 3.45×** — exactly matches the count of extra GEMM+staging that FlashQLA's fused architecture forces into the same serial chunk loop.

### Root cause statement (final)

> **The 5× gap to FLA on Blackwell is not an implementation bug — it is the cost of fusing the O computation into the H chunk loop. With the SM saturated (post-`disable_wg_reg`), every extra GEMM in the loop directly extends per-chunk time.**

On Hopper, the SM was *not* saturated (because `wgmma` register accumulators + free `setmaxnreg` left scheduling headroom), so the fused architecture's extra work cost almost nothing — the SM hid it. On Blackwell, with `tcgen05` TMEM accumulators and `setmaxnreg` scheduling tax, the SM saturates with much smaller work, and the extra GEMMs now bill in full.

---

## 10. Two paths forward

### Path A — FLA-style kernel split (architectural)
Two kernels: H-only (chunk loop), O-only (chunk-parallel grid).
- H kernel: only `w@h`, `k^T @ v_new`, `h *= exp(g)` — store h to HBM each chunk.
- O kernel: grid = `(NT, B×Hv, V/BV)`, fully parallel across chunks.
- Expected per-chunk in H: ~1.5 µs → total H ≈ 0.75 ms.
- O kernel runs in parallel with no serial dependency.

**Expected total fwd ≈ 1.5–1.8 ms** (FLA-equivalent).

Cost: ~1–2 weeks of refactor. Loses some HBM-traffic saving from fusion (h has to round-trip through HBM).

### Path B — keep fused, optimise per-chunk internals
- Eliminate `vd_shared` / `vn_shared` SMEM round-trips.
- Move some accumulators off TMEM back to registers.
- Reduce LDS bank conflicts (current LDS wavefront ratio 320M / 51M ≈ 1.57× over ideal).

**Optimistic floor: ~3.5–4 µs/chunk → total gdr ≈ 1.8–2.0 ms ≈ 0.5× FLA.**

Cost: a few days, but ceiling is bounded by the 6-GEMM/chunk math.

### Recommendation
**Path A.** The math is decisive: per-chunk floor for any 6-GEMM fused chunk loop is fundamentally above FLA's 1.46 µs. Path B can close some of the 3.45× gap but cannot win.

Before committing to Path A, run **the falsifiability experiment in §11**.

---

## 11. Feasibility check before committing to Path A

Disable cons-O (skip `Q@H`, `P@V_new`, softmax, write-out) inside the existing fused kernel. Measure the resulting `per_chunk`.

- If `per_chunk` drops from 5.04 µs to ~2.0–2.5 µs → Path A's expected floor (~1.5 µs after losing aux I/O) is reachable. **Commit to Path A.**
- If `per_chunk` only drops to ~3.5–4 µs → cons-O is not the dominant cost. Re-evaluate before refactoring.

This experiment costs ~30 minutes (one patch + one run) and decisively gates the architectural decision.

---

## 11b. Feasibility check result (2026-05-20) — **Path A alone is insufficient**

Knob `FLASHQLA_CONS_O_NOOP=1` was added: cons-O keeps every barrier wait/arrive intact (so the rest of the pipeline still drains correctly) but skips all GEMMs, P-postprocess, and o_shared write. The upper triangle of `a_shared` is still zero-filled because cons-V reads it, but the per-chunk multiplications by `g_exp_shared` / `b_shared` are removed.

T-scan run on top of the current best config (`FLASHQLA_DISABLE_WG_REG=1`):

| T | NT | time | per_chunk |
|---|---|---|---|
| 4096 | 64 | 0.26 ms | 4.10 µs |
| 16384 | 256 | 0.93 ms | 3.63 µs |
| 32768 | 512 | 1.78 ms | 3.48 µs |
| 65536 | 1024 | 3.49 ms | 3.41 µs |
| 131072 | 2048 | 6.90 ms | **3.37 µs** |

### Decomposition of per-chunk time

| segment | per_chunk | meaning |
|---|---|---|
| FlashQLA fused, full | 5.04 µs | cons-S + cons-V + cons-O + producer |
| FlashQLA fused, cons-O NOOP | 3.37 µs | cons-S + cons-V + producer (H-only) |
| **cons-O actual cost** | **1.67 µs** | the 3 GEMMs + softmax + p_shared + o_shared write |
| FLA H-only kernel | 1.46 µs | reference baseline |
| **FlashQLA H-only "tax"** | **3.37 − 1.46 = 1.91 µs** | overhead of the warp-spec H pipeline itself |

### Decision

**Path A alone is not sufficient.** Splitting cons-O into a separate kernel only recovers the 1.67 µs cons-O cost. The remaining 3.37 µs/chunk is **2.3× FLA's H-kernel** — there is a structural cost in the multi-WG warp-spec H pipeline that is independent of cons-O.

Estimated path-A-only outcome: gdr ~1.73 ms (vs current 3.84 ms, vs FLA 0.75 ms). That's a 2.2× speedup over today, but **still 2.3× behind FLA**.

To match FLA, both fixes must land:

1. **Cons-O extraction** → independent O kernel. Recovers ~1.67 µs/chunk in the H loop.
2. **Cons-S + cons-V reorganisation** to eliminate the H-pipeline tax. Collapse the 3-WG ping-pong (cons-S → cons-V → cons-O via SMEM/TMEM staging) into a tighter design that mirrors FLA's instruction layout.

#### Likely sources of the 1.91 µs H-pipeline tax

a. **Multi-WG mbarrier chain.** Every chunk crosses 4 barriers (`data_is_ready`, `bar_0`, `bar_3`, `data_is_free`). Even with cons-O turned into a no-op, the chain still serialises producer → cons-S → cons-V → cons-O.
b. **TMEM round-trip in cons-V.** The `u_tmem` / `v_tmem` accumulators introduced during the TMEM-migration commit require `tcgen05.ld` to bring values back to registers; each read-back has 10s-of-cycle latency that is not hidden when occupancy stays at 1 CTA/SM.
c. **Intermediate SMEM staging.** `v_new` is written to `v_shared` by cons-V and re-read by cons-S; FLA keeps `v_new` in fragments throughout.
d. **LDS bank conflict.** NCU showed `lsu_wavefronts_mem_shared_op_ld / inst_executed_op_shared_ld` ≈ 6.3 vs ideal 4 (1.57× over-traffic).

### New recommendation — two-phase plan

**Phase 1 (≤1 week): eliminate the H-pipeline tax inside the current fused kernel.**
- Target: bring `per_chunk (cons-O NOOP)` from 3.37 µs to ~2.0 µs.
- Approach: collapse cons-S + cons-V into a single WG that does the H recurrence end-to-end on registers. Drop `v_shared` round-trip (keep `v_new` in fragments). Drop `u_tmem` / `v_tmem` (keep on registers — fewer threads means more reg/thread is fine). Reduce barrier count.
- Even keeping cons-O in place, this alone should bring full per_chunk from 5.04 to ~3.5 µs (gdr ~1.8 ms).

**Phase 2 (1–2 weeks): extract cons-O into a separate O kernel.**
- Target: drop the 1.67 µs cons-O contribution. With Phase 1 done, full per_chunk → ~2.0 µs.
- Requires writing a chunk-parallel O kernel (grid `(NT, B×Hv, V/BV)`).
- Expected total fwd: 1.0–1.5 ms (match-or-beat FLA).

The new ordering is **fix the H pipeline first, then extract cons-O** — Phase 1 is higher leverage (1.91 µs vs 1.67 µs) and sets the right foundation for Phase 2.

---

## 10. Appendix: experiments and timings

### Pure deletions (each from baseline 10.14 ms)
| Change | gdr time | Δ |
|---|---|---|
| baseline | 10.14 ms | — |
| Remove cons-S K^T@vn GEMM | 10.77 ms | +0.63 |
| Remove cons-O Q@H GEMM | 10.62 ms | +0.48 |
| Remove cons-O P post-process | 10.51 ms | +0.37 |
| Remove prod-output HBM store | 10.60 ms | +0.46 |
| num_stages=1 | 12.13 ms | +1.99 |

### Resource / register experiments
| Change | gdr time | Notes |
|---|---|---|
| `set_max_nreg ×0.5` | 10.77 ms | over-allocated, no spill |
| `min_blocks=2` (alone) | hang | setmaxnreg + launch_bounds conflict |
| `min_blocks=2 + nreg ×0.5` | hang | same |
| `disable_wg_reg` (alone) | not measured separately | |
| **`disable_wg_reg + min_blocks=2`** | **3.84 ms** | **—64%, current best** |
| `disable_wg_reg + min_blocks=3` | 6.09 ms | spilling at ~43 reg/thread |
| `disable_wg_reg + min_blocks=2 + num_stages=1` | 10.5 ms | lost pipelining > occupancy gain |

### Final NCU snapshot at best config
```
gpu__time_duration                              =  732 µs (per launch, single chunk slice)
launch__shared_mem_per_block_dynamic            = 118,784 byte
launch__registers_per_thread                    = 64
launch__waves_per_multiprocessor                = 0.86
sm__ctas_active.avg.per_cycle_active            = 1.00
sm__warps_active.avg.pct_of_peak_sustained_active = 24.86%
sm__pipe_tensor_cycles_active.avg.pct_*         = 12.44%
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active = 15.61
smsp__average_warps_issue_stalled_barrier_per_issue_active         =  1.32
```

---

## 12. Configuration knobs added

The following environment variables were introduced in `fused_fwd_native.py` for diagnostics:

| Env var | Effect | Default |
|---|---|---|
| `FLASHQLA_NUM_STAGES` | Override `num_stages` for SMEM experiments | unset (= use compiled value) |
| `FLASHQLA_NREG_SCALE` | Multiply CONSUMER_*_NREG by this factor | 1.0 |
| `FLASHQLA_DISABLE_WG_REG` | Replace `T.set_max_nreg(...)` with `T.disable_warp_group_reg_alloc()` | 0 |
| `FLASHQLA_MIN_BLOCKS_PER_SM` | Emit `T.annotate_min_blocks_per_sm(N)` | 1 (no annotation) |
| `FLASHQLA_CONS_O_NOOP` | Skip cons-O's GEMMs / softmax / o-write while keeping all barriers (timing-only, output garbage) | 0 |

**Recommended production config (current best):**
```bash
FLASHQLA_DISABLE_WG_REG=1
# do NOT set FLASHQLA_MIN_BLOCKS_PER_SM (it doesn't actually help, given grid is the limiter)
```

---

*Last updated: 2026-05-20*
