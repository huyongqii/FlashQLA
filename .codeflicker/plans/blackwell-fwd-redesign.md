# Blackwell fwd 2-WG + TMEM redesign plan

Last updated: 2026-05-19. Owner: thread n52xnmsioygtezfnrfmg.

## Status

- `scripts/smoke_tcgen05_redesign.py` PASSES on B200:
  - tmem accum (c): rel 0.0000 — `clear_accum=False` accumulation across two
    `tcgen05_gemm` calls works.
  - multi-tmem (d): rel 0.0000 — two coexisting `T.alloc_tmem` writes by
    independent `tcgen05_gemm` calls work.
  - producer/MM WG mbarrier handshake works.
- TileLang TMEM constraint discovered (will trigger compile error
  `Tmem buffer X does not have a layout specified`):
  - **TMEM may only be written by `tcgen05_gemm`.**
  - `T.copy(fragment → tmem)` is illegal (tested, fails to compile).
  - Reading TMEM via `T.copy(tmem → fragment)` is fine.

## Why we are doing this redesign

Current `fused_fwd_native.py` runs at ~1/3 of FLA throughput on B200. Three
NCU-confirmed bottlenecks:

1. **1 block / SM** (Block Limit Reg = 1, Block Limit SMEM = 1). `h_fragment`
   takes 64KB of registers; `q+k_shared` × 2 stages takes 64KB SMEM.
2. **Eligible Warps = 0.11.** All 4 warpgroups are phase-locked on the same
   chunk's barrier chain (cons-S → cons-V → cons-O serial dependency). One
   block per SM means there is no second block to fill the gaps.
3. **Long Scoreboard 83%.** Symptom of (1)+(2), not the cause.

The redesign target:

- **2 WGs only** (PROD + MM). cons-S/V/O collapsed into one MM WG that
  sequentially issues 6 `tcgen05_gemm` calls. tcgen05 is async, so the TC
  pipeline is filled by issue-chain, not by 4 separate WGs.
- **TMEM accumulators** for H, P, O, U, V_d. Frees ~50KB of register pressure
  per CTA → average ≤ 64 reg/thread → Block Limit Reg goes from 1 to 2.
- **Single-stage q/k SMEM + block_DV=64**. SMEM drops to ~96KB → Block Limit
  SMEM goes from 1 to 2.
- Result: 2 blocks/SM × 148 SM = 296 active CTAs (was 64). Two blocks
  process different `(head, dv_block)` so they share no barriers; whenever
  one block is waiting, the other is issuing. Eligible warps should jump
  from 0.11 to ~3-5.

## CRITICAL OPEN PROBLEM (must solve before commit 1)

The cons-S inner loop computes:

```
H_new = g_last · H_old + K^T @ V_new
```

In the current code this is two separate ops on `h_fragment` (a register):

```python
h_fragment[j_k, j_v] *= g_last_local[0]     # elementwise scalar scale
T.gemm(K, V_new, h_fragment, clear_accum=False)
```

In TMEM, **neither op composes**:

- `tcgen05_gemm` only supports `clear_accum=True` (`D = A@B`) or
  `clear_accum=False` (`D += A@B`). No cuBLAS-style `D = α·C + A@B`.
- TMEM cannot be written elementwise (smoke confirmed).

### Three candidate fixes (pick before writing commit 1)

#### Option α — Absorb g_last into V (algebraic rewrite, recommended)

Define a normalized accumulator `H~ = H · prod(g_last_i)^{-1}` so that
`H_new~ = H_old~ + K^T @ V_new~` where `V_new~ = V_new / prod(g_last_<n)`.

- Pure GEMM accumulation, no scalar scale on TMEM.
- One additional scalar broadcast `V_new~ = V_new · running_inv_g` per
  iteration (cheap; reuses cons-V WG's elementwise pass).
- Numerical risk: `running_inv_g = exp(-Σ g_last)` can underflow for very
  long sequences. With T=32k / chunk 64 = 512 iters, Σ g_last is bounded
  because `g = logsigmoid(...)/16` keeps |g|<0.05, so Σ ≈ 25 → exp(-25)
  ≈ 1e-11, safely above bf16 underflow (~1e-38) but at the edge of fp32
  ULP for the final unscale.
- **Verdict**: viable for T ≤ 32k, must benchmark for T = 128k.

#### Option β — Round-trip H through SMEM each iteration

Each iter:
1. `T.copy(h_tmem → h_frag)` (read)
2. `h_frag *= g_last` (scale in register)
3. `T.copy(h_frag → h_smem)` (stage)
4. `tcgen05_gemm(h_smem, identity_smem, h_tmem, clear_accum=True)` (load)
5. `tcgen05_gemm(K, V_new, h_tmem, clear_accum=False)` (accumulate)

- 2 GEMMs per iter (one trivial 128×64 @ 64×64, one real 128×64 @ 64×64).
- The "load via identity GEMM" is the workaround for "TMEM has no non-GEMM
  write path". Trivial GEMM is ~0.1 μs but adds latency.
- **Verdict**: correctness-safe, perf-risky. Use as fallback if α fails.

#### Option γ — Keep H in register (give up on TMEM for H)

Move only P, O, U, V_d to TMEM. Keep h_fragment in register (current
behavior). Saves ~32KB register pressure (P+O+U+Vd freed) but not 64KB.

- Block Limit Reg goes from 1 to 1.5 (still 1, but easier to reach 2 if
  CONSUMER_S_NREG can drop from 168 to ~110).
- **Verdict**: weakest improvement. Only choose if α and β both fail.

### Recommended path: try α first, fall back to β.

## Commit roadmap

### Commit 1: TMEM for P/O/U/V_d (avoid the H problem entirely first) — 1.5h

- Move `p_fragment`, `o_fragment`, `u_fragment`, `v_fragment`,
  `vu_fragment` to TMEM allocations.
- `H` stays in register (`h_fragment` unchanged).
- All `T.gemm` writing to these accumulators → `T.tcgen05_gemm` with
  per-GEMM completion mbar.
- Sync points: each TMEM read needs `T.mbarrier_wait_parity(mbar, phase)`
  before the corresponding `T.copy(tmem → fragment)`.
- Validate correctness with `tests/test_gdr.py --skip-bwd`.
- Expected gain: 1.2-1.4× (frees ~32KB reg, but still 4 WG / 1 block per SM).

### Commit 2: H to TMEM via Option α — 1.5h

- Add `running_inv_g_local` (fp32 scalar) accumulating `-g_last` per iter.
- cons-V WG: `vn_shared[j,v] = V_new[j,v] * exp(running_inv_g)` instead of
  the current `vn_shared = v - g_exp · u`. Need to fold the existing
  `g_rev_exp` factor in too.
- cons-S: drop `h_fragment *= g_last`, drop `bar_5` round-trip; just
  `tcgen05_gemm(K, vn_shared, h_tmem, clear_accum=False)`.
- At store time: `ht = exp(Σ g_last) · h_tmem`. Fragment-level scale + write.
- Expected gain: 1.5× cumulative (Block Limit Reg should now be 2).

### Commit 3: Collapse cons-S/V/O into single MM WG — 2h

- New thread layout: 0..127 = MM (single WG), 128..255 = PROD (4 warps:
  TMA Q/K, TMA V/b, TMA A/g, store O).
- MM WG body: per chunk, issue 6 tcgen05 GEMMs in dependency order with
  per-GEMM mbar; do scalar postprocess (P decay, O scale) between issues
  while the TCs are busy on independent ops.
- arrive_count audit (4 WG → 2 WG removes 2/3 of barrier participants).
- Expected gain: 2.5-3× cumulative. Targets / matches FLA throughput.

### Commit 4: SMEM trim + Block Limit SMEM → 2 — 0.5h

- block_DV default 64 (already configurable; just bump default and verify).
- `q_shared` / `k_shared` to single stage if PROD-MM ratio allows.
- `ncu --metrics achieved_active_blocks_per_sm` should report ≥ 1.5.

## Files affected

- `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd_native.py`
  (the actual rewrite, ~400 lines changed)
- `scripts/smoke_tcgen05_redesign.py` (precondition test, already in repo)
- No changes to `prepare_h.py`, `kkt_solve.py`, `cp_fwd.py`, or
  `__init__.py` should be needed.

## Verification protocol per commit

1. `FLASHQLA_AUTOCP=0 python tests/test_gdr.py --skip-bwd` — correctness.
2. `FLASHQLA_AUTOCP=0 python tests/test_gdr.py --set profile --skip-bwd
   --hide-acc` — perf vs FLA.
3. `ncu --section LaunchStats --section Occupancy ...` — confirm
   block/SM, eligible warps progress as predicted.
4. Only proceed to next commit if both correctness and the predicted
   block/SM progression hold.

## Useful references

- Current native kernel: `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd_native.py`
- Hopper original (4 WG, fragment H): `flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py`
- Smoke test (PASSED): `scripts/smoke_tcgen05_redesign.py`
- Existing tcgen05 example: `scripts/smoke_tcgen05.py`
- prepare_h with `store_h=True` (independent H pipeline reference): `flash_qla/ops/gated_delta_rule/chunk/blackwell/prepare_h.py`
