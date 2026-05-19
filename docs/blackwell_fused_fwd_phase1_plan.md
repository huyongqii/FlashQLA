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
