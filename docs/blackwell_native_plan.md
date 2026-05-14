# FlashQLA Blackwell Native Kernel Plan

## Current State

On NVIDIA B200 (`sm_100`), FlashQLA currently runs through
`flash_qla.ops.gated_delta_rule.chunk.blackwell`, but the Blackwell module is a
compatibility wrapper around the Hopper TileLang kernels. SASS inspection of
TVM-generated artifacts shows `HMMA.16816.F32.BF16` / `HMMA.1688.F32.TF32`, not
`tcgen05.mma` or TMEM instructions.

This means B200 support is correctness-oriented only. It is not a native
Blackwell performance path.

## Dispatch Contract

- `chunk/hopper/*`: existing Hopper-compatible TileLang implementation.
- `chunk/blackwell/*`: future native Blackwell implementation.
- `FLASHQLA_REQUIRE_BLACKWELL_NATIVE=1`: fail if sm_100 would use the
  compatibility path.
- `FLASHQLA_BLACKWELL_NATIVE=1`: enable the experimental forward/kkt path that
  explicitly calls `T.tcgen05_gemm` and should fail instead of lowering to HMMA.
- `scripts/inspect_blackwell_mma.py --all`: verify whether generated artifacts
  contain `tcgen05`, `TMEM`, `WGMMA`, or `HMMA`.

## Priority

1. `fused_chunk_gdr_fwd`
2. `kkt_solve`
3. `prepare_h`
4. backward kernels

Forward is the SGLang-relevant path. Backward should not block inference
integration.

## Experimental Native Branch

`chunk/blackwell/fused_fwd.py` and `chunk/blackwell/kkt_solve.py` currently copy
the Hopper dataflow but replace each tensor-core `T.gemm` with explicit
`T.tcgen05_gemm(..., mbar=...)` plus `T.mbarrier_wait_parity(...)`. This is not
the final high-performance design. Its purpose is to force TileLang 0.1.9 to
either emit TCGEN05 or fail at the exact operand/layout that still needs TMEM
and layout work.

Run the first compile probe with:

```bash
FLASHQLA_BLACKWELL_NATIVE=1 python tests/test_gdr.py --set profile --skip-bwd --hide-lat
python scripts/inspect_blackwell_mma.py --no-run --all
```

If this fails with a TCGEN05 operand/layout error, the next implementation step
is to move the failing accumulator to `T.alloc_tmem` and annotate the matching
TCGEN05 layout. If it compiles but the inspector still reports HMMA, the explicit
TCGEN05 calls are not being selected and the lowering path must be debugged
before performance tuning.

## Forward GEMM Sites

The current Hopper-compatible `fused_fwd.py` has six tensor-core GEMM sites:

| Step | Expression | Shape family | Notes |
| --- | --- | --- | --- |
| State update | `K^T @ V'` | `128 x 64 x block_DV` | Updates recurrent state |
| Read state | `K @ S` | `64 x 128 x block_DV` | Produces `U` |
| Local value | `Ag @ W` | `64 x 64 x block_DV` | Uses chunk-local lower-triangular transform |
| Local score | `Q @ K^T` | `64 x 128 x 64` | Produces `P` |
| State output | `Q @ S` | `64 x 128 x block_DV` | Output from recurrent state |
| Local output | `Pg @ Vd` | `64 x 64 x block_DV` | Adds local contribution |

The Blackwell rewrite should focus on keeping these matrix products in native
Blackwell tensor-core form and reducing shared-memory round trips where TMEM can
hold accumulators.

## Initial Scope

Implement a native forward kernel for fixed-length, forward-only inference:

- `cu_seqlens is None`
- `output_h=False`
- `output_final_state=True`
- `output_o=True`
- `initial_state` supported
- `K=V=128`, `chunk_size=64`
- First target shapes:
  - TP8: `Hg=2`, `H=8`
  - TP2: `Hg=8`, `H=32`
  - TP1: `Hg=16`, `H=64`

Variable length, intra-card CP, `output_h=True`, and backward can remain on the
compatibility path until the fixed-length forward kernel is validated.

## Blackwell Design Direction

- Use native `tcgen05.mma` for BF16 GEMMs.
- Use TMEM for long-lived accumulators where possible:
  - recurrent state update accumulator
  - output accumulator
  - local chunk accumulator
- Split producer/consumer roles for Blackwell rather than reusing Hopper
  register budgets and 512-thread warp-specialized layout.
- Retune CTA shape independently for:
  - small head count / TP8
  - medium head count / TP2-TP4
  - large head count / TP1
- Avoid a single `grid_size -> block_DV` heuristic. Blackwell should have a
  per-shape policy table or generated autotune table.

## Difficulty

High. Because current codegen emits HMMA, the work is not a simple autotune.
Expected effort:

- 1-2 days: minimal native forward prototype for one fixed shape if TileLang
  exposes enough Blackwell primitives.
- 3-5 days: shape policy and correctness coverage for TP8/TP4/TP2/TP1.
- 1-2 weeks: robust production-quality path including varlen/CP and profiling.
- Longer if TileLang 0.1.9 cannot express `tcgen05`/TMEM directly and a lower
  level CUDA/CUTLASS-style kernel is required.

## Acceptance Criteria

For a Blackwell-native forward path:

1. `scripts/inspect_blackwell_mma.py --all` reports `tcgen05` and no HMMA-only
   result for FlashQLA forward artifacts.
2. Correctness passes against reference for fixed-length BF16:
   - TP8, TP4, TP2, TP1
   - `T in {2048, 8192, 32768}`
3. Benchmark meets:
   - TP8: at least current FlashQLA performance, preferably `>=1.2x FLA`
   - TP4: `>=1.0x FLA`
   - TP2/TP1: no worse than `0.95x FLA` before SGLang default consideration
4. SGLang integration remains shape-gated until all target shapes meet criteria.
