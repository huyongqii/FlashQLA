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
- `FLASHQLA_BLACKWELL_NATIVE_KERNELS=kkt`: choose which experimental kernels to
  enable. The default is empty, which keeps all kernels on the compatibility
  path even when `FLASHQLA_BLACKWELL_NATIVE=1` is set. Use `kkt` only for
  correctness/codegen experiments; the current KKT prototype is slower than the
  compatibility path. Use `kkt,fwd` only for debugging the mechanical TCGEN05
  fused-forward port, which currently compiles but can hang at runtime.
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
`T.tcgen05_gemm(..., mbar=...)` plus `T.mbarrier_wait_parity(...)`. TCGEN05 does
not accept `local.fragment` accumulators, so the experimental path uses TMEM
scratch accumulators and copies results back to fragments/shared buffers for the
existing elementwise epilogue. This is not the final high-performance design.
Its purpose is to force TileLang 0.1.9 to either emit TCGEN05 or fail at the
exact operand/layout that still needs deeper TMEM and layout work.

`kkt_solve` has one BF16 tensor-core GEMM (`K @ K^T`) and two FP32 32x32 helper
matrix products in the triangular inverse. TileLang 0.1.9 does not accept
`float32` shared/shared inputs for TCGEN05, so those helper products are kept as
explicit CUDA-core accumulation loops. This avoids accidental HMMA/TF32 fallback
while keeping the main BF16 product on TCGEN05.

Run the first compile probe with KKT isolated:

```bash
FLASHQLA_BLACKWELL_NATIVE=1 FLASHQLA_BLACKWELL_NATIVE_KERNELS=kkt \
  python tests/test_gdr.py --set profile --skip-bwd --hide-lat
python scripts/inspect_blackwell_mma.py --no-run --all
```

To bind SASS inspection to the latest benchmark instead of old TVM cache
directories, prefer:

```bash
python scripts/inspect_blackwell_mma.py --no-run --latest-tvm-dir
# or, immediately after a benchmark:
python scripts/inspect_blackwell_mma.py --no-run --since-minutes 5
```

To debug the mechanical fused-forward TCGEN05 port:

```bash
FLASHQLA_BLACKWELL_NATIVE=1 FLASHQLA_BLACKWELL_NATIVE_KERNELS=kkt,fwd \
  python tests/test_gdr.py --set profile --skip-bwd --hide-lat
python scripts/inspect_blackwell_mma.py --no-run --all
```

The active `fwd` experiment is `chunk/blackwell/fused_fwd_native.py`, a
fixed-length single-consumer TCGEN05 baseline. The older mechanical port remains
in `chunk/blackwell/fused_fwd.py` as a compiler probe only.

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

## Forward Rewrite Strategy

The mechanical `fused_fwd.py` port proves that TCGEN05 lowering is reachable,
but it is not a viable performance or correctness base because the Hopper
512-thread producer/three-consumer warp-specialized pipeline can deadlock after
TCGEN05/TMEM substitution.

The native forward rewrite should proceed as a new kernel, not as more patches
to the mechanical port:

1. Build a fixed-length, single-CTA, single-consumer-warpgroup baseline for
   `output_h=False` and `output_final_state=True`.
2. Use one TCGEN05 GEMM at a time with immediate `mbarrier_wait_parity`.
3. Keep `S` and `O` accumulators in TMEM across the operations that naturally
   consume them; avoid fragment->TMEM->fragment round trips in the final design.
4. Add the producer warpgroup and double-buffered shared loads only after the
   single-consumer version passes correctness.
5. Split policies by shape class:
   - TP8/small head: prioritize low overhead and enough CTAs.
   - TP2/TP1/large head: prioritize persistent TMEM state and high tensor-core
     utilization.
6. Reintroduce CP/varlen only after fixed-length paths are stable.

This is the route toward best performance. The mechanical port remains useful
only as a compiler probe for TileLang 0.1.9 TCGEN05 constraints.

## Current B200 Finding

For `B=1, Hk=64, Hv=64`, the KKT-only TCGEN05 prototype is currently slower than
the compatibility path. Example `T=32768`:

- FLA total: ~3.23 ms
- FlashQLA total with KKT prototype + compatibility fused GDR: ~12.85 ms
- FlashQLA `kkt_solve`: ~5.45 ms
- FlashQLA `fused_chunk_gdr_fwd`: ~7.40 ms

Therefore the current KKT prototype should not be enabled for performance runs
or SGLang experiments. It is a proof that TileLang 0.1.9 can emit TCGEN05/TMEM
for this operator family. The performance path must focus on a new native fused
forward kernel and a redesigned KKT solve that avoids TMEM copy-back and slow
FP32 helper loops.

Use the variant runner to compare clean subprocesses:

```bash
python scripts/bench_blackwell_variants.py --variants compat,fwd,kkt --set profile
```

For dispatch debugging:

```bash
FLASHQLA_DEBUG_BLACKWELL_DISPATCH=1 FLASHQLA_BLACKWELL_NATIVE=1 \
  FLASHQLA_BLACKWELL_NATIVE_KERNELS=fwd \
  python tests/test_gdr.py --set profile --skip-bwd --hide-lat
```

Native fused forward has a second guard because the first fixed-length
single-consumer prototype can still hang while TCGEN05/TMEM usage is being
isolated:

```bash
FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE=1 FLASHQLA_BLACKWELL_NATIVE=1 \
  FLASHQLA_BLACKWELL_NATIVE_KERNELS=fwd \
  python tests/test_gdr.py --set profile --skip-bwd --no-cp --hide-acc
```

To isolate native fused-forward runtime hangs, limit the number of chunks per
CTA:

```bash
FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE=1 FLASHQLA_BLACKWELL_NATIVE=1 \
  FLASHQLA_BLACKWELL_NATIVE_KERNELS=fwd FLASHQLA_BLACKWELL_FWD_MAX_ITERS=1 \
  python tests/test_gdr.py --set profile --skip-bwd --no-cp --hide-acc
```

If `MAX_ITERS=1` works but full length hangs, test the looped barrier path:

```bash
FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE=1 FLASHQLA_BLACKWELL_NATIVE=1 \
  FLASHQLA_BLACKWELL_NATIVE_KERNELS=fwd FLASHQLA_BLACKWELL_FWD_MAX_ITERS=2 \
  python tests/test_gdr.py --set profile --skip-bwd --no-cp --hide-acc
```

Any benchmark collected with `FLASHQLA_BLACKWELL_FWD_MAX_ITERS>0` is a runtime
debug probe only and is not a valid full-sequence performance result.

Before debugging that kernel, run the minimal TCGEN05 smoke test:

```bash
python scripts/smoke_tcgen05.py
```

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
