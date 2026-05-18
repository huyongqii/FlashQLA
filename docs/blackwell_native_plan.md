# FlashQLA Blackwell Native Kernel Plan

## Current State

FlashQLA has an experimental Blackwell native fixed-length forward path in
`chunk/blackwell/fused_fwd_native.py` and a native KKT path in
`chunk/blackwell/kkt_solve.py`. These kernels explicitly use TileLang
`T.tcgen05_gemm`, TMEM, and mbarriers. They have been validated on B200/B300
for the stable no-CP path with SASS inspection showing Blackwell tensor-core
instructions.

The stable production candidate is the AG forward kernel plus fixed-fast KKT.
Failed exploratory paths that reused precomputed P/Pg, chunk-parallel global
state materialization, or the mechanical Hopper pipeline port have been removed
from the runnable benchmark policy set because they either failed correctness or
were substantially slower on B300.

## Dispatch Contract

- `chunk/hopper/*`: existing Hopper-compatible TileLang implementation.
- `chunk/blackwell/fused_fwd_native.py`: stable experimental native forward
  candidate for fixed-length, no-CP inference.
- `chunk/blackwell/kkt_solve.py`: native fixed-fast KKT candidate.
- `FLASHQLA_REQUIRE_BLACKWELL_NATIVE=1`: fail when a requested Blackwell kernel
  would use Hopper-compatible fallback.
- `FLASHQLA_BLACKWELL_NATIVE=1`: enable experimental native dispatch.
- `FLASHQLA_BLACKWELL_NATIVE_KERNELS=fwd,kkt`: enable the current native
  forward and KKT candidates. Kernels not listed here remain on the
  Hopper-compatible path.
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

The active native branch is no longer the mechanical Hopper port. It is the
fixed-length AG forward kernel in `fused_fwd_native.py`, paired with fixed-fast
KKT. The kernel uses one CTA per `(batch, value head, DV block)` and scans chunks
serially inside that CTA. This is correct and fast enough for TP1/Hv64, but it
does not create enough CTAs for Qwen397 TP2/TP4/TP8 on B300.

TileLang 0.1.9 is sufficient for the current TCGEN05 baseline:

- `T.tcgen05_gemm`
- `T.alloc_tmem`
- `T.mbarrier_wait_parity`
- shared/TMEM copies

It has not been sufficient so far for a robust high-performance Blackwell
producer/consumer pipeline in this recurrent operator. The failed experiments
showed fragile behavior around operand layout, P/Pg reuse, CP, and mechanical
Hopper pipeline translation.

To bind SASS inspection to the latest benchmark instead of old TVM cache
directories, prefer:

```bash
python scripts/inspect_blackwell_mma.py --no-run --latest-tvm-dir
# or, immediately after a benchmark:
python scripts/inspect_blackwell_mma.py --no-run --since-minutes 5
```

Run the stable native candidate with:

```bash
FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE=1 \
FLASHQLA_BLACKWELL_NATIVE=1 \
FLASHQLA_BLACKWELL_NATIVE_KERNELS=fwd,kkt \
FLASHQLA_BLACKWELL_FWD_POLICY=native \
FLASHQLA_BLACKWELL_BLOCK_DV=64 \
FLASHQLA_BLACKWELL_FWD_THREADS=256 \
FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS=load,h \
python tests/test_gdr.py --set profile --skip-bwd --no-cp
```

`FLASHQLA_BLACKWELL_TMEM_WIDTH` defaults to `128`. Use
`FLASHQLA_BLACKWELL_TMEM_WIDTH=exact` to test exact `block_DV`-wide TMEM
accumulators. The exact path is kept as an A/B option because B300 measurements
did not show a structural win for Qwen397 TP2/TP4/TP8.

Do not benchmark this native path with `--with-cp` yet. The existing CP
preprocess/fused-output combination is not a valid Blackwell native performance
path for the current forward kernel: repeated correctness runs show final state
can be close while output is wrong at early chunks. CP must be reintroduced as a
separate state-prefix design, not by passing the current CP sequence map into
the AG kernel.

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
- Do not materialize all intermediate chunk states globally just to create
  parallelism; B300 measurements showed that this loses badly.
- Do not continue the precomputed P/Pg path until TileLang operand-layout and
  synchronization behavior is understood at the TCGEN05/TMEM level.
- Split the next major optimization into a state-prefix/CP design that creates
  more independent chunk work for TP2/TP4/TP8 without writing full per-chunk
  `DK x DV` states to global memory.
- Keep the current AG native path as the baseline for TP1 and as the correctness
  reference for new Blackwell experiments.

TCGEN05 mbarrier reuse note: the native fwd prototype uses 4 mbarrier slots per
GEMM site and flips wait parity by reuse round:

```text
slot = chunk_idx % 4
phase = (chunk_idx // 4) % 2
```

This was introduced after `MAX_ITERS=4` passed but `MAX_ITERS=8` hung with fixed
phase 0, indicating that barrier reuse needs explicit phase progression.

## Forward Rewrite Strategy

The current AG baseline already implements the first stable fixed-length
TCGEN05 path. The next large optimization should not be another local reorder or
barrier sweep. It should address the B300 occupancy issue directly:

1. Define a per-segment summary for the recurrent state update:
   `S_out = alpha * S_in + delta`.
2. Compute summaries for multiple chunk ranges in parallel.
3. Prefix-scan those summaries to recover each segment's initial state.
4. Run the existing AG forward on each segment with the recovered state.
5. Keep summaries compact enough that global memory traffic is far below the
   rejected chunk-parallel full-state materialization.

This is algorithmically closer to the existing CP idea than to attention-style
TMA pipelining. Attention kernels pipeline because each tile is mostly
independent once the softmax state is carried. GDR has a true recurrent state,
so extra parallelism requires a scan/correction step, not only producer/consumer
overlap inside one CTA.

## Current B300 Finding

For Qwen397-style no-CP fixed-length inference, the current native path is
competitive for TP1 but slower for small value-head counts:

- TP1/Hv64: about `1.06x-1.11x` FLA.
- TP2/Hv32: about `0.75x-0.79x` FLA.
- TP4/Hv16: about `0.67x-0.70x` FLA.
- TP8/Hv8: about `0.64x-0.67x` FLA.

The kernel breakdown shows KKT is not the main bottleneck. The fused GDR stage
dominates, and for small head counts the problem is insufficient parallel work
on B300 rather than a single obvious TCGEN05 micro-tuning issue.

For dispatch debugging:

```bash
FLASHQLA_DEBUG_BLACKWELL_DISPATCH=1 FLASHQLA_BLACKWELL_NATIVE=1 \
  FLASHQLA_BLACKWELL_NATIVE_KERNELS=fwd \
  python tests/test_gdr.py --set profile --skip-bwd --hide-lat
```

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
