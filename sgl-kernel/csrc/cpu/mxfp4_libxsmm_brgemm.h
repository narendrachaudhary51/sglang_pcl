#pragma once

// ============================================================================
// libxsmm MXFP4 (e2m1 weight + e8m0 micro-scale)  x  BF16 activation  GEMM
// backend.
// ----------------------------------------------------------------------------
// This is an optional drop-in replacement for the default oneDNN/AMX tinygemm
// path used by the mxfp4 GEMM and fused-MoE kernels. It dispatches a single
// mixed-precision AMX GEMM micro-kernel (A = MXFP4 weight in VNNI2 layout,
// B = BF16 activation, C = F32 accumulator) via libxsmm.
//
// Unlike the fp8 block-scaled path, the per-32-element e8m0 micro-scale is
// applied INTERNALLY by libxsmm via the a.tertiary scale pointer, so there is
// NO manual accumulator rescale loop: one GEMM over the full K produces the
// final F32 result, which the caller converts to bf16 (+bias).
//
// All public entry points are safe to call regardless of whether libxsmm is
// compiled in. When SGL_KERNEL_HAS_LIBXSMM is not defined, enabled() always
// returns false and the gemm entry point is a no-op returning false, so callers
// transparently fall back to the default path.
//
// Runtime knobs (read once, cached):
//   SGLANG_CPU_MXFP4_LIBXSMM          (0/1) master toggle (default 0 = off)
//   SGLANG_CPU_MXFP4_LIBXSMM_TILECFG  (0/1) hoist AMX tile config (default 1)
//   SGLANG_CPU_MXFP4_LIBXSMM_BR_UNROLL (int) batch-reduce unroll factor:
//        <0 (default) -> fully unroll all K/32 BR blocks (matches the standalone
//                        reference; straight-line weight streaming, best for
//                        small-M/decode where the kernel is issue-bound);
//         0           -> runtime-counted BR loop (no unroll);
//        >0           -> explicit unroll factor.
//   SGLANG_CPU_MXFP4_LIBXSMM_VERBOSE  (0/1) log kernel JIT / exec stats (def 0)
// ============================================================================

#include <ATen/ATen.h>

#include <cstdint>
#include <cstdlib>

namespace sgl_mxfp4_libxsmm {

inline int env_int(const char* name, int def) {
  const char* v = std::getenv(name);
  return (v != nullptr) ? std::atoi(v) : def;
}

// Master toggle. Cached on first call (no per-GEMM env lookup overhead).
inline bool enabled() {
#ifdef SGL_KERNEL_HAS_LIBXSMM
  static const int v = env_int("SGLANG_CPU_MXFP4_LIBXSMM", 0);
  return v != 0;
#else
  return false;
#endif
}

inline bool cfg_cached_tilecfg() {
  static const int v = env_int("SGLANG_CPU_MXFP4_LIBXSMM_TILECFG", 1);
  return v != 0;
}

// Batch-reduce unroll factor for the STRIDE BRGEMM over the K/32 micro-scale
// blocks. The standalone reference fully unrolls (passes the full br_count as
// the unroll hint), which generates a straight-line, fully-pipelined weight
// stream; passing 0 instead emits a runtime-counted BR loop whose per-block
// branch throttles the weight TILELOADD rate at small M (n=1), leaving DRAM
// bandwidth on the table. Default (<0) = fully unroll to match the reference.
// Returns the raw configured value; the dispatcher maps <0 to the full br_count.
inline int cfg_br_unroll() {
  static const int v = env_int("SGLANG_CPU_MXFP4_LIBXSMM_BR_UNROLL", -1);
  return v;
}

inline bool cfg_verbose() {
  static const int v = env_int("SGLANG_CPU_MXFP4_LIBXSMM_VERBOSE", 0);
  return v != 0;
}

// Write the GEMM result straight to the bf16 output (libxsmm downconverts in the
// store from its internal F32 accumulator), skipping the per-thread F32 Ctmp
// scratch and the F32->bf16 copy epilogue. Only used when the output is bf16 and
// there is no bias. Default 1 (on). Set 0 to force the F32-scratch + convert
// path (the original behavior) for A/B comparison.
inline bool cfg_bf16_out() {
  static const int v = env_int("SGLANG_CPU_MXFP4_LIBXSMM_BF16_OUT", 1);
  return v != 0;
}

// Hot-path 1-entry thread-local kernel-pointer memo: every full-N tile in a GEMM
// shares the same kernel key, so caching the last (key,pointer) avoids re-hashing
// the 7-int key + an unordered_map probe on every tile (the standalone keeps a
// single resolved kernel pointer). Default 1 (on). Set 0 to always go through the
// unordered_map for A/B comparison.
inline bool cfg_ptr_cache() {
  static const int v = env_int("SGLANG_CPU_MXFP4_LIBXSMM_PTR_CACHE", 1);
  return v != 0;
}

// True when the libxsmm bf16-direct path will own EVERY tile of this dense GEMM,
// so the caller can skip allocating the (otherwise unused) F32/unpack scratch:
//   - backend enabled and bf16-direct output selected,
//   - output is bf16 and there is no bias (bf16-direct requirement),
//   - N is an exact multiple of block_n so there are no remainder tiles that
//     would fall back to the default unpack + F32 path.
// When the backend is enabled the brgemm path is taken for every M (small-M is
// force-routed through libxsmm because the weights are stored in VNNI2 layout;
// see the use_brgemm guards in gemm_fp8.cpp / moe_fp8.cpp), so M does not gate
// this decision. block_n is block_size_n() passed in by the caller (gemm.h is
// not visible here).
inline bool dense_skip_scratch(int64_t M, int64_t N, bool has_bias, bool out_is_bf16, int block_n) {
#ifdef SGL_KERNEL_HAS_LIBXSMM
  if (!enabled() || !cfg_bf16_out()) return false;
  if (!out_is_bf16 || has_bias) return false;
  if (block_n <= 0 || (N % block_n) != 0) return false;
  (void)M;
  return true;
#else
  (void)M;
  (void)N;
  (void)has_bias;
  (void)out_is_bf16;
  (void)block_n;
  return false;
#endif
}

#ifdef SGL_KERNEL_HAS_LIBXSMM

// Pre-compile (JIT) every GEMM micro-kernel shape that the mxfp4 path will
// request, serially on the calling thread.
//
// libxsmm's first-dispatch JIT mutates a process-global code registry and is
// NOT safe to run concurrently with kernel execution on other threads. Callers
// MUST invoke this once, on a single thread, BEFORE entering the OpenMP region
// that calls brgemm_mxfp4_bf16, so that inside the region the kernel lookup is
// a pure (lock-free) cache lookup and never triggers a concurrent JIT.
//
// The per-tile token count (M) varies between callers: dense/shared-expert use
// uniform BLOCK_M tiles (+ a remainder), while fused-MoE uses a different
// m_size per expert block. The caller passes the exact set of M tile sizes it
// will use via [m_sizes, m_sizes + num_m_sizes]; N is tiled here the same way
// the kernels tile it (BLOCK_N columns), and K is the full reduction depth.
//
// c_bf16 / ldc: when c_bf16 is true the kernels are JIT'd to write the result
// directly as bf16 (skipping the F32 scratch), using ldc as the output leading
// dim; otherwise (default) they accumulate into an F32 scratch with ldc ==
// BLOCK_N. ldc is ignored when c_bf16 is false.
void prewarm_mxfp4(
    const int* m_sizes, int num_m_sizes, int N, int K, int lda_act, bool c_bf16 = false, int ldc = 0);

// MXFP4 weight x BF16 activation GEMM (single full-K reduction).
//
//   A_act    : BF16 activation, row-major [M, K], leading dim lda_act.
//   B_weight : MXFP4 weight prepacked VNNI2 as [K/2, n_size] bytes (2 e2m1
//              nibbles per byte, consecutive K), leading dim ldb_weight.
//   C_f32    : F32 output accumulator, row-major [M, ldc], leading dim ldc.
//   scale    : per-32-K-block e8m0 weight scales, layout [K/32, n_size],
//              leading dim (per 32-K block) == ldb_weight.
//
// Computes C[m, j] = sum_k A[m, k] * mxfp4_dequant(B, scale)[k, j], where the
// e8m0 micro-scale is applied internally by libxsmm per 32-element K block.
//
// Returns true if the libxsmm kernel ran, false if it could not be dispatched
// (shape not pre-warmed, lookup miss, or n_size != BLOCK_N) so the caller falls
// back to the default unpack + AMX path.
bool brgemm_mxfp4_bf16(
    const at::BFloat16* __restrict__ A_act,
    const uint8_t* __restrict__ B_weight,
    float* __restrict__ C_f32,
    const uint8_t* __restrict__ scale,
    int M,
    int n_size,
    int K,
    int lda_act,
    int ldb_weight,
    int ldc);

// Same as brgemm_mxfp4_bf16 but writes the result DIRECTLY as bf16 (libxsmm
// downconverts from its internal F32 accumulator in the store), so no F32 Ctmp
// scratch and no convert epilogue are needed. C_bf16 is the real output tile
// (row-major [M, n_size], leading dim ldc == output row stride). Requires the
// matching c_bf16 kernel to have been prewarmed; returns false on a miss /
// remainder tile so the caller falls back.
bool brgemm_mxfp4_bf16_out(
    const at::BFloat16* __restrict__ A_act,
    const uint8_t* __restrict__ B_weight,
    at::BFloat16* __restrict__ C_bf16,
    const uint8_t* __restrict__ scale,
    int M,
    int n_size,
    int K,
    int lda_act,
    int ldb_weight,
    int ldc);

// Issue the AMX tile-release for the current thread. Call at the end of an OMP
// parallel region that used brgemm_mxfp4_bf16 (mirrors brgemm_release).
void region_end();

#else  // !SGL_KERNEL_HAS_LIBXSMM

inline void prewarm_mxfp4(const int*, int, int, int, int, bool, int) {}

inline bool brgemm_mxfp4_bf16(
    const at::BFloat16*,
    const uint8_t*,
    float*,
    const uint8_t*,
    int,
    int,
    int,
    int,
    int,
    int) {
  return false;
}

inline bool brgemm_mxfp4_bf16_out(
    const at::BFloat16*,
    const uint8_t*,
    at::BFloat16*,
    const uint8_t*,
    int,
    int,
    int,
    int,
    int,
    int) {
  return false;
}

inline void region_end() {}

#endif  // SGL_KERNEL_HAS_LIBXSMM

}  // namespace sgl_mxfp4_libxsmm
