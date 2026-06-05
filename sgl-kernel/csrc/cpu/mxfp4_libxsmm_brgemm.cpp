// ============================================================================
// libxsmm MXFP4 (e2m1 weight + e8m0 micro-scale) x BF16 GEMM backend
// implementation.
//
// Only compiled into the extension when SGL_KERNEL_HAS_LIBXSMM is defined
// (i.e. LIBXSMM_ROOT was provided at configure time). See mxfp4_libxsmm_brgemm.h
// for the public interface and the runtime knobs.
//
// libxsmm interprets A as MXFP4-VNNI2 (flag INTERPRETE_A_AS_MXFP4_VNNI2): the
// weight is a byte container holding two e2m1 nibbles per byte (consecutive K),
// and the per-32-K e8m0 micro-scale is supplied through gemm_param.a.tertiary
// and applied internally.
//
// The reduction is expressed as a STRIDE batch-reduce GEMM (BRGEMM) over the
// e8m0 micro-scale groups: the per-block contraction depth is BK = 32 (the
// e8m0 group size) and the batch-reduce count is K / 32. Each BR block consumes
// one 32-K weight slab and one e8m0 byte per weight column; libxsmm auto-strides
// the weight (a.primary), the activation (b.primary) and the scale (a.tertiary)
// across the K/32 blocks and accumulates into a single F32 tile. Expressing it
// this way (rather than one full-K GEMM) is what selects the optimized AMX
// micro-kernel instead of libxsmm's scalar reference path.
// ============================================================================

#include "mxfp4_libxsmm_brgemm.h"

#ifdef SGL_KERNEL_HAS_LIBXSMM

#include <libxsmm.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "gemm.h"  // block_size_n()

namespace sgl_mxfp4_libxsmm {
namespace {

// MXFP4 e8m0 micro-scale group size, i.e. the per-BR-block contraction depth.
// The full-K reduction is run as K / BK batch-reduce blocks.
constexpr int BK = 32;

// JIT-compiled kernels for one micro-GEMM shape.
struct Kernels {
  libxsmm_gemmfunction gemm = nullptr;     // MXFP4 x BF16 -> F32 GEMM
  libxsmm_tilecfgfunction cfg = nullptr;   // AMX tile setup   (cached tilecfg only)
  libxsmm_tilecfgfunction rls = nullptr;   // AMX tile release (cached tilecfg only)
};

struct Key {
  int m;          // = n_size  (weight tile columns, libxsmm A rows)
  int n;          // = M       (activation rows / tokens)
  int k;          // = K       (full reduction depth)
  int lda;        // weight leading dim (bytes per K-pair row)
  int ldb;        // activation leading dim
  int ldc;        // output leading dim
  int cached_tc;  // hoisted tile config?
  int c_bf16;     // C dtype: 1 = bf16 (direct store), 0 = f32 scratch

  bool operator==(const Key& o) const {
    return m == o.m && n == o.n && k == o.k && lda == o.lda && ldb == o.ldb && ldc == o.ldc &&
           cached_tc == o.cached_tc && c_bf16 == o.c_bf16;
  }
};

struct KeyHash {
  size_t operator()(const Key& key) const {
    size_t h = 1469598103934665603ull;
    auto mix = [&h](int v) {
      h ^= static_cast<size_t>(static_cast<unsigned>(v));
      h *= 1099511628211ull;
    };
    mix(key.m);
    mix(key.n);
    mix(key.k);
    mix(key.lda);
    mix(key.ldb);
    mix(key.ldc);
    mix(key.cached_tc);
    mix(key.c_bf16);
    return h;
  }
};

// Global registry. libxsmm's first-dispatch JIT path mutates a global code
// registry and is NOT thread-safe, so all dispatch happens under this mutex.
// Map nodes are never erased, so pointers handed out stay valid for the process
// lifetime, allowing a lock-free thread-local memo on the hot path.
std::mutex g_mutex;
std::unordered_map<Key, Kernels, KeyHash> g_cache;

// Currently active hoisted tile configuration for THIS thread (or nullptr).
thread_local const Kernels* g_active_cfg = nullptr;

// ---------------------------------------------------------------------------
// Hot-path EXECUTION counters (proof that the libxsmm kernel actually ran).
//
// Cosine parity alone does NOT prove libxsmm executed: a silent prewarm-key /
// hot-path-key mismatch makes the entry point bail to the default GEMM, which
// yields IDENTICAL output and default-like timing. These atomics record how
// many times the libxsmm micro-kernel was actually issued ("exec") vs how many
// times the hot path fell back on a lookup miss ("fallback"). A run with
// exec > 0 and fallback == 0 is hard proof the op is fully on libxsmm. They are
// only touched when VERBOSE is set (an unconditional shared atomic in the hot
// path is a multi-thread scaling disaster), and dumped at process exit.
// ---------------------------------------------------------------------------
std::atomic<uint64_t> g_exec{0};      // tile-GEMMs issued
std::atomic<uint64_t> g_fallback{0};  // calls that bailed out

void dump_exec_stats() {
  std::fprintf(
      stderr,
      "[sgl mxfp4 libxsmm] EXEC STATS  exec=%llu fallback=%llu\n",
      static_cast<unsigned long long>(g_exec.load()),
      static_cast<unsigned long long>(g_fallback.load()));
}

// ---------------------------------------------------------------------------
// Per-GEMM-shape prewarm memo (mirrors the fp8 backend). dispatch_kernels()
// caches every JIT'd micro-kernel for the process lifetime, but the prewarm
// enumeration that (re)discovers which tiles a GEMM needs would otherwise re-run
// on EVERY call. This memo records the GEMM shapes already prewarmed so repeat
// calls of the same shape return after a single lock-free thread-local lookup.
// ---------------------------------------------------------------------------
struct PrewarmKey {
  uint64_t m_hash;  // hash of the m_sizes multiset for this call
  int N;
  int K;
  int lda;
  int cached_tc;

  bool operator==(const PrewarmKey& o) const {
    return m_hash == o.m_hash && N == o.N && K == o.K && lda == o.lda && cached_tc == o.cached_tc;
  }
};

struct PrewarmKeyHash {
  size_t operator()(const PrewarmKey& key) const {
    size_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
      h ^= v;
      h *= 1099511628211ull;
    };
    mix(key.m_hash);
    mix(static_cast<uint64_t>(static_cast<unsigned>(key.N)));
    mix(static_cast<uint64_t>(static_cast<unsigned>(key.K)));
    mix(static_cast<uint64_t>(static_cast<unsigned>(key.lda)));
    mix(static_cast<uint64_t>(static_cast<unsigned>(key.cached_tc)));
    return h;
  }
};

// Global set of prewarmed shapes (guarded by g_mutex). Never erased.
std::unordered_set<PrewarmKey, PrewarmKeyHash> g_prewarmed;

inline uint64_t hash_m_sizes(const int* m_sizes, int num_m_sizes) {
  uint64_t h = 1469598103934665603ull;
  for (int i = 0; i < num_m_sizes; ++i) {
    h ^= static_cast<uint64_t>(static_cast<unsigned>(m_sizes[i]));
    h *= 1099511628211ull;
  }
  return h;
}

// Returns true if this GEMM shape was ALREADY prewarmed (caller skips the tile
// enumeration); otherwise records it and returns false. The hot fast path is
// the lock-free thread-local set; the global set is consulted only the first
// time a thread sees a shape. A (hash) collision is safe: at worst the hot path
// misses lookup_kernels() and falls back to the default GEMM.
inline bool already_prewarmed(const PrewarmKey& key) {
  thread_local std::unordered_set<PrewarmKey, PrewarmKeyHash> tls;
  if (!tls.insert(key).second) {
    return true;  // already prewarmed by this thread
  }
  std::lock_guard<std::mutex> guard(g_mutex);
  return !g_prewarmed.insert(key).second;  // true if another thread did it first
}

void ensure_init() {
  static std::once_flag once;
  std::call_once(once, []() {
    libxsmm_init();
    if (cfg_verbose()) {
      std::atexit(dump_exec_stats);
    }
  });
}

// JIT-compile the kernels for one shape and insert them into the global cache.
// MUST be called serially (this is the only place that triggers libxsmm JIT).
// Returns a stable pointer into the cache (never erased for the process life).
const Kernels* dispatch_kernels(const Key& key) {
  ensure_init();

  std::lock_guard<std::mutex> guard(g_mutex);
  auto found = g_cache.find(key);
  if (found != g_cache.end()) {
    return &found->second;
  }

  // A is a byte container of MXFP4-VNNI2 nibbles; the MXFP4 flag tells libxsmm
  // how to decode it and where to read the e8m0 micro-scale (a.tertiary).
  const libxsmm_datatype a_type = LIBXSMM_DATATYPE_I8;    // weight  (MXFP4-VNNI2)
  const libxsmm_datatype b_type = LIBXSMM_DATATYPE_BF16;  // activation
  // C is either the F32 accumulator scratch or, when c_bf16 is set, the bf16
  // output written directly (libxsmm downconverts from its internal F32 accum in
  // the store). The compute type stays F32 either way.
  const libxsmm_datatype c_type = key.c_bf16 ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32;
  const libxsmm_datatype comp = LIBXSMM_DATATYPE_F32;

  // BRGEMM micro-kernel: the per-block contraction depth is BK (one e8m0 group);
  // the full K reduction runs as br_count = K / BK batch-reduce blocks.
  libxsmm_gemm_shape shape =
      libxsmm_create_gemm_shape(key.m, key.n, BK, key.lda, key.ldb, key.ldc, a_type, b_type, c_type, comp);

  // STRIDE batch-reduce: libxsmm advances each operand by a fixed byte stride
  // per BR block.
  //   A (weight, MXFP4-VNNI2): one 32-K slab = lda columns * BK rows * 0.5 byte
  //     = key.lda * BK / 2 bytes.
  //   B (activation, plain BF16, col-major [k, n] with ldb = K): advance BK
  //     elements along K = BK * sizeof(bf16) bytes.
  // The e8m0 micro-scale (a.tertiary) is auto-strided by libxsmm using lda and
  // the per-block k (BK), one byte per weight column per block, so it needs no
  // explicit stride hint here.
  //
  // br_unroll_hint: the standalone reference passes the FULL br_count so libxsmm
  // emits a fully-unrolled, straight-line BR loop. Passing 0 instead emits a
  // runtime-counted loop whose per-block branch throttles the weight TILELOADD
  // rate at small M (n=1) and leaves DRAM bandwidth on the table. We default to
  // the full unroll (cfg_br_unroll() < 0 -> br_count) to match the reference.
  const int br_count = key.k / BK;
  int br_unroll = cfg_br_unroll();
  if (br_unroll < 0 || br_unroll > br_count) {
    br_unroll = br_count;  // fully unroll (reference behavior)
  }

  libxsmm_gemm_batch_reduce_config brconfig = libxsmm_create_gemm_batch_reduce_config(
      LIBXSMM_GEMM_BATCH_REDUCE_STRIDE,
      /*br_stride_a_hint=*/static_cast<libxsmm_blasint>(key.lda) * BK / 2,
      /*br_stride_b_hint=*/static_cast<libxsmm_blasint>(BK) * sizeof(at::BFloat16),
      /*br_unroll_hint=*/static_cast<unsigned int>(br_unroll));

  const libxsmm_bitfield flags =
      LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI2 | LIBXSMM_GEMM_FLAG_VNNI_A | LIBXSMM_GEMM_FLAG_BETA_0;


  Kernels kernels;
  if (key.cached_tc) {
    const libxsmm_bitfield cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | flags;
    const libxsmm_bitfield rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | flags;
    const libxsmm_bitfield gemm_flags =
        LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | flags;
    kernels.cfg = libxsmm_dispatch_tilecfg_gemm(shape, cfg_flags);
    kernels.rls = libxsmm_dispatch_tilecfg_gemm(shape, rls_flags);
    kernels.gemm = libxsmm_dispatch_brgemm(shape, gemm_flags, LIBXSMM_GEMM_PREFETCH_NONE, brconfig);
  } else {
    kernels.gemm = libxsmm_dispatch_brgemm(shape, flags, LIBXSMM_GEMM_PREFETCH_NONE, brconfig);
  }

  if (kernels.gemm == nullptr) {
    std::fprintf(
        stderr,
        "[sgl mxfp4 libxsmm] FAILED to JIT brgemm m=%d n=%d k=%d (bk=%d br=%d) lda=%d ldb=%d ldc=%d\n",
        key.m,
        key.n,
        key.k,
        BK,
        br_count,
        key.lda,
        key.ldb,
        key.ldc);
  } else if (cfg_verbose()) {
    libxsmm_kernel_info info;
    std::memset(&info, 0, sizeof(info));
    libxsmm_get_kernel_info(reinterpret_cast<const void*>(kernels.gemm), &info);
    std::fprintf(
        stderr,
        "[sgl mxfp4 libxsmm] JIT brgemm m=%d n=%d k=%d (bk=%d br=%d unroll=%d) lda=%d ldb=%d ldc=%d tc=%d "
        "reference_kernel=%u\n",
        key.m,
        key.n,
        key.k,
        BK,
        br_count,
        br_unroll,
        key.lda,
        key.ldb,
        key.ldc,
        key.cached_tc,
        info.is_reference_kernel);
    if (info.is_reference_kernel) {
      std::fprintf(
          stderr,
          "[sgl mxfp4 libxsmm] WARNING: dispatched the SCALAR REFERENCE kernel "
          "(not the optimized AMX path) for m=%d n=%d k=%d\n",
          key.m,
          key.n,
          key.k);
    }
  }

  return &g_cache.emplace(key, kernels).first->second;
}

// Lock-free cache lookup for the hot path. Returns nullptr if the shape was not
// pre-compiled by prewarm_mxfp4; the caller must then fall back rather than JIT,
// because JIT concurrent with execution on other threads is unsafe.
const Kernels* lookup_kernels(const Key& key) {
  // 1-entry thread-local memo: every full-N tile in a GEMM shares the same key,
  // so the previous (key, ptr) almost always hits, turning the per-tile lookup
  // into a struct compare instead of a hash + unordered_map probe (the analog of
  // the standalone's single saved kernel pointer). Toggle with cfg_ptr_cache().
  static const bool ptr_cache = cfg_ptr_cache();
  thread_local bool memo_valid = false;
  thread_local Key memo_key;
  thread_local const Kernels* memo_ptr = nullptr;
  if (ptr_cache && memo_valid && memo_key == key) {
    return memo_ptr;
  }

  thread_local std::unordered_map<Key, const Kernels*, KeyHash> tls;
  const Kernels* result = nullptr;
  auto it = tls.find(key);
  if (it != tls.end()) {
    result = it->second;
  } else {
    const Kernels* ptr = nullptr;
    {
      std::lock_guard<std::mutex> guard(g_mutex);
      auto found = g_cache.find(key);
      if (found != g_cache.end()) {
        ptr = &found->second;
      }
    }
    if (ptr != nullptr) {
      tls.emplace(key, ptr);
    }
    result = ptr;
  }

  if (ptr_cache && result != nullptr) {
    memo_valid = true;
    memo_key = key;
    memo_ptr = result;
  }
  return result;
}

inline void ensure_tilecfg(const Kernels* kernels) {
  if (kernels->cfg != nullptr && g_active_cfg != kernels) {
    kernels->cfg(nullptr);
    g_active_cfg = kernels;
  }
}

// Build the kernel Key for one mxfp4 GEMM tile (used both for prewarming and on
// the hot path so the two stay perfectly in sync). The weight leading dim
// (lda) doubles as the e8m0 scale stride (one byte per weight column per 32-K
// block), so libxsmm indexes scale[s * lda + col] exactly as sglang lays it out
// when n_size == BLOCK_N.
inline Key make_key(int n_size, int M, int K, int lda_act, int ldb_weight, int ldc, int cached_tc, int c_bf16) {
  return Key{n_size, M, K, ldb_weight, lda_act, ldc, cached_tc, c_bf16};
}

}  // namespace

void prewarm_mxfp4(const int* m_sizes, int num_m_sizes, int N, int K, int lda_act, bool c_bf16, int ldc_out) {
  const int cached_tc = cfg_cached_tilecfg() ? 1 : 0;
  const int BLOCK_N = block_size_n();
  const int c_bf16_i = c_bf16 ? 1 : 0;
  // F32 scratch kernels accumulate into a per-tile [M, BLOCK_N] buffer (ldc ==
  // BLOCK_N); bf16-direct kernels store into the real output (ldc == ldc_out).
  const int ldc = c_bf16 ? ldc_out : BLOCK_N;

  // Skip the whole enumeration if this exact GEMM shape was already prewarmed
  // in a previous invocation (one lock-free thread-local lookup).
  const PrewarmKey pk{hash_m_sizes(m_sizes, num_m_sizes), N, K, lda_act ^ (c_bf16_i << 30) ^ (ldc << 1), cached_tc};
  if (already_prewarmed(pk)) {
    return;
  }

  // The libxsmm path only runs on full BLOCK_N tiles (so the weight leading dim
  // == the e8m0 scale stride == BLOCK_N). Every full N tile shares the same
  // shape, so enumerate one kernel per distinct M tile size; K is the single
  // full-depth reduction and the output leading dim is the per-tile F32 scratch
  // stride (BLOCK_N) or, for the bf16-direct path, the real output row stride.
  // If N has no full BLOCK_N tile there is nothing to compile (the hot path will
  // always fall back).
  if (N < BLOCK_N) {
    return;
  }
  for (int mi = 0; mi < num_m_sizes; ++mi) {
    const int M = m_sizes[mi];
    if (M <= 0) {
      continue;
    }
    dispatch_kernels(make_key(BLOCK_N, M, K, lda_act, /*ldb_weight*/ BLOCK_N, ldc, cached_tc, c_bf16_i));
  }
}

namespace {

// Shared hot-path impl for both the F32-scratch and bf16-direct entry points.
// C_ptr is the F32 scratch (c_bf16 == 0) or the bf16 output tile (c_bf16 == 1).
inline bool run_mxfp4_gemm(
    const at::BFloat16* __restrict__ A_act,
    const uint8_t* __restrict__ B_weight,
    void* __restrict__ C_ptr,
    const uint8_t* __restrict__ scale,
    int M,
    int n_size,
    int K,
    int lda_act,
    int ldb_weight,
    int ldc,
    int c_bf16) {
  const int BLOCK_N = block_size_n();

  // libxsmm indexes the e8m0 scale with stride lda (== ldb_weight). sglang lays
  // the scale out with stride BLOCK_N, so the two only agree on a full BLOCK_N
  // tile. Remainder tiles fall back to the default unpack path.
  if (n_size != BLOCK_N || ldb_weight != BLOCK_N) {
    if (cfg_verbose()) {
      g_fallback.fetch_add(1, std::memory_order_relaxed);
    }
    return false;
  }

  const int cached_tc = cfg_cached_tilecfg() ? 1 : 0;

  // Key must match exactly what prewarm_mxfp4 compiled. The kernels are looked
  // up lock-free; a miss means prewarm was skipped, so we bail out to the
  // default path rather than JIT concurrently (which would be unsafe).
  Key key = make_key(n_size, M, K, lda_act, ldb_weight, ldc, cached_tc, c_bf16);
  const Kernels* kernels = lookup_kernels(key);
  if (kernels == nullptr || kernels->gemm == nullptr) {
    if (cfg_verbose()) {
      g_fallback.fetch_add(1, std::memory_order_relaxed);
    }
    return false;
  }
  ensure_tilecfg(kernels);

  // The weight tile already arrives in standard MXFP4-VNNI2 layout (the sglang ->
  // libxsmm re-shuffle is folded into the upfront weight prepack, NOT done here on
  // the timed path). libxsmm reads A directly and applies the per-32-K e8m0
  // micro-scale internally, writing the result (F32 scratch or bf16 direct) to C.
  //
  // libxsmm A = MXFP4 weight tile [n_size, K] (VNNI2 bytes), B = BF16 activation
  // tile [K, M], C = result col-major [n_size, M] == row-major [M, n_size].
  // The reduction runs as a STRIDE BRGEMM over K / BK micro-scale blocks; the
  // batch-reduce count is passed through op.tertiary and the weight/activation/
  // scale bases are auto-strided by the JIT'd strides.
  const unsigned long long br_count = static_cast<unsigned long long>(K / BK);
  libxsmm_gemm_param param;
  std::memset(&param, 0, sizeof(param));
  param.op.tertiary = const_cast<unsigned long long*>(&br_count);
  param.a.primary = const_cast<uint8_t*>(B_weight);
  param.a.tertiary = const_cast<uint8_t*>(scale);
  param.b.primary = const_cast<at::BFloat16*>(A_act);
  param.c.primary = C_ptr;
  kernels->gemm(&param);
  if (cfg_verbose()) {
    g_exec.fetch_add(1, std::memory_order_relaxed);
  }
  return true;
}

}  // namespace

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
    int ldc) {
  return run_mxfp4_gemm(A_act, B_weight, C_f32, scale, M, n_size, K, lda_act, ldb_weight, ldc, /*c_bf16*/ 0);
}

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
    int ldc) {
  return run_mxfp4_gemm(A_act, B_weight, C_bf16, scale, M, n_size, K, lda_act, ldb_weight, ldc, /*c_bf16*/ 1);
}

void region_end() {
  if (g_active_cfg != nullptr) {
    if (g_active_cfg->rls != nullptr) {
      g_active_cfg->rls(nullptr);
    }
    g_active_cfg = nullptr;
  }
}

}  // namespace sgl_mxfp4_libxsmm

#endif  // SGL_KERNEL_HAS_LIBXSMM
