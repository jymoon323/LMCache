// SPDX-License-Identifier: Apache-2.0

#include "nt_memcpy.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
  #define LMCACHE_HAS_X86_AVX2_DISPATCH 1
  #include <immintrin.h>
#else
  #define LMCACHE_HAS_X86_AVX2_DISPATCH 0
#endif

namespace lmcache::lmcache_native {
namespace {

bool ranges_overlap(std::uintptr_t dst, std::uintptr_t src,
                    std::size_t nbytes) {
  if (dst == src) {
    return true;
  }
  // Subtraction after ordering avoids overflow near UINTPTR_MAX.
  return dst < src ? src - dst < nbytes : dst - src < nbytes;
}

#if LMCACHE_HAS_X86_AVX2_DISPATCH

bool cpu_supports_avx2() {
  static const bool supported = [] {
    __builtin_cpu_init();
    return static_cast<bool>(__builtin_cpu_supports("avx2"));
  }();
  return supported;
}

__attribute__((target("avx2"))) void avx2_nt_copy(void* dst_void,
                                                  const void* src_void,
                                                  std::size_t nbytes) {
  auto* dst = static_cast<unsigned char*>(dst_void);
  auto* src = static_cast<const unsigned char*>(src_void);

  // _mm256_stream_si256 itself requires 32-byte alignment. Align to a full
  // cache line so the temporal prefix/tail never shares a line with an NT
  // store; the hot, multi-MiB body then bypasses the cache cleanly.
  const std::size_t prefix = std::min<std::size_t>(
      (64U - (reinterpret_cast<std::uintptr_t>(dst) & 63U)) & 63U, nbytes);
  if (prefix != 0) {
    std::memcpy(dst, src, prefix);
    dst += prefix;
    src += prefix;
    nbytes -= prefix;
  }

  // Eight vectors per iteration provide enough independent loads to sustain
  // sequential Device-DAX reads without requiring AVX-512.
  while (nbytes >= 256) {
    const __m256i v0 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 0));
    const __m256i v1 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
    const __m256i v2 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 64));
    const __m256i v3 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 96));
    const __m256i v4 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 128));
    const __m256i v5 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 160));
    const __m256i v6 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 192));
    const __m256i v7 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 224));
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 0), v0);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 32), v1);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 64), v2);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 96), v3);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 128), v4);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 160), v5);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 192), v6);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 224), v7);
    dst += 256;
    src += 256;
    nbytes -= 256;
  }

  if (nbytes != 0) {
    std::memcpy(dst, src, nbytes);
  }

  // Each worker must publish its own weakly ordered streaming stores before
  // its DevDaxCopyBatch settles and the caller can submit the bounce-to-GPU
  // DMA.
  _mm_sfence();
  _mm256_zeroupper();
}

#endif  // LMCACHE_HAS_X86_AVX2_DISPATCH

}  // namespace

const char* devdax_memcpy_nt_backend() {
#if LMCACHE_HAS_X86_AVX2_DISPATCH
  if (cpu_supports_avx2()) {
    return "avx2_nt";
  }
#endif
  return "libc";
}

void devdax_memcpy_nt(std::uintptr_t dst, std::uintptr_t src,
                      std::size_t nbytes) {
  if (nbytes == 0) {
    return;
  }
  if (dst == 0 || src == 0) {
    throw std::invalid_argument(
        "devdax_memcpy_nt requires non-null addresses for a non-empty copy");
  }
  if (dst == src) {
    return;
  }

  // NT stores are not overlap-safe, and they lose to tuned libc for the small
  // fragments that occasionally arise at object boundaries.
  if (nbytes < kDevDaxMemcpyNtMinBytes || ranges_overlap(dst, src, nbytes)) {
    std::memmove(reinterpret_cast<void*>(dst),
                 reinterpret_cast<const void*>(src), nbytes);
    return;
  }

#if LMCACHE_HAS_X86_AVX2_DISPATCH
  if (cpu_supports_avx2()) {
    avx2_nt_copy(reinterpret_cast<void*>(dst),
                 reinterpret_cast<const void*>(src), nbytes);
    return;
  }
#endif
  std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
               nbytes);
}

}  // namespace lmcache::lmcache_native
