// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace lmcache::lmcache_native {

// Non-temporal stores only pay off for the multi-MiB Device-DAX gather
// segments. Keep smaller copies on the platform's tuned libc implementation.
inline constexpr std::size_t kDevDaxMemcpyNtMinBytes = 1U << 20;

// Copy a Device-DAX gather segment into its DRAM bounce range. On x86 this
// dispatches to an AVX2 non-temporal-store implementation when the running CPU
// supports AVX2; every other build/CPU falls back to std::memmove.
void devdax_memcpy_nt(std::uintptr_t dst, std::uintptr_t src,
                      std::size_t nbytes);

// Backend that would be used for a sufficiently large, non-overlapping copy.
// This is exposed for status reporting and operational verification.
const char* devdax_memcpy_nt_backend();

}  // namespace lmcache::lmcache_native
