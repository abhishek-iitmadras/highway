// Copyright 2024 Google LLC
// Copyright 2025 Fujitsu India Pvt Ltd. (talk to Ragesh-pending)
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Per-target include guard
#if defined(HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_INL_H_
#undef HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_INL_H_
#endif

#include <cstddef>
#include <cstdint>

#include "hwy/highway.h"

// Architecture detection for 64-bit scalar fallback
#ifndef HWY_INTDIV_SCALAR64
  #if (HWY_TARGET == HWY_NEON) || (HWY_TARGET == HWY_PPC8) || (HWY_TARGET == HWY_VSX)
    #define HWY_INTDIV_SCALAR64 1
  #else
    #define HWY_INTDIV_SCALAR64 0
  #endif
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Integer division by invariant integers using multiplication.
// Based on T. Granlund and P. L. Montgomery, "Division by invariant integers
// using multiplication" (PLDI 1994).
// https://gmplib.org/~tege/divcnst-pldi94.pdf

// ============================================================================
// Type traits for wider multiplier types
// ============================================================================

// For division, the multiplier needs to be wider than the input type
// to avoid truncation of the magic constant
template <typename T>
struct MulType {
  using type = T;
};

// 8-bit types need 16-bit multipliers
template <>
struct MulType<uint8_t> {
  using type = uint16_t;
};

template <>
struct MulType<int8_t> {
  using type = int16_t;
};

// 16-bit types need 32-bit multipliers
template <>
struct MulType<uint16_t> {
  using type = uint32_t;
};

template <>
struct MulType<int16_t> {
  using type = int32_t;
};

// 32-bit and 64-bit use same-width multipliers (special handling)
template <>
struct MulType<uint32_t> {
  using type = uint32_t;
};

template <>
struct MulType<int32_t> {
  using type = int32_t;
};

template <>
struct MulType<uint64_t> {
  using type = uint64_t;
};

template <>
struct MulType<int64_t> {
  using type = int64_t;
};

template <typename T>
using MulType_t = typename MulType<T>::type;

// ============================================================================
// Divisor parameter structures - separate for unsigned and signed
// ============================================================================

// Parameters for unsigned division
template <typename T>
struct DivisorParamsU {
  MulType_t<T> multiplier;
  int shift1;
  int shift2;
  bool is_pow2;
  int pow2_shift;  // Only valid if is_pow2
  T divisor;        // NEW: original divisor for scalar fallback
};

// Parameters for signed division
template <typename T>
struct DivisorParamsS {
  MulType_t<T> multiplier;
  int shift;
  T divisor;  // Original divisor for floor division
  bool is_pow2;
  int pow2_shift;  // Only valid if is_pow2
};

// ============================================================================
// Helper functions
// ============================================================================

namespace detail {

// Check if value is a power of 2
template <typename T>
HWY_INLINE bool IsPow2(T x) {
  return x > 0 && (x & (x - 1)) == 0;
}

// Count trailing zeros (for power-of-2 divisors)
HWY_INLINE int CountTrailingZeros32(uint32_t x) {
  if (x == 0) return 32;
#if HWY_COMPILER_GCC_ACTUAL || HWY_COMPILER_CLANG
  return __builtin_ctz(x);
#elif HWY_COMPILER_MSVC
  unsigned long index;
  _BitScanForward(&index, x);
  return static_cast<int>(index);
#else
  int count = 0;
  while ((x & 1) == 0) {
    x >>= 1;
    count++;
  }
  return count;
#endif
}

HWY_INLINE int CountTrailingZeros64(uint64_t x) {
  if (x == 0) return 64;
#if HWY_COMPILER_GCC_ACTUAL || HWY_COMPILER_CLANG
  return __builtin_ctzll(x);
#elif HWY_COMPILER_MSVC && HWY_ARCH_X86_64
  unsigned long index;
  _BitScanForward64(&index, x);
  return static_cast<int>(index);
#else
  uint32_t lo = static_cast<uint32_t>(x);
  if (lo != 0) {
    return CountTrailingZeros32(lo);
  }
  return 32 + CountTrailingZeros32(static_cast<uint32_t>(x >> 32));
#endif
}

HWY_INLINE unsigned LeadingZeroCount32(uint32_t x) {
  if (x == 0) return 32;
#if HWY_COMPILER_GCC_ACTUAL || HWY_COMPILER_CLANG
  return static_cast<unsigned>(__builtin_clz(x));
#elif HWY_COMPILER_MSVC
  unsigned long index;
  _BitScanReverse(&index, x);
  return 31 - static_cast<unsigned>(index);
#else
  unsigned n = 0;
  if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
  if (x <= 0x00FFFFFF) { n += 8;  x <<= 8; }
  if (x <= 0x0FFFFFFF) { n += 4;  x <<= 4; }
  if (x <= 0x3FFFFFFF) { n += 2;  x <<= 2; }
  if (x <= 0x7FFFFFFF) { n += 1; }
  return n;
#endif
}

HWY_INLINE unsigned LeadingZeroCount64(uint64_t x) {
  if (x == 0) return 64;
#if HWY_COMPILER_GCC_ACTUAL || HWY_COMPILER_CLANG
  return static_cast<unsigned>(__builtin_clzll(x));
#elif HWY_COMPILER_MSVC && HWY_ARCH_X86_64
  unsigned long index;
  _BitScanReverse64(&index, x);
  return 63 - static_cast<unsigned>(index);
#else
  uint32_t hi = static_cast<uint32_t>(x >> 32);
  if (hi != 0) {
    return LeadingZeroCount32(hi);
  }
  return 32 + LeadingZeroCount32(static_cast<uint32_t>(x));
#endif
}

HWY_INLINE uint64_t DivideHighBy(uint64_t high, uint64_t divisor) {
  HWY_DASSERT(divisor != 0);

#if defined(__SIZEOF_INT128__)
  using uint128_t = unsigned __int128;
  return static_cast<uint64_t>((static_cast<uint128_t>(high) << 64) / divisor);

#elif HWY_COMPILER_MSVC && HWY_ARCH_X86_64 && _MSC_VER >= 1920
  uint64_t rem;
  return _udiv128(high, 0, divisor, &rem);

#else
  // Reduce first: low 64 bits of ((high<<64)/d) depend only on (high % d).
  high %= divisor;
  if (high == 0) return 0;

  const unsigned ldz = /* detail:: */ LeadingZeroCount64(divisor);
  const uint64_t d_norm = divisor << ldz;
  const uint64_t n_norm = high    << ldz;

  const uint32_t dh = static_cast<uint32_t>(d_norm >> 32);
  const uint32_t dl = static_cast<uint32_t>(d_norm & 0xFFFFFFFFu);

  // First digit
  uint64_t qh  = n_norm / dh;
  uint64_t rem = n_norm - qh * dh;

  // Adjust as per Knuth D
  const uint64_t base32 = 1ull << 32;
  while (qh >= base32 || qh * dl > (rem << 32)) {
    --qh;
    rem += dh;
    if (rem >= base32) break;
  }

  // Second digit (matches NumPy; use shift instead of multiply for clarity)
  const uint64_t dividend_pairs = (n_norm << 32) - d_norm * qh;
  const uint32_t ql = static_cast<uint32_t>(dividend_pairs / dh);

  return (qh << 32) | ql;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Uniform scalar logical/arithmetic right shift with target-aware fast paths.
// - Signed lanes => arithmetic shift (as ShiftRight<imm>).
// - Unsigned lanes => logical shift.
// - Clamps sh to [0, bits-1] to preserve signed semantics for large shifts.
// - Uses native per-lane variable shifts only where they're known fast.
template <class D, class V = Vec<D>>
HWY_INLINE V ShiftRightUniform(D d, V v, int sh) {
  using T = TFromD<D>;
  const int bits = int(sizeof(T) * 8);
  if (sh <= 0) return v;
  if (sh >= bits) sh = bits - 1;
  (void)d;

#if HWY_TARGET == HWY_NEON
  return ShiftRightSame(v, sh);
#elif HWY_TARGET == HWY_AVX3
  return ShiftRightSame(v, sh);
#elif HWY_TARGET == HWY_AVX2
  if constexpr (sizeof(T) == 4) {
    return ShiftRightSame(v, sh);
  }
#endif

  // Portable fallback: decompose using only valid immediates for this T.
  if constexpr (sizeof(T) * 8 > 32) {  // 64-bit lanes
    if (sh & 32) v = ShiftRight<32>(v);
  }
  if constexpr (sizeof(T) * 8 > 16) {  // 32/64-bit lanes
    if (sh & 16) v = ShiftRight<16>(v);
  }
  if constexpr (sizeof(T) * 8 > 8) {   // 16/32/64-bit lanes
    if (sh & 8)  v = ShiftRight<8>(v);
  }
  if constexpr (sizeof(T) * 8 > 4) {   // 8/16/32/64-bit lanes
    if (sh & 4)  v = ShiftRight<4>(v);
  }
  if constexpr (sizeof(T) * 8 > 2) {
    if (sh & 2)  v = ShiftRight<2>(v);
  }
  if constexpr (sizeof(T) * 8 > 1) {
    if (sh & 1)  v = ShiftRight<1>(v);
  }
  return v;
}


template <class D, class V = Vec<D>, typename T = TFromD<D>>
HWY_INLINE V ScalarDivPerLane(D d, V dividend, T divisor) {
  // Store vector -> scalars
  const size_t N = Lanes(d);
  HWY_ALIGN T buf[HWY_MAX_BYTES / sizeof(T)];
  Store(dividend, d, buf);
  for (size_t i = 0; i < N; ++i) {
    buf[i] = static_cast<T>(buf[i] / divisor);  // truncation semantics
  }
  return Load(d, buf);
}


}  // namespace detail

// ============================================================================
// ComputeDivisorParams: Precompute multiplication factors for division
// ============================================================================

// For unsigned 8-bit
template <typename T, HWY_IF_T_SIZE(T, 1), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params;
  params.divisor = divisor;
  
  // if (divisor == 0) {
  //   params.multiplier = 0;
  //   params.shift1 = params.shift2 = 0;
  //   params.is_pow2 = false;
  //   return params;
  // }
  
  if (detail::IsPow2(divisor)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(divisor);
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (divisor == 1) {
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  unsigned l = 32 - detail::LeadingZeroCount32(divisor - 1);
  uint16_t two_l = static_cast<uint16_t>(1U << l);
  if (l == 8) two_l = 0;
  
  uint32_t m = ((static_cast<uint32_t>(two_l - divisor) << 8) / divisor) + 1;
  params.multiplier = static_cast<uint16_t>(m);
  params.shift1 = 1;
  params.shift2 = static_cast<int>(l - 1);
  
  return params;
}

// For unsigned 16-bit
template <typename T, HWY_IF_T_SIZE(T, 2), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params;
  params.divisor = divisor;
  
  // if (divisor == 0) {
  //   params.multiplier = 0;
  //   params.shift1 = params.shift2 = 0;
  //   params.is_pow2 = false;
  //   return params;
  // }
  
  if (detail::IsPow2(divisor)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(divisor);
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (divisor == 1) {
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  unsigned l = 32 - detail::LeadingZeroCount32(divisor - 1);
  uint32_t two_l = 1U << l;
  if (l == 16) two_l = 0;
  
  const uint64_t tmp = ((static_cast<uint64_t>(two_l - divisor) << 16) / divisor) + 1;
  const uint32_t m = static_cast<uint32_t>(tmp);
  params.multiplier = m;
  params.shift1 = 1;
  params.shift2 = static_cast<int>(l - 1);
  
  return params;
}

// For unsigned 32-bit
template <typename T, HWY_IF_T_SIZE(T, 4), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params;
  params.divisor = divisor;
  
  // if (divisor == 0) {
  //   params.multiplier = 0;
  //   params.shift1 = params.shift2 = 0;
  //   params.is_pow2 = false;
  //   return params;
  // }
  
  if (detail::IsPow2(divisor)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(divisor);
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (divisor == 1) {
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  unsigned l = 32 - detail::LeadingZeroCount32(divisor - 1);
  uint64_t two_l = 1ULL << l;
  if (l == 32) two_l = 0;
  
  uint64_t m = ((two_l - divisor) << 32) / divisor + 1;
  params.multiplier = static_cast<T>(m);
  params.shift1 = 1;
  params.shift2 = static_cast<int>(l - 1);
  
  return params;
}

// For unsigned 64-bit
template <typename T, HWY_IF_T_SIZE(T, 8), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  DivisorParamsU<T> params;
  params.divisor = divisor;
  
  // if (divisor == 0) {
  //   params.multiplier = 0;
  //   params.shift1 = params.shift2 = 0;
  //   params.is_pow2 = false;
  //   return params;
  // }
  
  if (detail::IsPow2(divisor)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros64(divisor);
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (divisor == 1) {
    params.multiplier = 1;
    params.shift1 = params.shift2 = 0;
    return params;
  }
  
  unsigned l = 64 - detail::LeadingZeroCount64(divisor - 1);
  uint64_t two_l_minus_d = (l < 64) ? ((1ULL << l) - divisor) : (0 - divisor);
  uint64_t m = detail::DivideHighBy(two_l_minus_d, divisor) + 1;
  
  params.multiplier = m;
  params.shift1 = 1;
  params.shift2 = static_cast<int>(l - 1);
  
  return params;
}

// For signed 8-bit
template <typename T, HWY_IF_T_SIZE(T, 1), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params;
  params.divisor = divisor;  // Store original for floor division
  
  // if (divisor == 0) {
  //   params.multiplier = 0;
  //   params.shift = 0;
  //   params.is_pow2 = false;
  //   return params;
  // }
  
  UT abs_d = static_cast<UT>(divisor < 0 ? -divisor : divisor);
  
  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (abs_d == 1) {
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  if (static_cast<UT>(divisor) == 0x80U) {
    params.multiplier = static_cast<int16_t>(0x81);
    params.shift = 6;
    return params;
  }
  
  unsigned sh = 31 - detail::LeadingZeroCount32(static_cast<uint32_t>(abs_d - 1));
  uint32_t m = (256U << sh) / abs_d + 1;
  params.multiplier = static_cast<int16_t>(m);
  params.shift = static_cast<int>(sh);
  
  return params;
}

// For signed 16-bit
template <typename T, HWY_IF_T_SIZE(T, 2), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params;
  params.divisor = divisor;
  
  // if (divisor == 0) {
  //   params.multiplier = 0;
  //   params.shift = 0;
  //   params.is_pow2 = false;
  //   return params;
  // }
  
  UT abs_d = static_cast<UT>(divisor < 0 ? -divisor : divisor);
  
  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (abs_d == 1) {
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  if (static_cast<UT>(divisor) == 0x8000U) {
    params.multiplier = static_cast<int32_t>(0x8001);
    params.shift = 14;
    return params;
  }
  
  unsigned sh = 31 - detail::LeadingZeroCount32(static_cast<uint32_t>(abs_d - 1));
  const uint64_t tmp = ((uint64_t{1} << 16) << sh) / abs_d + 1;
  const uint32_t m = static_cast<uint32_t>(tmp);
  params.multiplier = static_cast<int32_t>(m);
  params.shift = static_cast<int>(sh);
  
  return params;
}

// For signed 32-bit
template <typename T, HWY_IF_T_SIZE(T, 4), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params;
  params.divisor = divisor;
  
  // if (divisor == 0) {
  //   params.multiplier = 0;
  //   params.shift = 0;
  //   params.is_pow2 = false;
  //   return params;
  // }
  
  UT abs_d = static_cast<UT>(divisor < 0 ? -divisor : divisor);
  
  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros32(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (abs_d == 1) {
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  if (static_cast<UT>(divisor) == 0x80000000U) {
    params.multiplier = static_cast<T>(0x80000001);
    params.shift = 30;
    return params;
  }
  
  unsigned sh = 31 - detail::LeadingZeroCount32(abs_d - 1);
  uint64_t m = (0x100000000ULL << sh) / abs_d + 1;
  params.multiplier = static_cast<T>(m);
  params.shift = static_cast<int>(sh);
  
  return params;
}

// For signed 64-bit
template <typename T, HWY_IF_T_SIZE(T, 8), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) {
    HWY_ABORT("intdiv: division by zero in ComputeDivisorParams");
  }
  using UT = MakeUnsigned<T>;
  DivisorParamsS<T> params;
  params.divisor = divisor;
  
  // if (divisor == 0) {
  //   params.multiplier = 0;
  //   params.shift = 0;
  //   params.is_pow2 = false;
  //   return params;
  // }
  
  UT abs_d = static_cast<UT>(divisor < 0 ? -divisor : divisor);
  
  if (detail::IsPow2(abs_d)) {
    params.is_pow2 = true;
    params.pow2_shift = detail::CountTrailingZeros64(abs_d);
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  params.is_pow2 = false;
  
  if (abs_d == 1) {
    params.multiplier = 1;
    params.shift = 0;
    return params;
  }
  
  if (static_cast<UT>(divisor) == 0x8000000000000000ULL) {
    params.multiplier = static_cast<T>(0x8000000000000001LL);
    params.shift = 62;
    return params;
  }
  
  unsigned sh = 63 - detail::LeadingZeroCount64(abs_d - 1);
  uint64_t m = detail::DivideHighBy(1ULL << sh, abs_d) + 1;
  params.multiplier = static_cast<T>(m);
  params.shift = static_cast<int>(sh);
  
  return params;
}

// ============================================================================
// IntDiv: Perform integer division using precomputed parameters (TRUNC)
// ============================================================================

// Division for unsigned types
template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V IntDiv(D d, V dividend, const DivisorParamsU<T>& params) {
  HWY_DASSERT(params.divisor != 0);
  // Fast path: power of 2
  if (params.is_pow2) {
    return detail::ShiftRightUniform(d, dividend, params.pow2_shift);
  }
  
  // Handle division by 1
  if (params.shift1 == 0 && params.shift2 == 0 && params.multiplier == 1) {
    return dividend;
  }
  
  // ========== ARM/Power (and optionally others) 64-bit: scalar fallback ==========
  #if HWY_INTDIV_SCALAR64
    if constexpr (sizeof(T) == 8) {
      return detail::ScalarDivPerLane(d, dividend, params.divisor);  // truncation semantics
    }
  #endif

// Special handling for 8-bit and 16-bit (need wider multiplier)
if constexpr (sizeof(T) == 1) {
  // For 1-lane partial vectors, avoid PromoteUpperTo entirely.
    if constexpr (D::kPrivateLanes < 2) {
      return detail::ScalarDivPerLane(d, dividend, params.divisor);
    } else {
      // For uint8_t -> uint16_t promotion
      // Need descriptor with uint16_t type and half the lanes
      using TWide = MulType_t<T>;  // uint16_t
      const Repartition<TWide, D> d_wide;  // Same byte size, half the lanes
      
      // Promote lower and upper halves directly
      const auto lo_wide = PromoteLowerTo(d_wide, dividend);
      const auto hi_wide = PromoteUpperTo(d_wide, dividend);
      
      const auto mul_wide = Set(d_wide, static_cast<TWide>(params.multiplier));
      
      const auto prod_lo = Mul(lo_wide, mul_wide);
      const auto prod_hi = Mul(hi_wide, mul_wide);
      
      // Extract high bytes (shift right by 8) and demote back
      #if defined(HWY_HAVE_ORDEREDDEMOTE2TO)
        const V t1 = OrderedDemote2To(d, ShiftRight<8>(prod_lo), ShiftRight<8>(prod_hi));
      #else
        const Half<D> d_half;
        const auto t1_lo = DemoteTo(d_half, ShiftRight<8>(prod_lo));
        const auto t1_hi = DemoteTo(d_half, ShiftRight<8>(prod_hi));
        const V t1 = Combine(d, t1_hi, t1_lo);
      #endif
      
      const V diff = Sub(dividend, t1);
      const V shifted = detail::ShiftRightUniform(d, diff, params.shift1);
      const V sum = Add(t1, shifted);
      return detail::ShiftRightUniform(d, sum, params.shift2);
    }
  
} else if constexpr (sizeof(T) == 2) {
  if constexpr (D::kPrivateLanes < 2) {
      return detail::ScalarDivPerLane(d, dividend, params.divisor);
    } else {
      // For uint16_t -> uint32_t promotion
      using TWide = MulType_t<T>;  // uint32_t
      const Repartition<TWide, D> d_wide;
      
      const auto lo_wide = PromoteLowerTo(d_wide, dividend);
      const auto hi_wide = PromoteUpperTo(d_wide, dividend);
      
      const auto mul_wide = Set(d_wide, static_cast<TWide>(params.multiplier));
      
      const auto prod_lo = Mul(lo_wide, mul_wide);
      const auto prod_hi = Mul(hi_wide, mul_wide);
      
      #if defined(HWY_HAVE_ORDEREDDEMOTE2TO)
        const V t1 = OrderedDemote2To(d, ShiftRight<16>(prod_lo), ShiftRight<16>(prod_hi));
      #else
        const Half<D> d_half;
        const auto t1_lo = DemoteTo(d_half, ShiftRight<16>(prod_lo));
        const auto t1_hi = DemoteTo(d_half, ShiftRight<16>(prod_hi));
        const V t1 = Combine(d, t1_hi, t1_lo);
      #endif
     
     const V diff = Sub(dividend, t1);
     const V shifted = detail::ShiftRightUniform(d, diff, params.shift1);
     const V sum = Add(t1, shifted);
     return detail::ShiftRightUniform(d, sum, params.shift2);
    }
} else if constexpr (sizeof(T) == 4) {
    // 32-bit uses MulHigh
    const V multiplier = Set(d, params.multiplier);
    const V t1 = MulHigh(dividend, multiplier);
    const V diff = Sub(dividend, t1);
    const V shifted = detail::ShiftRightUniform(d, diff, params.shift1);
    const V sum = Add(t1, shifted);
    return detail::ShiftRightUniform(d, sum, params.shift2);
    
  } else {
    // u64: MulHigh may be emulated on many targets (AVX2/NEON/VSX).
    // If HWY_INTDIV_SCALAR64=1, we returned earlier via scalar fallback.
    // 64-bit: use MulHigh even if emulated - correctness over speed
    // Highway will emulate MulHigh where needed, which may be slower
    // but gives correct results. Users can add target-specific optimizations.
    const V multiplier = Set(d, params.multiplier);
    const V t1 = MulHigh(dividend, multiplier);
    const V diff = Sub(dividend, t1);
    const V shifted = detail::ShiftRightUniform(d, diff, params.shift1);
    const V sum = Add(t1, shifted);
    return detail::ShiftRightUniform(d, sum, params.shift2);
  }
}

// Division for signed types (truncating division)
template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V IntDiv(D d, V dividend, const DivisorParamsS<T>& params) {
  const bool neg_divisor = params.divisor < 0;
  HWY_DASSERT(params.divisor != 0);
  
  // Fast path: power of 2 (truncation toward zero using bias trick)
  if (params.is_pow2) {
    // For signed truncating division by ±2^k, we cannot use abs()
    // because abs(INT_MIN) overflows. Instead, use the bias trick:
    // q = (dividend + bias) >> k, where bias = (dividend < 0) ? (2^k - 1) : 0
    // This ensures truncation toward zero without overflow
    
    using UT = MakeUnsigned<T>;
    
    // Compute (2^k - 1) safely in unsigned domain to avoid UB
    HWY_DASSERT(params.pow2_shift >= 0 && params.pow2_shift < static_cast<int>(8 * sizeof(T)));
    // shift count as unsigned avoids UB when T is signed
    const T mask_val = static_cast<T>((static_cast<UT>(1) << static_cast<unsigned>(params.pow2_shift)) - 1);

    const V mask = Set(d, mask_val);
    
    // Get sign mask: all ones if negative, all zeros if positive
    constexpr int kSignBit = int(sizeof(T) * 8) - 1;
    const V sign = ShiftRight<kSignBit>(dividend);  // Arithmetic shift
    
    // bias = (dividend < 0) ? (2^k - 1) : 0
    const V bias = And(sign, mask);
    
    // Arithmetic shift with bias for truncation toward zero
    V q = detail::ShiftRightUniform(d, Add(dividend, bias), params.pow2_shift);
    
    // Negate result if divisor is negative
    if (neg_divisor) {
      q = Neg(q);
    }
    
    return q;
  }
  
  // Handle division by ±1
  if (params.shift == 0 && params.multiplier == 1) {
    if (neg_divisor) {
      return Neg(dividend);
    }
    return dividend;
  }

  // ========== ARM/Power (and optionally others) 64-bit: scalar fallback ==========
  #if HWY_INTDIV_SCALAR64
    if constexpr (sizeof(T) == 8) {
      return detail::ScalarDivPerLane(d, dividend, params.divisor);  // truncation semantics
    }
  #endif

  V q0;
  
if constexpr (sizeof(T) == 1) {
  if constexpr (D::kPrivateLanes < 2) {
    q0 = detail::ScalarDivPerLane(d, dividend, params.divisor);
    } else {
      using TWide = MulType_t<T>;  // int16_t
      const Repartition<TWide, D> d_wide;
      
      const auto lo_wide = PromoteLowerTo(d_wide, dividend);
      const auto hi_wide = PromoteUpperTo(d_wide, dividend);
      
      const auto mul_wide = Set(d_wide, static_cast<TWide>(params.multiplier));
      
      const auto prod_lo = Mul(lo_wide, mul_wide);
      const auto prod_hi = Mul(hi_wide, mul_wide);
      
      #if defined(HWY_HAVE_ORDEREDDEMOTE2TO)
        const auto high = OrderedDemote2To(d, ShiftRight<8>(prod_lo), ShiftRight<8>(prod_hi));
      #else
        const Half<D> d_half;
        const auto high_lo = DemoteTo(d_half, ShiftRight<8>(prod_lo));
        const auto high_hi = DemoteTo(d_half, ShiftRight<8>(prod_hi));
        const auto high = Combine(d, high_hi, high_lo);
      #endif
      
      q0 = Add(dividend, high);
    }
  
} else if constexpr (sizeof(T) == 2) {
  if constexpr (D::kPrivateLanes < 2) {
    q0 = detail::ScalarDivPerLane(d, dividend, params.divisor);
    } else {
      using TWide = MulType_t<T>;  // int32_t
      const Repartition<TWide, D> d_wide;
      
      const auto lo_wide = PromoteLowerTo(d_wide, dividend);
      const auto hi_wide = PromoteUpperTo(d_wide, dividend);
      
      const auto mul_wide = Set(d_wide, static_cast<TWide>(params.multiplier));
      
      const auto prod_lo = Mul(lo_wide, mul_wide);
      const auto prod_hi = Mul(hi_wide, mul_wide);
      
      #if defined(HWY_HAVE_ORDEREDDEMOTE2TO)
        const auto high = OrderedDemote2To(d, ShiftRight<16>(prod_lo), ShiftRight<16>(prod_hi));
      #else
        const Half<D> d_half;
        const auto high_lo = DemoteTo(d_half, ShiftRight<16>(prod_lo));
        const auto high_hi = DemoteTo(d_half, ShiftRight<16>(prod_hi));
        const auto high = Combine(d, high_hi, high_lo);
      #endif
      
      q0 = Add(dividend, high);
    }
} else if constexpr (sizeof(T) == 4) {
    const V multiplier = Set(d, params.multiplier);
    const V mulh = MulHigh(dividend, multiplier);
    q0 = Add(dividend, mulh);
    
  } else {
    // 64-bit: use MulHigh even if emulated - correctness over speed
    const V multiplier = Set(d, params.multiplier);
    const V mulh = MulHigh(dividend, multiplier);
    q0 = Add(dividend, mulh);
  }
  
  // Arithmetic shift right
  // q0 = a + MULSH(m, a); then arithmetic shift
  q0 = detail::ShiftRightUniform(d, q0, params.shift);
  
  // Subtract sign of dividend (no BitCast needed, ShiftRight is arithmetic for signed)
  // Subtract sign(a): q0 -= sign(a)
  constexpr int kSignBit2 = int(sizeof(T) * 8) - 1;
  const V sign_dividend = ShiftRight<kSignBit2>(dividend);
  q0 = Sub(q0, sign_dividend);
  
  // Apply sign of divisor
  // Apply sign of divisor: q = EOR(q0, dsign) - dsign
  if (neg_divisor) {
    const V neg_one = Set(d, static_cast<T>(-1));
    q0 = Xor(q0, neg_one);
    q0 = Sub(q0, neg_one);
  }
  
  return q0;
}

// ============================================================================
// IntDivFloor: Floor division (Python/NumPy semantics)
// ============================================================================

// Floor division for signed types - always correct
template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V IntDivFloor(D d, V dividend, const DivisorParamsS<T>& params) {
  // Get truncating division result
  V q = IntDiv(d, dividend, params);
  
  // Floor correction: q - ((a != q*d) & (sign(a) != sign(d)))
  const V divisor = Set(d, params.divisor);
  const V prod = Mul(q, divisor);
  const auto neq = Ne(dividend, prod);
  const auto sdiff = Xor(Lt(dividend, Zero(d)), Lt(divisor, Zero(d)));
  const V one = Set(d, static_cast<T>(1));
  
  // Use IfThenElse instead of BitCast for portability
  return Sub(q, IfThenElse(And(neq, sdiff), one, Zero(d)));
}

// For unsigned types, floor division is the same as regular division
template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V IntDivFloor(D d, V dividend, const DivisorParamsU<T>& params) {
  return IntDiv(d, dividend, params);
}

// ============================================================================
// Convenience functions
// ============================================================================

// Divide vector by scalar with precomputation (truncating)
template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V DivideByScalar(D d, V dividend, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");
  
  // Fast path for power of 2 before params computation
  if (detail::IsPow2(divisor)) {
    const int ctz = (sizeof(T) == 8)
        ? detail::CountTrailingZeros64(static_cast<uint64_t>(divisor))
        : detail::CountTrailingZeros32(static_cast<uint32_t>(divisor));
    return detail::ShiftRightUniform(d, dividend, ctz);
  }
  
  const auto params = ComputeDivisorParams(divisor);
  return IntDiv(d, dividend, params);
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V DivideByScalar(D d, V dividend, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");
  
  const auto params = ComputeDivisorParams(divisor);
  return IntDiv(d, dividend, params);
}

// Floor divide vector by scalar
template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V FloorDivideByScalar(D d, V dividend, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");
  
  const auto params = ComputeDivisorParams(divisor);
  return IntDivFloor(d, dividend, params);
}

template <class D, class V = Vec<D>, typename T = TFromD<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V FloorDivideByScalar(D d, V dividend, T divisor) {
  return DivideByScalar(d, dividend, divisor);  // Same for unsigned
}

// Broadcast division: divide array by scalar (truncating)
template <typename T, HWY_IF_UNSIGNED(T)>
HWY_INLINE void DivideArrayByScalar(T* HWY_RESTRICT array, size_t count, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");
  
  const ScalableTag<T> d;
  const size_t N = Lanes(d);
  const auto params = ComputeDivisorParams(divisor);
  
  size_t i = 0;
  for (; i + N <= count; i += N) {
    const auto vec = LoadU(d, array + i);
    const auto result = IntDiv(d, vec, params);
    StoreU(result, d, array + i);
  }
  
  // Handle remainder
  if (i < count) {
    const size_t remaining = count - i;
    const auto vec = LoadN(d, array + i, remaining);
    const auto result = IntDiv(d, vec, params);
    StoreN(result, d, array + i, remaining);
  }
}

template <typename T, HWY_IF_SIGNED(T)>
HWY_INLINE void DivideArrayByScalar(T* HWY_RESTRICT array, size_t count, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");
  
  const ScalableTag<T> d;
  const size_t N = Lanes(d);
  const auto params = ComputeDivisorParams(divisor);
  
  size_t i = 0;
  for (; i + N <= count; i += N) {
    const auto vec = LoadU(d, array + i);
    const auto result = IntDiv(d, vec, params);
    StoreU(result, d, array + i);
  }
  
  // Handle remainder
  if (i < count) {
    const size_t remaining = count - i;
    const auto vec = LoadN(d, array + i, remaining);
    const auto result = IntDiv(d, vec, params);
    StoreN(result, d, array + i, remaining);
  }
}

// Floor divide array by scalar
template <typename T, HWY_IF_SIGNED(T)>
HWY_INLINE void FloorDivideArrayByScalar(T* HWY_RESTRICT array, size_t count, T divisor) {
  HWY_DASSERT(divisor != 0);
  if (HWY_UNLIKELY(divisor == 0)) HWY_ABORT("intdiv: division by zero");
  
  const ScalableTag<T> d;
  const size_t N = Lanes(d);
  const auto params = ComputeDivisorParams(divisor);
  
  size_t i = 0;
  for (; i + N <= count; i += N) {
    const auto vec = LoadU(d, array + i);
    const auto result = IntDivFloor(d, vec, params);
    StoreU(result, d, array + i);
  }
  
  // Handle remainder
  if (i < count) {
    const size_t remaining = count - i;
    const auto vec = LoadN(d, array + i, remaining);
    const auto result = IntDivFloor(d, vec, params);
    StoreN(result, d, array + i, remaining);
  }
}

template <typename T, HWY_IF_UNSIGNED(T)>
HWY_INLINE void FloorDivideArrayByScalar(T* HWY_RESTRICT array, size_t count, T divisor) {
  DivideArrayByScalar(array, count, divisor);  // Same for unsigned
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // per-target include guard