// Copyright 2024 Google LLC
// Copyright 2025 Fujitsu India Pvt Ltd.
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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/intdiv/intdiv_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

#include "hwy/contrib/intdiv/intdiv-inl.h"   // internal helper (detail::DivideHighBy)
#include "hwy/contrib/intdiv/intdiv.h"       // public API under test
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

using hwy::HWY_NAMESPACE::ComputeDivisorParams;
using hwy::HWY_NAMESPACE::DivideArrayByScalar;
using hwy::HWY_NAMESPACE::DivideByScalar;
using hwy::HWY_NAMESPACE::FloorDivideArrayByScalar;
using hwy::HWY_NAMESPACE::FloorDivideByScalar;
using hwy::HWY_NAMESPACE::IntDiv;
using hwy::HWY_NAMESPACE::IntDivFloor;

// ============================================================================
// Utilities (Random, Safe floor reference, pow2 check)  [UTILS]
// ============================================================================
template <typename T>
static T RandWithin(hwy::RandomState& rng) {
  if constexpr (sizeof(T) <= 4) {
    return static_cast<T>(Random32(&rng));
  } else {
    const uint64_t hi = Random32(&rng);
    const uint64_t lo = Random32(&rng);
    return static_cast<T>((hi << 32) ^ lo);
  }
}

template <typename T>
static T SafeFloorDivScalar(T a, T b) {
  if constexpr (!hwy::IsSigned<T>()) {
    return static_cast<T>(a / b);
  } else {
    if (b == T(-1) && a == std::numeric_limits<T>::min()) {
      // Avoid scalar UB; caller skips verifying this lane.
      return T();
    }
    const T q = static_cast<T>(a / b);  // trunc toward zero
    const bool adjust = (a != q * b) && ((a < 0) != (b < 0));
    return static_cast<T>(q - (adjust ? 1 : 0));
  }
}

template <typename T>
static bool IsPow2(T x) {
  using U = typename hwy::MakeUnsigned<T>;
  const U ux = static_cast<U>(x);
  return ux != 0 && (ux & (ux - 1)) == 0;
}

// ============================================================================
// Unsigned Integer Division Tests  [UNSIGNED]
// ============================================================================
template <typename T>
class TestIntDivUnsigned {
  template <class D>
  static HWY_NOINLINE void TestDivisor(D d, T divisor) {
    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);
    HWY_ASSERT(lanes && expected && actual);

    const auto params = ComputeDivisorParams(divisor);

    // Param sanity quick checks
    if (IsPow2(divisor)) {
      HWY_ASSERT(params.is_pow2);
    }
    if (divisor == T(1)) {
      HWY_ASSERT_EQ(1, params.multiplier);
      HWY_ASSERT_EQ(0, params.shift1);
      HWY_ASSERT_EQ(0, params.shift2);
    }
    HWY_ASSERT_EQ(divisor, params.divisor);

    const T tv[] = {T(0),
                    T(1),
                    T(2),
                    T(3),
                    T(divisor - 1),
                    divisor,
                    T(divisor + 1),
                    T(divisor * 2),
                    T(divisor * 3),
                    T(7),
                    T(10),
                    T(100),
                    T(1000),
                    T(12345),
                    T(std::numeric_limits<T>::max() / 2),
                    T(std::numeric_limits<T>::max() - 1),
                    T(std::numeric_limits<T>::max())};

    for (T base : tv) {
      for (size_t i = 0; i < N; ++i) {
        const T v = static_cast<T>(base + static_cast<T>(i));
        lanes[i] = v;
        expected[i] = static_cast<T>(v / divisor);
      }
      const auto vec = Load(d, lanes.get());
      const auto got = IntDiv(d, vec, params);
      Store(got, d, actual.get());
      for (size_t i = 0; i < N; ++i) {
        HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }

 public:
  template <class T2, class D>
  HWY_NOINLINE void operator()(T2 /*unused*/, D d) {
    // Baseline set incl. powers-of-two
    for (T divisor : {T(1), T(2), T(3), T(5), T(7), T(10),
                      T(16), T(17), T(25), T(32), T(64), T(100), T(127), T(128), T(255), T(256), T(1000)}) {
      TestDivisor(d, divisor);
    }
    // Width-specific boundaries
    if constexpr (sizeof(T) == 4) {
      for (T divisor : {T(65535), T(65536), T(0x7FFFFFFF)}) TestDivisor(d, divisor);
    } else if constexpr (sizeof(T) == 8) {
      for (T divisor : {T(0xFFFFFFFFull), T(0x100000000ull)}) TestDivisor(d, divisor);
    }

    // Random values (deterministic pattern)
    {
      for (T divisor : {T(3), T(7), T(17), T(100), T(1000)}) {
        const auto params = ComputeDivisorParams(divisor);
        for (int i = 0; i < 100; ++i) {
          const T dividend = static_cast<T>(i * T(123456789)) % std::numeric_limits<T>::max();
          const auto v = Set(d, dividend);
          const auto q = IntDiv(d, v, params);
          HWY_ASSERT_EQ(static_cast<T>(dividend / divisor), GetLane(q));
        }
      }
    }
  }
};

// Array helper for unsigned
struct TestUnsignedDivideArrayByScalar {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D /*d*/) {
    std::vector<T> array = {T(0), T(1), T(7), T(100), T(1000), T(std::numeric_limits<T>::max() / 2)};
    std::vector<T> expected = array;
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<T>(expected[i] / T(7));
    DivideArrayByScalar(array.data(), array.size(), T(7));
    for (size_t i = 0; i < array.size(); ++i) HWY_ASSERT_EQ(expected[i], array[i]);
  }
};

// ============================================================================
// Signed Integer Division Tests (Trunc)  [SIGNED_TRUNC]
// ============================================================================
template <typename T>
class TestIntDivSigned {
  template <class D>
  static HWY_NOINLINE void TestDivisor(D d, T divisor) {
    const size_t N = Lanes(d);
    auto lanes = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);
    HWY_ASSERT(lanes && expected && actual);

    const auto params = ComputeDivisorParams(divisor);

    const T tv[] = {T(0),
                    T(1),
                    T(-1),
                    T(2),
                    T(-2),
                    T(divisor),
                    T(-divisor),
                    T(divisor - 1),
                    T(divisor + 1),
                    T(-divisor - 1),
                    T(-divisor + 1),
                    T(100),
                    T(-100),
                    T(1234),
                    T(-1234),
                    T(std::numeric_limits<T>::max() / 2),
                    T(std::numeric_limits<T>::min() / 2),
                    T(std::numeric_limits<T>::max()),
                    T(std::numeric_limits<T>::min() + 1)};

    for (T base : tv) {
      for (size_t i = 0; i < N; ++i) {
        const T v = static_cast<T>(base + static_cast<T>(i));
        lanes[i] = v;
        if (divisor == T(-1) && v == std::numeric_limits<T>::min()) {
          // Scalar reference would overflow; keep lane for exercising path.
          expected[i] = T();
        } else {
          expected[i] = static_cast<T>(v / divisor);  // trunc
        }
      }
      const auto vec = Load(d, lanes.get());
      const auto got = IntDiv(d, vec, params);
      Store(got, d, actual.get());
      for (size_t i = 0; i < N; ++i) {
        if (divisor == T(-1) && lanes[i] == std::numeric_limits<T>::min()) continue;
        HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }

 public:
  template <class T2, class D>
  HWY_NOINLINE void operator()(T2 /*unused*/, D d) {
    // Basic positives/negatives
    for (T divisor : {T(3), T(5), T(7), T(-3), T(-5), T(-7)}) TestDivisor(d, divisor);

    // Truncation semantics spot checks
    {
      auto p3 = ComputeDivisorParams(T(3));
      auto pm3 = ComputeDivisorParams(T(-3));
      HWY_ASSERT_EQ(T(-2), GetLane(IntDiv(d, Set(d, T(-7)), p3)));
      HWY_ASSERT_EQ(T(-2), GetLane(IntDiv(d, Set(d, T(7)), pm3)));
      HWY_ASSERT_EQ(T(2), GetLane(IntDiv(d, Set(d, T(-7)), pm3)));
    }

    // Power-of-two divisors (signed)
    for (int shift = 0; shift < static_cast<int>(sizeof(T) * 8) - 1; ++shift) {
      const T divisor = static_cast<T>(T(1) << shift);
      const auto params = ComputeDivisorParams(divisor);
      HWY_ASSERT(params.is_pow2);
      HWY_ASSERT_EQ(shift, params.pow2_shift);
      for (T dividend : {T(-100), T(-1), T(0), T(1), T(100)}) {
        HWY_ASSERT_EQ(static_cast<T>(dividend / divisor), GetLane(IntDiv(d, Set(d, dividend), params)));
      }
    }

    // +/-1 divisors
    {
      const auto p1 = ComputeDivisorParams(T(1));
      const auto m1 = ComputeDivisorParams(T(-1));
      for (T a : {T(-100), T(-1), T(0), T(1), T(100)}) {
        HWY_ASSERT_EQ(a, GetLane(IntDiv(d, Set(d, a), p1)));
        if (a == std::numeric_limits<T>::min()) continue;  // scalar UB reference
        HWY_ASSERT_EQ(static_cast<T>(-a), GetLane(IntDiv(d, Set(d, a), m1)));
      }
    }

    // Random values (deterministic pattern)
    {
      for (T divisor : {T(3), T(7), T(-3), T(-7), T(17), T(-17)}) {
        const auto params = ComputeDivisorParams(divisor);
        for (int i = 0; i < 50; ++i) {
          const T dividend = static_cast<T>(i * T(123456789));
          HWY_ASSERT_EQ(static_cast<T>(dividend / divisor), GetLane(IntDiv(d, Set(d, dividend), params)));
        }
      }
    }
  }
};

// Signed array helper
struct TestSignedDivideArrayByScalar {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D /*d*/) {
    std::vector<T> array = {T(-100), T(-7), T(-1), T(0), T(1), T(7), T(100)};
    std::vector<T> expected = array;
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<T>(expected[i] / T(7));
    DivideArrayByScalar(array.data(), array.size(), T(7));
    for (size_t i = 0; i < array.size(); ++i) HWY_ASSERT_EQ(expected[i], array[i]);
  }
};

// ============================================================================
// Signed Floor Division Tests (Python/NumPy)  [SIGNED_FLOOR]
// ============================================================================
template <typename T>
class TestSignedFloorDivision {
 public:
  template <class T2, class D>
  HWY_NOINLINE void operator()(T2 /*unused*/, D d) {
    // Basic floor division
    for (T divisor : {T(3), T(5), T(-3), T(-5)}) {
      const auto params = ComputeDivisorParams(divisor);
      for (T dividend : {T(-100), T(-7), T(-1), T(0), T(1), T(7), T(100)}) {
        const T actual = GetLane(IntDivFloor(d, Set(d, dividend), params));
        const T expect = SafeFloorDivScalar(dividend, divisor);
        HWY_ASSERT_EQ(expect, actual);
      }
    }

    // Floor vs truncation difference
    {
      const auto p3 = ComputeDivisorParams(T(3));
      const T trunc_neg = GetLane(IntDiv(d, Set(d, T(-7)), p3));
      const T floor_neg = GetLane(IntDivFloor(d, Set(d, T(-7)), p3));
      HWY_ASSERT_EQ(T(-2), trunc_neg);
      HWY_ASSERT_EQ(T(-3), floor_neg);

      const T trunc_pos = GetLane(IntDiv(d, Set(d, T(7)), p3));
      const T floor_pos = GetLane(IntDivFloor(d, Set(d, T(7)), p3));
      HWY_ASSERT_EQ(T(2), trunc_pos);
      HWY_ASSERT_EQ(T(2), floor_pos);
    }

    // Floor divide array
    {
      std::vector<T> array = {T(-100), T(-7), T(-1), T(0), T(1), T(7), T(100)};
      std::vector<T> expected = array;
      for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = SafeFloorDivScalar(expected[i], T(3));
      }
      FloorDivideArrayByScalar(array.data(), array.size(), T(3));
      for (size_t i = 0; i < array.size(); ++i) HWY_ASSERT_EQ(expected[i], array[i]);
    }
  }
};

// ============================================================================
// Edge & Special  [EDGE_SPECIAL]
// ============================================================================
struct TestIntDivEdgeCases {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto dividend = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);
    HWY_ASSERT(dividend && actual);

    // Division by 1 (identity)
    {
      const auto params = ComputeDivisorParams(T(1));
      for (size_t i = 0; i < N; ++i) dividend[i] = static_cast<T>(i + 1);
      const auto v = Load(d, dividend.get());
      const auto q = IntDiv(d, v, params);
      Store(q, d, actual.get());
      for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(dividend[i], actual[i]);
    }

    // Division by a power of two (mid shift)
    if (sizeof(T) >= 2) {
      const int k = int(sizeof(T) * 4);
      const T pow2 = static_cast<T>(T(1) << k);
      const auto params = ComputeDivisorParams(pow2);
      for (size_t i = 0; i < N; ++i) dividend[i] = static_cast<T>(pow2 * T(i + 1));
      const auto v = Load(d, dividend.get());
      const auto q = IntDiv(d, v, params);
      Store(q, d, actual.get());
      for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(static_cast<T>(i + 1), actual[i]);
    }

    // Directed checks: a<div -> 0 ; a==div -> 1
    {
      const T divisor = T(7);
      const auto params = ComputeDivisorParams(divisor);
      HWY_ASSERT_EQ(T(0), GetLane(IntDiv(d, Set(d, T(3)), params)));
      HWY_ASSERT_EQ(T(1), GetLane(IntDiv(d, Set(d, divisor), params)));
    }

    // MaxUnsigned (only for unsigned T)
    if constexpr (!hwy::IsSigned<T>()) {
      const auto p = ComputeDivisorParams(T(7));
      const T a = std::numeric_limits<T>::max();
      HWY_ASSERT_EQ(static_cast<T>(a / T(7)), GetLane(IntDiv(d, Set(d, a), p)));
    }

    // MaxSigned (only for signed T)
    if constexpr (hwy::IsSigned<T>()) {
      const auto p = ComputeDivisorParams(T(7));
      const T a = std::numeric_limits<T>::max();
      HWY_ASSERT_EQ(static_cast<T>(a / T(7)), GetLane(IntDiv(d, Set(d, a), p)));
    }

    // MinSignedPositiveDivisor (avoid exact INT_MIN/-1 scalar reference)
    if constexpr (hwy::IsSigned<T>()) {
      const auto p = ComputeDivisorParams(T(7));
      const T a = static_cast<T>(std::numeric_limits<T>::min() + 1);
      HWY_ASSERT_EQ(static_cast<T>(a / T(7)), GetLane(IntDiv(d, Set(d, a), p)));
    }
  }
};

// ============================================================================
// Convenience API  [CONVENIENCE]
// ============================================================================
struct TestDivideByScalarConvenience {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto dividend = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);
    HWY_ASSERT(dividend && expected && actual);

    const T divisor = T(7);  // prime

    for (size_t i = 0; i < N; ++i) {
      dividend[i] = static_cast<T>(i * 10);
      expected[i] = static_cast<T>(dividend[i] / divisor);  // trunc semantics
    }

    const auto v = Load(d, dividend.get());
    const auto q = DivideByScalar(d, v, divisor);
    Store(q, d, actual.get());
    for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(expected[i], actual[i]);

    if constexpr (hwy::IsSigned<T>()) {
      const auto fq = FloorDivideByScalar(d, v, divisor);
      T out[HWY_MAX_BYTES / sizeof(T)];
      Store(fq, d, out);
      for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(SafeFloorDivScalar(dividend[i], divisor), out[i]);
    }
  }
};

// ============================================================================
// Fuzz/Directed  [FUZZ_DIRECTED]
// ============================================================================
struct TestDirectedLoopSnippet {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    hwy::RandomState rng(777);
    const T divisors[] = {T(1), T(2), T(3), T(7), T(10), T(100),
                          T(std::numeric_limits<T>::max())};
    for (T divisor : divisors) {
      const auto params = ComputeDivisorParams(divisor);
      for (int i = 0; i < 64; ++i) {
        T dividend;
        switch (i) {
          case 0:
            dividend = T(0);
            break;
          case 1:
            dividend = T(1);
            break;
          case 2:
            dividend = hwy::IsSigned<T>() ? T(-1) : T(1);
            break;
          case 3:
            dividend = std::numeric_limits<T>::max();
            break;
          case 4:
            dividend = hwy::IsSigned<T>() ? std::numeric_limits<T>::min() : T(0);
            break;
          default:
            dividend = RandWithin<T>(rng);
            break;
        }
        const T got = GetLane(IntDiv(d, Set(d, dividend), params));
        if constexpr (hwy::IsSigned<T>()) {
          if (divisor == T(-1) && dividend == std::numeric_limits<T>::min()) continue;
        }
        const T expect = static_cast<T>(dividend / divisor);
        HWY_ASSERT_EQ(expect, got);
      }
    }
  }
};

// ============================================================================
// Internal Helper (DivideHighBy)  [INTERNAL_HELPER]
// ============================================================================
struct TestDivideHighBySanity {
  HWY_INLINE void operator()() const {
    using detail::DivideHighBy;
    // high=1, divisor=3 → 0x555...555
    {
      const uint64_t out = DivideHighBy(/*high=*/1ull, /*divisor=*/3ull);
      HWY_ASSERT_EQ(0x5555555555555555ull, out);
    }
    // high=2^63, divisor=2^63 → 0
    {
      const uint64_t high = 1ull << 63;
      const uint64_t div = 1ull << 63;
      const uint64_t out = DivideHighBy(high, div);
      HWY_ASSERT_EQ(0ull, out);
    }
    // high=1, divisor=2^64-1 → 1
    {
      const uint64_t out = DivideHighBy(/*high=*/1ull, /*divisor=*/~0ull);
      HWY_ASSERT_EQ(1ull, out);
    }
  }
};

// ============================================================================
// Optional Performance  [PERF_OPTIONAL]
// ============================================================================
struct TestOptionalPerf {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    std::vector<T> data(10000);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<T>(i) * T(123456789);

    const auto params = ComputeDivisorParams(T(7));
    const auto start = std::chrono::high_resolution_clock::now();
    size_t i = 0;
    for (; i + Lanes(d) <= data.size(); i += Lanes(d)) {
      const auto v = LoadU(d, data.data() + i);
      const auto q = IntDiv(d, v, params);
      StoreU(q, d, data.data() + i);
    }
    if (i < data.size()) {
      const size_t rem = data.size() - i;
      const auto v = LoadN(d, data.data() + i, rem);
      const auto q = IntDiv(d, v, params);
      StoreN(q, d, data.data() + i, rem);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "intdiv perf (T=" << sizeof(T) * 8 << "): " << us << " us\n";
  }
};

// ============================================================================
// === Added: Port of second test file blocks (now with correct template) ====
// ============================================================================

// Reference floor division (mirrors FloorDivScalar implementation)
template <typename T, HWY_IF_SIGNED(T)>
T AddedFloorDivScalar(T a, T b) {
  T q = a / b;  // trunc
  T r = a % b;
  if ((r != 0) && ((a ^ b) < 0)) q -= 1;
  return q;
}
template <typename T, HWY_IF_UNSIGNED(T)>
T AddedFloorDivScalar(T a, T b) {
  return a / b;
}

// [ADDED_BASIC]
template <typename T>
struct AddedBasicDivision {
  template <class T2, class D>
  HWY_NOINLINE void operator()(T2 /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto dividend = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);
    HWY_ASSERT(dividend && expected && actual);

    const T divisors[] = {1, 2, 3, 5, 7, 10, 11, 13, 16, 17, 31, 32, 100, 127};
    for (T divisor : divisors) {
      const auto params = ComputeDivisorParams(divisor);
      for (size_t base = 0; base < 256; base += 17) {
        for (size_t i = 0; i < N; ++i) {
          const T v = static_cast<T>(base + i);
          dividend[i] = v;
          expected[i] = static_cast<T>(v / divisor);
        }
        const auto vec = Load(d, dividend.get());
        const auto q = IntDiv(d, vec, params);
        Store(q, d, actual.get());
        for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }
};

// [ADDED_PO2]
template <typename T>
struct AddedPowerOf2Division {
  template <class T2, class D>
  HWY_NOINLINE void operator()(T2 /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto dividend = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);
    HWY_ASSERT(dividend && expected && actual);

    for (int shift = 0; shift < static_cast<int>(sizeof(T) * 8 - hwy::IsSigned<T>()); ++shift) {
      const T divisor = static_cast<T>(T(1) << shift);
      if (divisor <= 0) break;
      const auto params = ComputeDivisorParams(divisor);
      HWY_ASSERT(params.is_pow2);
      HWY_ASSERT_EQ(shift, params.pow2_shift);

      const T start = hwy::IsSigned<T>() ? static_cast<T>(-100) : T(0);
      const T end = static_cast<T>(100);
      for (T base = start; base < end; base += 7) {
        for (size_t i = 0; i < N; ++i) {
          dividend[i] = static_cast<T>(base + i);
          expected[i] = static_cast<T>(dividend[i] / divisor);
        }
        const auto vec = Load(d, dividend.get());
        const auto q = IntDiv(d, vec, params);
        Store(q, d, actual.get());
        for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }
};

// [ADDED_SIGNED_EDGE]
template <typename T>
struct AddedSignedEdgeCases {
  template <class T2, class D>
  HWY_NOINLINE void operator()(T2 /*unused*/, D d) {
    if (!hwy::IsSigned<T>()) return;
    const size_t N = Lanes(d);
    auto dividend = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);
    HWY_ASSERT(dividend && expected && actual);

    const T kMin = std::numeric_limits<T>::min();
    const T kMax = std::numeric_limits<T>::max();

    // INT_MIN / -1 -> saturates to INT_MIN in our trunc path (scalar ref is UB)
    {
      const auto params = ComputeDivisorParams(T(-1));
      for (size_t i = 0; i < N; ++i) {
        dividend[i] = kMin;
        expected[i] = kMin;
      }
      const auto q = IntDiv(d, Load(d, dividend.get()), params);
      Store(q, d, actual.get());
      for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(kMin, actual[i]);
    }

    // Division by -1 for various values, with special-case for kMin
    {
      const auto params = ComputeDivisorParams(T(-1));
      const T vals[] = {kMin, T(kMin + 1), T(-100), T(-1), T(0), T(1), T(100), T(kMax - 1), kMax};
      for (T v : vals) {
        for (size_t i = 0; i < N; ++i) {
          dividend[i] = v;
          expected[i] = (v == kMin) ? kMin : static_cast<T>(-v);
        }
        const auto q = IntDiv(d, Load(d, dividend.get()), params);
        Store(q, d, actual.get());
        for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }

    // Negative power-of-2 divisors
    for (int shift = 1; shift < 7 && shift < static_cast<int>(sizeof(T) * 8 - 1); ++shift) {
      const T divisor = static_cast<T>(-(T(1) << shift));
      const auto params = ComputeDivisorParams(divisor);
      const T vals[] = {T(-64), T(-17), T(-1), T(0), T(1), T(17), T(64)};
      for (T v : vals) {
        for (size_t i = 0; i < N; ++i) {
          dividend[i] = v;
          expected[i] = static_cast<T>(v / divisor);
        }
        const auto q = IntDiv(d, Load(d, dividend.get()), params);
        Store(q, d, actual.get());
        for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }
};

// [ADDED_FLOOR]
template <typename T>
struct AddedFloorDivision {
  template <class T2, class D>
  HWY_NOINLINE void operator()(T2 /*unused*/, D d) {
    const size_t N = Lanes(d);
    auto dividend = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);
    HWY_ASSERT(dividend && expected && actual);

    std::vector<T> divisors = {1, 2, 3, 5, 7, 11, 17, 100};
    if (hwy::IsSigned<T>()) {
      std::vector<T> neg = {T(-1), T(-2), T(-3), T(-5), T(-7), T(-11), T(-17), T(-100)};
      divisors.insert(divisors.end(), neg.begin(), neg.end());
    }
    for (T divisor : divisors) {
      const auto params = ComputeDivisorParams(divisor);
      const T start = hwy::IsSigned<T>() ? static_cast<T>(-50) : T(0);
      const T end = static_cast<T>(50);
      for (T base = start; base < end; base += 3) {
        for (size_t i = 0; i < N; ++i) {
          const T v = static_cast<T>(base + i);
          dividend[i] = v;
          expected[i] = AddedFloorDivScalar(v, divisor);
        }
        const auto q = IntDivFloor(d, Load(d, dividend.get()), params);
        Store(q, d, actual.get());
        for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }
};

// [ADDED_CONVENIENCE]
struct AddedDivideByScalar {
  template <typename T2, class D>
  HWY_NOINLINE void operator()(T2 /*unused*/, D d) {
    using T = T2;
    const size_t N = Lanes(d);
    auto dividend = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);
    HWY_ASSERT(dividend && expected && actual);

    const T divisors[] = {1, 3, 7, 16, 31, 100};
    for (T divisor : divisors) {
      if (divisor <= 0) continue;  // keep unsigned-friendly
      for (size_t i = 0; i < N; ++i) {
        dividend[i] = static_cast<T>(i * 7 + 100);
        expected[i] = static_cast<T>(dividend[i] / divisor);
      }
      const auto v = Load(d, dividend.get());
      const auto q = DivideByScalar(d, v, divisor);
      Store(q, d, actual.get());
      for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(expected[i], actual[i]);
    }
  }
};

// [ADDED_ARRAY]
struct AddedArrayDivision {
  template <typename T>
  void RunOne() {
    constexpr size_t kCount = 127;
    auto array = AllocateAligned<T>(kCount);
    auto expected = AllocateAligned<T>(kCount);
    HWY_ASSERT(array && expected);
    const T divisor = 11;
    for (size_t i = 0; i < kCount; ++i) {
      array[i] = static_cast<T>((i * 13) % 256);
      expected[i] = static_cast<T>(array[i] / divisor);
    }
    DivideArrayByScalar(array.get(), kCount, divisor);
    for (size_t i = 0; i < kCount; ++i) HWY_ASSERT_EQ(expected[i], array[i]);
  }
  template <typename T2, class D>
  HWY_NOINLINE void operator()(T2 /*unused*/, D /*d*/) {
    using T = T2;
    RunOne<T>();
    if (hwy::IsSigned<T>()) {
      constexpr size_t kCount = 100;
      auto array = AllocateAligned<T>(kCount);
      auto expected = AllocateAligned<T>(kCount);
      HWY_ASSERT(array && expected);
      const T divisor = -7;
      for (size_t i = 0; i < kCount; ++i) {
        array[i] = static_cast<T>(static_cast<int>(i) - 50);
        expected[i] = AddedFloorDivScalar(array[i], divisor);
      }
      auto copy = AllocateAligned<T>(kCount);
      memcpy(copy.get(), array.get(), kCount * sizeof(T));
      FloorDivideArrayByScalar(copy.get(), kCount, divisor);
      for (size_t i = 0; i < kCount; ++i) HWY_ASSERT_EQ(expected[i], copy[i]);
    }
  }
};

// [ADDED_RANDOM]
template <typename T>
struct AddedRandomDivision {
  template <class T2, class D>
  HWY_NOINLINE void operator()(T2 /*unused*/, D d) {
    const size_t N = Lanes(d);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dividend_dist(hwy::IsSigned<T>() ? -1000 : 0, 1000);
    std::uniform_int_distribution<int> divisor_dist(hwy::IsSigned<T>() ? -100 : 1, 100);

    auto dividend = AllocateAligned<T>(N);
    auto expected = AllocateAligned<T>(N);
    auto actual = AllocateAligned<T>(N);
    HWY_ASSERT(dividend && expected && actual);

    for (size_t iter = 0; iter < 100; ++iter) {
      T divisor = static_cast<T>(divisor_dist(rng));
      if (divisor == 0) divisor = 1;
      const auto params = ComputeDivisorParams(divisor);

      for (size_t i = 0; i < N; ++i) {
        dividend[i] = static_cast<T>(dividend_dist(rng));
        expected[i] = static_cast<T>(dividend[i] / divisor);
      }
      const auto q = IntDiv(d, Load(d, dividend.get()), params);
      Store(q, d, actual.get());
      for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(expected[i], actual[i]);

      if (hwy::IsSigned<T>()) {
        for (size_t i = 0; i < N; ++i) expected[i] = AddedFloorDivScalar(dividend[i], divisor);
        const auto fq = IntDivFloor(d, Load(d, dividend.get()), params);
        Store(fq, d, actual.get());
        for (size_t i = 0; i < N; ++i) HWY_ASSERT_EQ(expected[i], actual[i]);
      }
    }
  }
};

// ============================================================================
// Drivers  [DRIVERS]
// ============================================================================
HWY_NOINLINE void TestAllIntDivUnsigned() {
  ForUnsignedTypes(ForPartialVectors<TestIntDivUnsigned<uint8_t>>());
  ForUnsignedTypes(ForPartialVectors<TestIntDivUnsigned<uint16_t>>());
  ForUnsignedTypes(ForPartialVectors<TestIntDivUnsigned<uint32_t>>());
#if HWY_HAVE_INTEGER64
  ForUnsignedTypes(ForPartialVectors<TestIntDivUnsigned<uint64_t>>());
#endif
  ForUnsignedTypes(ForPartialVectors<TestUnsignedDivideArrayByScalar>());
}

HWY_NOINLINE void TestAllIntDivSigned() {
  ForSignedTypes(ForPartialVectors<TestIntDivSigned<int8_t>>());
  ForSignedTypes(ForPartialVectors<TestIntDivSigned<int16_t>>());
  ForSignedTypes(ForPartialVectors<TestIntDivSigned<int32_t>>());
#if HWY_HAVE_INTEGER64
  ForSignedTypes(ForPartialVectors<TestIntDivSigned<int64_t>>());
#endif
  ForSignedTypes(ForPartialVectors<TestSignedDivideArrayByScalar>());
}

HWY_NOINLINE void TestAllSignedFloor() {
  ForSignedTypes(ForPartialVectors<TestSignedFloorDivision<int8_t>>());
  ForSignedTypes(ForPartialVectors<TestSignedFloorDivision<int16_t>>());
  ForSignedTypes(ForPartialVectors<TestSignedFloorDivision<int32_t>>());
#if HWY_HAVE_INTEGER64
  ForSignedTypes(ForPartialVectors<TestSignedFloorDivision<int64_t>>());
#endif
}

HWY_NOINLINE void TestAllIntDivEdge() {
  ForIntegerTypes(ForPartialVectors<TestIntDivEdgeCases>());
}

HWY_NOINLINE void TestAllDivideByScalar() {
  ForIntegerTypes(ForPartialVectors<TestDivideByScalarConvenience>());
}

HWY_NOINLINE void TestAllDirectedLoopSnippet() {
  ForIntegerTypes(ForPartialVectors<TestDirectedLoopSnippet>());
}

HWY_NOINLINE void TestAllDivideHighBySanity() {
  TestDivideHighBySanity{}();
}

HWY_NOINLINE void TestAllOptionalPerf() {
  ForPartialVectors<TestOptionalPerf>()(HWY_FULL(uint32_t)());
#if HWY_HAVE_INTEGER64
  ForPartialVectors<TestOptionalPerf>()(HWY_FULL(uint64_t)());
#endif
}

// ---- Added drivers (from your second file) ----
HWY_NOINLINE void TestAllAddedBasicDivision() {
  ForUnsignedTypes(ForPartialVectors<AddedBasicDivision<uint8_t>>());
  ForUnsignedTypes(ForPartialVectors<AddedBasicDivision<uint16_t>>());
  ForUnsignedTypes(ForPartialVectors<AddedBasicDivision<uint32_t>>());
#if HWY_HAVE_INTEGER64
  ForUnsignedTypes(ForPartialVectors<AddedBasicDivision<uint64_t>>());
#endif
  ForSignedTypes(ForPartialVectors<AddedBasicDivision<int8_t>>());
  ForSignedTypes(ForPartialVectors<AddedBasicDivision<int16_t>>());
  ForSignedTypes(ForPartialVectors<AddedBasicDivision<int32_t>>());
#if HWY_HAVE_INTEGER64
  ForSignedTypes(ForPartialVectors<AddedBasicDivision<int64_t>>());
#endif
}

HWY_NOINLINE void TestAllAddedPowerOf2() {
  ForUnsignedTypes(ForPartialVectors<AddedPowerOf2Division<uint8_t>>());
  ForUnsignedTypes(ForPartialVectors<AddedPowerOf2Division<uint16_t>>());
  ForUnsignedTypes(ForPartialVectors<AddedPowerOf2Division<uint32_t>>());
#if HWY_HAVE_INTEGER64
  ForUnsignedTypes(ForPartialVectors<AddedPowerOf2Division<uint64_t>>());
#endif
  ForSignedTypes(ForPartialVectors<AddedPowerOf2Division<int8_t>>());
  ForSignedTypes(ForPartialVectors<AddedPowerOf2Division<int16_t>>());
  ForSignedTypes(ForPartialVectors<AddedPowerOf2Division<int32_t>>());
#if HWY_HAVE_INTEGER64
  ForSignedTypes(ForPartialVectors<AddedPowerOf2Division<int64_t>>());
#endif
}

HWY_NOINLINE void TestAllAddedSignedEdge() {
  ForSignedTypes(ForPartialVectors<AddedSignedEdgeCases<int8_t>>());
  ForSignedTypes(ForPartialVectors<AddedSignedEdgeCases<int16_t>>());
  ForSignedTypes(ForPartialVectors<AddedSignedEdgeCases<int32_t>>());
#if HWY_HAVE_INTEGER64
  ForSignedTypes(ForPartialVectors<AddedSignedEdgeCases<int64_t>>());
#endif
}

HWY_NOINLINE void TestAllAddedFloor() {
  ForUnsignedTypes(ForPartialVectors<AddedFloorDivision<uint8_t>>());
  ForUnsignedTypes(ForPartialVectors<AddedFloorDivision<uint16_t>>());
  ForUnsignedTypes(ForPartialVectors<AddedFloorDivision<uint32_t>>());
#if HWY_HAVE_INTEGER64
  ForUnsignedTypes(ForPartialVectors<AddedFloorDivision<uint64_t>>());
#endif
  ForSignedTypes(ForPartialVectors<AddedFloorDivision<int8_t>>());
  ForSignedTypes(ForPartialVectors<AddedFloorDivision<int16_t>>());
  ForSignedTypes(ForPartialVectors<AddedFloorDivision<int32_t>>());
#if HWY_HAVE_INTEGER64
  ForSignedTypes(ForPartialVectors<AddedFloorDivision<int64_t>>());
#endif
}

HWY_NOINLINE void TestAllAddedConvenience() {
  ForUnsignedTypes(ForPartialVectors<AddedDivideByScalar>());
  ForSignedTypes(ForPartialVectors<AddedDivideByScalar>());
}

HWY_NOINLINE void TestAllAddedArrayOps() {
  ForUnsignedTypes(ForPartialVectors<AddedArrayDivision>());
  ForSignedTypes(ForPartialVectors<AddedArrayDivision>());
}

HWY_NOINLINE void TestAllAddedRandom() {
  ForUnsignedTypes(ForPartialVectors<AddedRandomDivision<uint8_t>>());
  ForUnsignedTypes(ForPartialVectors<AddedRandomDivision<uint16_t>>());
  ForUnsignedTypes(ForPartialVectors<AddedRandomDivision<uint32_t>>());
#if HWY_HAVE_INTEGER64
  ForUnsignedTypes(ForPartialVectors<AddedRandomDivision<uint64_t>>());
#endif
  ForSignedTypes(ForPartialVectors<AddedRandomDivision<int8_t>>());
  ForSignedTypes(ForPartialVectors<AddedRandomDivision<int16_t>>());
  ForSignedTypes(ForPartialVectors<AddedRandomDivision<int32_t>>());
#if HWY_HAVE_INTEGER64
  ForSignedTypes(ForPartialVectors<AddedRandomDivision<int64_t>>());
#endif
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(IntDivTest);

// Core batteries
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllIntDivUnsigned);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllIntDivSigned);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllSignedFloor);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllIntDivEdge);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllDivideByScalar);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllDirectedLoopSnippet);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllDivideHighBySanity);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllOptionalPerf);

// Added suites (your second file)
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllAddedBasicDivision);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllAddedPowerOf2);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllAddedSignedEdge);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllAddedFloor);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllAddedConvenience);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllAddedArrayOps);
HWY_EXPORT_AND_TEST_P(IntDivTest, TestAllAddedRandom);

HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE