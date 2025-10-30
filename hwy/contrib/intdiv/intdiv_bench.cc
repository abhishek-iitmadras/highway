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

// ============================================================================
// Per-target compilation setup (Approach A strength)
// ============================================================================
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/intdiv/intdiv_bench.cc"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "hwy/foreach_target.h"   // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"     // IWYU pragma: keep (for MeasureClosure)
#include "hwy/tests/hwy_gtest.h" 

// Use the *public* header that re-exports the HWY_NAMESPACE API.
#include "hwy/contrib/intdiv/intdiv.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// ============================================================================
// Type Helpers
// ============================================================================

template <typename T>
static const char* TypeName() {
  if constexpr (std::is_same<T, uint8_t>::value)  return "u8";
  if constexpr (std::is_same<T, int8_t>::value)   return "i8";
  if constexpr (std::is_same<T, uint16_t>::value) return "u16";
  if constexpr (std::is_same<T, int16_t>::value)  return "i16";
  if constexpr (std::is_same<T, uint32_t>::value) return "u32";
  if constexpr (std::is_same<T, int32_t>::value)  return "i32";
#if HWY_HAVE_INTEGER64
  if constexpr (std::is_same<T, uint64_t>::value) return "u64";
  if constexpr (std::is_same<T, int64_t>::value)  return "i64";
#endif
  return "unknown";
}

// Choose the right param type (unsigned vs signed)
template <typename T>
using DivParamsT = std::conditional_t<std::is_signed<T>::value,
                                      DivisorParamsS<T>, DivisorParamsU<T>>;

// ============================================================================
// Benchmark Configuration
// ============================================================================

constexpr size_t kDefaultWorkingSet = 1u << 20;  // 1M elements

// ============================================================================
// Benchmark State (unified structure)
// ============================================================================

template <typename T>
struct DivisionBenchState {
  std::vector<T> dividend;
  std::vector<T> result;
  std::vector<T> result_floor;  // For floor division testing
  T divisor;
  DivParamsT<T> params;
  size_t n;

  explicit DivisionBenchState(size_t count = kDefaultWorkingSet)
      : dividend(count), result(count), result_floor(count), n(count) {
    std::mt19937 rng(12345);

    using Lim = std::numeric_limits<T>;
    if constexpr (std::is_signed<T>::value) {
      // Keep within ~half the range to avoid edge-overflows when multiplying.
      const auto lo = T(Lim::min() / 2);
      const auto hi = T(Lim::max() / 2);
      std::uniform_int_distribution<int64_t> dist(int64_t(lo), int64_t(hi));
      for (size_t i = 0; i < n; ++i) {
        dividend[i] = static_cast<T>(dist(rng));
      }
      divisor = T(12345);
      if (divisor == 0) divisor = T(7);
    } else {
      const auto hi = typename std::make_unsigned<T>::type(Lim::max()) / 2u;
      std::uniform_int_distribution<uint64_t> dist(0ull, uint64_t(hi));
      for (size_t i = 0; i < n; ++i) {
        dividend[i] = static_cast<T>(dist(rng));
      }
      divisor = T(12345u);
      if (divisor == 0) divisor = T(7u);
    }

    params = ComputeDivisorParams(divisor);
  }
};

// ============================================================================
// Scalar Reference Implementation
// ============================================================================

template <typename T>
HWY_NOINLINE void ScalarDivide(const T* HWY_RESTRICT src, T* HWY_RESTRICT dst,
                                size_t count, T divisor) {
  for (size_t i = 0; i < count; ++i) {
    dst[i] = src[i] / divisor;  // C truncation semantics
  }
}

template <typename T, HWY_IF_SIGNED(T)>
HWY_NOINLINE void ScalarFloorDivide(const T* HWY_RESTRICT src,
                                     T* HWY_RESTRICT dst, size_t count,
                                     T divisor) {
  for (size_t i = 0; i < count; ++i) {
    T q = src[i] / divisor;
    T r = src[i] % divisor;
    // Floor correction: if remainder has opposite sign of divisor, adjust
    if ((r != 0) && ((r < 0) != (divisor < 0))) {
      q -= 1;
    }
    dst[i] = q;
  }
}

// ============================================================================
// Highway SIMD Implementation
// ============================================================================

template <typename T>
HWY_NOINLINE void SimdDivide(const T* HWY_RESTRICT src, T* HWY_RESTRICT dst,
                              size_t count, const DivParamsT<T>& params) {
  const ScalableTag<T> d;
  const size_t N = Lanes(d);
  size_t i = 0;

  // Vectorized main loop
  for (; i + N <= count; i += N) {
    const auto v = LoadU(d, src + i);
    const auto qv = IntDiv(d, v, params);
    StoreU(qv, d, dst + i);
  }

  // Tail: use LoadN/StoreN for consistency
  if (i < count) {
    const size_t remain = count - i;
    const auto v = LoadN(d, src + i, remain);
    const auto qv = IntDiv(d, v, params);
    StoreN(qv, d, dst + i, remain);
  }
}

template <typename T, HWY_IF_SIGNED(T)>
HWY_NOINLINE void SimdFloorDivide(const T* HWY_RESTRICT src,
                                   T* HWY_RESTRICT dst, size_t count,
                                   const DivParamsT<T>& params) {
  const ScalableTag<T> d;
  const size_t N = Lanes(d);
  size_t i = 0;

  // Vectorized main loop
  for (; i + N <= count; i += N) {
    const auto v = LoadU(d, src + i);
    const auto qv = IntDivFloor(d, v, params);
    StoreU(qv, d, dst + i);
  }

  // Tail
  if (i < count) {
    const size_t remain = count - i;
    const auto v = LoadN(d, src + i, remain);
    const auto qv = IntDivFloor(d, v, params);
    StoreN(qv, d, dst + i, remain);
  }
}

// ============================================================================
// Correctness Verification
// ============================================================================

template <typename T>
static void PrintOne(const T v, bool is_signed) {
  if (is_signed) {
    std::printf("%lld", static_cast<long long>(v));
  } else {
    using U = std::make_unsigned_t<T>;
    std::printf("%llu", static_cast<unsigned long long>(static_cast<U>(v)));
  }
}

template <typename T>
bool VerifyResults(const T* simd_result, const T* scalar_result, size_t count,
                   const char* test_name) {
  for (size_t i = 0; i < count; ++i) {
    if (simd_result[i] != scalar_result[i]) {
      std::fprintf(stderr, "❌ %s FAILED at index %zu: SIMD=", test_name, i);
      PrintOne(simd_result[i], std::is_signed<T>::value);
      std::fprintf(stderr, ", Scalar=");
      PrintOne(scalar_result[i], std::is_signed<T>::value);
      std::fprintf(stderr, "\n");
      return false;
    }
  }
  std::printf("✓ %s passed\n", test_name);
  return true;
}

// ============================================================================
// Nanobenchmark Measurement Utility
// ============================================================================

// Result structure holding benchmark measurements
struct BenchmarkTiming {
  float ticks;        // Raw CPU ticks from nanobenchmark.h Result
  float variability;  // Measurement variability (median absolute deviation)
  bool success;
};

// Measures a closure using nanobenchmark infrastructure (accounts for timer
// overhead, fence instructions, and provides robust statistics).
// Returns raw CPU ticks and measurement variability.
template <class Closure>
static BenchmarkTiming MeasureViaNanobench(const Closure& run_once) {
  // Any non-empty input distribution works; we ignore it in the closure.
  const hwy::FuncInput inputs[1] = {0};
  hwy::Result res{};
  hwy::Params p;
  p.verbose = false;  // Quiet output
  p.max_evals = 9;    // Default is fine

  const size_t wrote = hwy::MeasureClosure(run_once, inputs, 1, &res, p);
  if (wrote == 0) return {0, 0, false};

  // Return raw CPU ticks from nanobenchmark Result
  return {res.ticks, res.variability, true};
}

// ============================================================================
// Comprehensive Benchmark Suite
// ============================================================================

template <typename T>
struct BenchmarkResults {
  float scalar_ticks;
  float simd_ticks;
  double speedup;
  float ticks_per_element_scalar;
  float ticks_per_element_simd;
  bool verified;
};

template <typename T>
static BenchmarkResults<T> BenchmarkDivisor(const char* type_name,
                                             const char* divisor_type,
                                             T divisor, size_t working_set) {
  DivisionBenchState<T> st(working_set);
  st.divisor = divisor;
  st.params = ComputeDivisorParams(divisor);

  // Step 1: Correctness Verification
  std::printf("  Verifying %s (div by ", divisor_type);
  PrintOne(divisor, std::is_signed<T>::value);
  std::printf(", working_set=%zu)... ", working_set);

  // Full verification on smaller subset
  size_t verify_count = std::min(size_t{1000}, working_set);
  ScalarDivide(st.dividend.data(), st.result.data(), verify_count, st.divisor);
  SimdDivide(st.dividend.data(), st.result_floor.data(), verify_count,
             st.params);

  if (!VerifyResults(st.result_floor.data(), st.result.data(), verify_count,
                     "Truncating division")) {
    return {0, 0, 0, 0, 0, false};
  }

  // Step 2: Benchmark Scalar using nanobenchmark
  auto scalar_once = [&]() -> hwy::FuncOutput {
    ScalarDivide(st.dividend.data(), st.result.data(), st.n, st.divisor);
    // Return proof-of-work to prevent compiler elision
    return static_cast<hwy::FuncOutput>(st.result[0]);
  };

  const auto scalar_timing = MeasureViaNanobench(scalar_once);
  if (!scalar_timing.success) {
    return {0, 0, 0, 0, 0, false};
  }

  // Step 3: Benchmark SIMD using nanobenchmark
  auto simd_once = [&]() -> hwy::FuncOutput {
    SimdDivide(st.dividend.data(), st.result_floor.data(), st.n, st.params);
    // Return proof-of-work to prevent compiler elision
    return static_cast<hwy::FuncOutput>(st.result_floor[0]);
  };

  const auto simd_timing = MeasureViaNanobench(simd_once);
  if (!simd_timing.success) {
    return {scalar_timing.ticks, 0, 0, 0, 0, false};
  }

  // Step 4: Final verification
  std::printf("  Final verification... ");
  if (!VerifyResults(st.result_floor.data(), st.result.data(), st.n,
                     "Final")) {
    return {scalar_timing.ticks, simd_timing.ticks, 0,
            scalar_timing.ticks / st.n, simd_timing.ticks / st.n, false};
  }

  double speedup = (simd_timing.ticks > 0.0f)
                       ? static_cast<double>(scalar_timing.ticks) /
                             static_cast<double>(simd_timing.ticks)
                       : 0.0;

  float scalar_tpe = scalar_timing.ticks / static_cast<float>(st.n);
  float simd_tpe = simd_timing.ticks / static_cast<float>(st.n);

  std::printf(
      "\n    Scalar: %.2f ticks/elem  |  SIMD: %.2f ticks/elem  |  Speedup: %.2f×\n",
      scalar_tpe, simd_tpe, speedup);

  return {scalar_timing.ticks, simd_timing.ticks, speedup, scalar_tpe, simd_tpe,
          true};
}

template <typename T, HWY_IF_SIGNED(T)>
static BenchmarkResults<T> BenchmarkFloorDivisor(const char* type_name,
                                                  const char* divisor_type,
                                                  T divisor,
                                                  size_t working_set) {
  DivisionBenchState<T> st(working_set);
  st.divisor = divisor;
  st.params = ComputeDivisorParams(divisor);

  // Mix positive and negative dividends
  std::mt19937 rng(0xDEADBEEF);
  using Lim = std::numeric_limits<T>;
  const auto lo = T(Lim::min() / 2);
  const auto hi = T(Lim::max() / 2);
  std::uniform_int_distribution<int64_t> dist(int64_t(lo), int64_t(hi));
  for (size_t i = 0; i < st.n; ++i) {
    st.dividend[i] = static_cast<T>(dist(rng));
  }

  std::printf("  Verifying floor %s (div by ", divisor_type);
  PrintOne(divisor, std::is_signed<T>::value);
  std::printf(", working_set=%zu)... ", working_set);

  size_t verify_count = std::min(size_t{1000}, working_set);
  ScalarFloorDivide(st.dividend.data(), st.result.data(), verify_count,
                    st.divisor);
  SimdFloorDivide(st.dividend.data(), st.result_floor.data(), verify_count,
                  st.params);

  if (!VerifyResults(st.result_floor.data(), st.result.data(), verify_count,
                     "Floor division")) {
    return {0, 0, 0, 0, 0, false};
  }

  // Benchmark scalar floor division using nanobenchmark
  auto scalar_floor_once = [&]() -> hwy::FuncOutput {
    ScalarFloorDivide(st.dividend.data(), st.result.data(), st.n, st.divisor);
    return static_cast<hwy::FuncOutput>(st.result[0]);
  };

  const auto scalar_timing = MeasureViaNanobench(scalar_floor_once);
  if (!scalar_timing.success) {
    return {0, 0, 0, 0, 0, false};
  }

  // Benchmark SIMD floor division using nanobenchmark
  auto simd_floor_once = [&]() -> hwy::FuncOutput {
    SimdFloorDivide(st.dividend.data(), st.result_floor.data(), st.n,
                    st.params);
    return static_cast<hwy::FuncOutput>(st.result_floor[0]);
  };

  const auto simd_timing = MeasureViaNanobench(simd_floor_once);
  if (!simd_timing.success) {
    return {scalar_timing.ticks, 0, 0, 0, 0, false};
  }

  std::printf("  Final verification... ");
  if (!VerifyResults(st.result_floor.data(), st.result.data(), st.n,
                     "Final floor")) {
    return {scalar_timing.ticks, simd_timing.ticks, 0,
            scalar_timing.ticks / st.n, simd_timing.ticks / st.n, false};
  }

  double speedup = (simd_timing.ticks > 0.0f)
                       ? static_cast<double>(scalar_timing.ticks) /
                             static_cast<double>(simd_timing.ticks)
                       : 0.0;

  float scalar_tpe = scalar_timing.ticks / static_cast<float>(st.n);
  float simd_tpe = simd_timing.ticks / static_cast<float>(st.n);

  std::printf(
      "\n    Scalar: %.2f ticks/elem  |  SIMD: %.2f ticks/elem  |  Speedup: %.2f×\n",
      scalar_tpe, simd_tpe, speedup);

  return {scalar_timing.ticks, simd_timing.ticks, speedup, scalar_tpe, simd_tpe,
          true};
}

// ============================================================================
// Comprehensive Benchmark Suites (Approach A strength)
// ============================================================================

template <typename T>
static void BasicBenchmark() {
  std::printf("\n=== Basic Throughput (%s) ===\n", TypeName<T>());
  BenchmarkDivisor<T>(TypeName<T>(), "basic", static_cast<T>(12345),
                      kDefaultWorkingSet);
}

template <typename T>
static void CacheEffectsBenchmark() {
  std::printf("\n=== Cache Effects (%s: divisor=12345) ===\n", TypeName<T>());

  const size_t sizes[] = {
      size_t{1024},             // ~1K   (L1 happy)
      size_t{16 * 1024},        // ~16K  (L2)
      size_t{256 * 1024},       // ~256K (L3)
      size_t{4u * 1024 * 1024}  // ~4M   (beyond cache)
  };

  for (size_t sz : sizes) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%zu elems (%.1f KiB)",
             sz, (sz * sizeof(T)) / 1024.0);
    BenchmarkDivisor<T>(TypeName<T>(), buffer, static_cast<T>(12345), sz);
  }
}

template <typename T>
static void DivisorSweepBenchmark() {
  std::printf("\n=== Divisor Sweep (%s: 1M elements) ===\n", TypeName<T>());

  // Comprehensive divisor set: powers-of-2, primes, composites
  const T divisors[] = {
      T{1},     T{2},     T{3},     T{5},     T{7},     T{10},
      T{16},    T{17},    T{31},    T{32},    T{63},    T{64},
      T{65},    T{100},   T{127},   T{128},   T{255},   T{256},
      T{257},   T{511},   T{512},   T{1024},  T{4095},  T{4096},
      T{12345}, T{65535}, T{65536}};

  for (T d : divisors) {
    if (d == 0) continue;

    // Determine divisor type
    const char* dtype = "other";
    if (d == (d & -d)) {
      dtype = "pow2";
    } else if (d < 10) {
      dtype = "small";
    } else if (d < 100) {
      dtype = "medium";
    } else if (d < 1000) {
      dtype = "large";
    }

    char dstr[64];
    snprintf(dstr, sizeof(dstr), "%s=%llu", dtype,
             static_cast<unsigned long long>(
                 static_cast<std::make_unsigned_t<T>>(d)));

    BenchmarkDivisor<T>(TypeName<T>(), dstr, d, kDefaultWorkingSet);
  }
}

template <typename T, HWY_IF_SIGNED(T)>
static void FloorDivisionBenchmark() {
  std::printf("\n=== Floor Division (%s) ===\n", TypeName<T>());

  const T divisors[] = {T{7}, T{16}, T{127}, T{-13}, T{-127}};

  for (T d : divisors) {
    if (d == 0) continue;

    char dstr[64];
    snprintf(dstr, sizeof(dstr), "floor (div=%lld)", static_cast<long long>(d));

    BenchmarkFloorDivisor<T>(TypeName<T>(), dstr, d, kDefaultWorkingSet);
  }
}

// ============================================================================
// Main Benchmark Runner (Approach A: foreach_target integration)
// ============================================================================

static void RunAll() {
  std::printf("\n");
  std::printf(
      "╔═══════════════════════════════════════════════════════════════════╗\n");
  std::printf(
      "║    Highway Integer Division: Comprehensive Benchmark Suite        ║\n");
  std::printf(
      "║    Target: %-49s ║\n",
      hwy::TargetName(HWY_TARGET));
  std::printf(
      "╚═══════════════════════════════════════════════════════════════════╝\n");

  // ========================================================================
  // SECTION 1: Basic Throughput (all types)
  // ========================================================================
  std::printf("\n");
  std::printf(
      "┌───────────────────────────────────────────────────────────────────┐\n");
  std::printf(
      "│ SECTION 1: Basic Throughput (1M elements, div by 12345)          │\n");
  std::printf(
      "└───────────────────────────────────────────────────────────────────┘\n");

  BasicBenchmark<uint8_t>();
  BasicBenchmark<int8_t>();
  BasicBenchmark<uint16_t>();
  BasicBenchmark<int16_t>();
  BasicBenchmark<uint32_t>();
  BasicBenchmark<int32_t>();
#if HWY_HAVE_INTEGER64
  BasicBenchmark<uint64_t>();
  BasicBenchmark<int64_t>();
#endif

  // ========================================================================
  // SECTION 2: Cache Effects (int32_t as representative)
  // ========================================================================
  std::printf("\n");
  std::printf(
      "┌───────────────────────────────────────────────────────────────────┐\n");
  std::printf(
      "│ SECTION 2: Cache Effects (int32_t: L1 → L2 → L3 → Memory)       │\n");
  std::printf(
      "└───────────────────────────────────────────────────────────────────┘\n");

  CacheEffectsBenchmark<int32_t>();

  // ========================================================================
  // SECTION 3: Divisor Sweep (int32_t: pow2, primes, composites)
  // ========================================================================
  std::printf("\n");
  std::printf(
      "┌───────────────────────────────────────────────────────────────────┐\n");
  std::printf(
      "│ SECTION 3: Divisor Sweep (int32_t: powers-of-2, primes, etc.)   │\n");
  std::printf(
      "└───────────────────────────────────────────────────────────────────┘\n");

  DivisorSweepBenchmark<int32_t>();

  // ========================================================================
  // SECTION 4: Floor Division (signed types only)
  // ========================================================================
  std::printf("\n");
  std::printf(
      "┌───────────────────────────────────────────────────────────────────┐\n");
  std::printf("│ SECTION 4: Floor Division (Python/NumPy semantics)             │\n");
  std::printf(
      "└───────────────────────────────────────────────────────────────────┘\n");

  FloorDivisionBenchmark<int8_t>();
  FloorDivisionBenchmark<int16_t>();
  FloorDivisionBenchmark<int32_t>();
#if HWY_HAVE_INTEGER64
  FloorDivisionBenchmark<int64_t>();
#endif

  // ========================================================================
  // Summary
  // ========================================================================
  std::printf("\n");
  std::printf(
      "╔═══════════════════════════════════════════════════════════════════╗\n");
  std::printf(
      "║    Benchmark Suite Complete                                       ║\n");
  std::printf(
      "║    All tests verified for correctness before measurement          ║\n");
  std::printf(
      "║    Timing: nanobenchmark.h (CPU ticks, timer fences)              ║\n");
  std::printf(
      "║    Metric: ticks/element (frequency-independent)                  ║\n");
  std::printf(
      "║    Multi-target: foreach_target.h (per-target compilation)        ║\n");
  std::printf(
      "╚═══════════════════════════════════════════════════════════════════╝\n");
  std::printf("\n");
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

// ============================================================================
// Highway Test Infrastructure Integration (Approach A strength)
// ============================================================================
#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(IntDivBench);
HWY_EXPORT_AND_TEST_P(IntDivBench, RunAll);
HWY_AFTER_TEST();
}  // namespace hwy

HWY_TEST_MAIN();
#endif