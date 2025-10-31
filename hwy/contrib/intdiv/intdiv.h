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

// hwy/contrib/intdiv/intdiv.h
#ifndef HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_H_
#define HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_H_

// Public entry for invariant-integer division helpers built on Highway.
// This header provides the dispatcher and re-exports the API from the
// per-target implementation in intdiv-inl.h.

#include "hwy/highway.h"

// Tell foreach_target.h which implementation to include for each target.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/intdiv/intdiv-inl.h"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// After all targets have compiled the inl, expose a stable public API.
#if HWY_ONCE

namespace hwy {

/// ============================================================================
// Type aliases to access target-namespace types from dispatcher
// ============================================================================
template <class D>
using VecD = HWY_NAMESPACE::Vec<D>;

template <class D>
using TFromD_ = HWY_NAMESPACE::TFromD<D>;

// ============================================================================
// Re-export public types
// ============================================================================
template <typename T>
using DivisorParamsU = HWY_NAMESPACE::DivisorParamsU<T>;

template <typename T>
using DivisorParamsS = HWY_NAMESPACE::DivisorParamsS<T>;

// ============================================================================
// Re-export public functions (D-tagged, inline forwarders)
// ============================================================================
//
// We forward to the per-target implementations inside HWY_NAMESPACE.
// These wrappers are header-only and keep template signatures unchanged.

// ComputeDivisorParams (unsigned)
template <typename T, HWY_IF_T_SIZE(T, 1), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T d) {
  return HWY_NAMESPACE::ComputeDivisorParams<T>(d);
}
template <typename T, HWY_IF_T_SIZE(T, 2), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T d) {
  return HWY_NAMESPACE::ComputeDivisorParams<T>(d);
}
template <typename T, HWY_IF_T_SIZE(T, 4), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T d) {
  return HWY_NAMESPACE::ComputeDivisorParams<T>(d);
}
template <typename T, HWY_IF_T_SIZE(T, 8), HWY_IF_UNSIGNED(T)>
HWY_INLINE DivisorParamsU<T> ComputeDivisorParams(T d) {
  return HWY_NAMESPACE::ComputeDivisorParams<T>(d);
}

// ComputeDivisorParams (signed)
template <typename T, HWY_IF_T_SIZE(T, 1), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T d) {
  return HWY_NAMESPACE::ComputeDivisorParams<T>(d);
}
template <typename T, HWY_IF_T_SIZE(T, 2), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T d) {
  return HWY_NAMESPACE::ComputeDivisorParams<T>(d);
}
template <typename T, HWY_IF_T_SIZE(T, 4), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T d) {
  return HWY_NAMESPACE::ComputeDivisorParams<T>(d);
}
template <typename T, HWY_IF_T_SIZE(T, 8), HWY_IF_SIGNED(T)>
HWY_INLINE DivisorParamsS<T> ComputeDivisorParams(T d) {
  return HWY_NAMESPACE::ComputeDivisorParams<T>(d);
}

// IntDiv (unsigned / signed)
template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V IntDiv(D d, V a, const DivisorParamsU<T>& p) {
  return HWY_NAMESPACE::IntDiv(d, a, p);
}
template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V IntDiv(D d, V a, const DivisorParamsS<T>& p) {
  return HWY_NAMESPACE::IntDiv(d, a, p);
}

// IntDivFloor (unsigned == IntDiv, signed has floor semantics)
template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V IntDivFloor(D d, V a, const DivisorParamsU<T>& p) {
  return HWY_NAMESPACE::IntDivFloor(d, a, p);
}
template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V IntDivFloor(D d, V a, const DivisorParamsS<T>& p) {
  return HWY_NAMESPACE::IntDivFloor(d, a, p);
}

// DivideByScalar / FloorDivideByScalar (D-tagged convenience)
template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V DivideByScalar(D d, V a, T div) {
  return HWY_NAMESPACE::DivideByScalar(d, a, div);
}
template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V DivideByScalar(D d, V a, T div) {
  return HWY_NAMESPACE::DivideByScalar(d, a, div);
}
template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_SIGNED_D(D)>
HWY_INLINE V FloorDivideByScalar(D d, V a, T div) {
  return HWY_NAMESPACE::FloorDivideByScalar(d, a, div);
}
template <class D, class V = VecD<D>, typename T = TFromD_<D>, HWY_IF_UNSIGNED_D(D)>
HWY_INLINE V FloorDivideByScalar(D d, V a, T div) {
  return HWY_NAMESPACE::FloorDivideByScalar(d, a, div);
}

// --- Array helpers -----------------------------------------------------------
//
// These accept raw pointers and sizes. They’re good candidates for dynamic
// dispatch, but simple inline forwarders are also fine; users that want
// runtime dispatch can wrap with HWY_DYNAMIC_DISPATCH themselves or you can
// later add HWY_EXPORT + dynamic wrappers here.

template <typename T, HWY_IF_UNSIGNED(T)>
HWY_INLINE void DivideArrayByScalar(T* HWY_RESTRICT arr, size_t n, T div) {
  HWY_NAMESPACE::DivideArrayByScalar(arr, n, div);
}

template <typename T, HWY_IF_SIGNED(T)>
HWY_INLINE void DivideArrayByScalar(T* HWY_RESTRICT arr, size_t n, T div) {
  HWY_NAMESPACE::DivideArrayByScalar(arr, n, div);
}

template <typename T, HWY_IF_SIGNED(T)>
HWY_INLINE void FloorDivideArrayByScalar(T* HWY_RESTRICT arr, size_t n, T div) {
  HWY_NAMESPACE::FloorDivideArrayByScalar(arr, n, div);
}

template <typename T, HWY_IF_UNSIGNED(T)>
HWY_INLINE void FloorDivideArrayByScalar(T* HWY_RESTRICT arr, size_t n, T div) {
  HWY_NAMESPACE::FloorDivideArrayByScalar(arr, n, div);
}

}  // namespace hwy

#endif  // HWY_ONCE

#endif  // HIGHWAY_HWY_CONTRIB_INTDIV_INTDIV_H_
