// SPDX-License-Identifier: Apache-2.0
#include "core/registration.h"
#include "nt_split_mainloop.hpp"

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUFunctions.h>
#include <ATen/ops/linear.h>
#include <cutlass/numeric_types.h>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <torch/library.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>

namespace vllm::fp16_linear_tla {

constexpr int kHidden = 2816;
constexpr int kRouterN = 128;
constexpr int kTileK = 32;

constexpr int kRouterSplits = 16;
using Element = cutlass::half_t;

template <int TileN, int Subgroups>
using NtMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, Element>>,
    Layout<Shape<_8, Int<TileN>, _32>>,
    Layout<Shape<_1, Int<Subgroups>, _1>, Stride<Int<Subgroups>, _1, _0>>>::
    TiledMMA;

class Fp16RouterNtSplitK;
class Fp16RouterReduce;

at::Tensor fp16_router_impl(const at::Tensor& input, const at::Tensor& weight) {
  const int64_t m = input.size(0);
  constexpr int n = kRouterN;
  const c10::DeviceGuard guard(input.device());
  auto output = at::empty({m, n}, input.options());
  const auto* a = reinterpret_cast<const Element*>(input.data_ptr<at::Half>());
  const auto* b = reinterpret_cast<const Element*>(weight.data_ptr<at::Half>());
  auto* out = reinterpret_cast<Element*>(output.data_ptr<at::Half>());
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  const syclex::properties props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};

  using Mma = NtMma<16, 1>;
  constexpr int group_size = size(Mma{});
  static_assert(group_size == 16);
  auto partial =
      at::empty({kRouterSplits, m, n}, input.options().dtype(at::kFloat));
  auto* p = partial.data_ptr<float>();
  // Narrow N gives only eight N tiles. Split K into 16 groups so that 128
  // work-groups execute 5/6 K tiles each; all 88 K tiles are covered exactly.
  queue.parallel_for<Fp16RouterNtSplitK>(
      sycl::nd_range<3>(
          sycl::range<3>(kRouterSplits, n / 16, group_size),
          sycl::range<3>(1, 1, group_size)),
      props,
      [=](sycl::nd_item<3> item) {
        const int split = item.get_group(0);
        constexpr int tiles_k = kHidden / kTileK;
        const int k_begin = tiles_k * split / kRouterSplits;
        const int k_end = tiles_k * (split + 1) / kRouterSplits;
        auto A = make_tensor(
            make_gmem_ptr(a),
            make_layout(
                make_shape(int(m), kHidden), make_stride(kHidden, _1{})));
        auto B = make_tensor(
            make_gmem_ptr(b),
            make_layout(make_shape(n, kHidden), make_stride(kHidden, _1{})));
        auto C = make_tensor(
            make_gmem_ptr(p + int64_t(split) * m * n),
            make_layout(make_shape(int(m), n), make_stride(n, _1{})));
        nt_split_mainloop<void, void, void>(
            A,
            B,
            C,
            make_coord(0, int(item.get_group(1)), _, 0),
            Mma{},
            k_begin,
            k_end);
      });
  const int total = int(m) * n;
  queue.parallel_for<Fp16RouterReduce>(
      sycl::nd_range<1>(((total + 255) / 256) * 256, 256),
      [=](sycl::nd_item<1> item) {
        const int index = item.get_global_linear_id();
        if (index < total) {
          float sum = 0.0f;
#pragma unroll
          for (int split = 0; split < kRouterSplits; ++split) {
            sum += p[int64_t(split) * total + index];
          }
          // Preserve GateLinear's FP16 GEMM storage rounding. Its later
          // FP32 cast remains outside this operator and is not folded away.
          out[index] = Element(sum);
        }
      });
  return output;
}

// All selection happens before candidate allocations or device submissions.
bool fp16_router_supported(const at::Tensor& input, const at::Tensor& weight) {
  if (!input.is_xpu() || input.scalar_type() != at::kHalf || input.dim() != 2 ||
      !input.is_contiguous() || input.size(0) < 1 || input.size(0) > 8 ||
      weight.device() != input.device() || weight.scalar_type() != at::kHalf ||
      weight.dim() != 2 || !weight.is_contiguous() ||
      weight.size(1) != input.size(1) ||
      (weight.size(0) != kRouterN || input.size(1) != kHidden)) {
    return false;
  }
  if (reinterpret_cast<uintptr_t>(input.data_ptr()) % 64 != 0 ||
      reinterpret_cast<uintptr_t>(weight.data_ptr()) % 64 != 0) {
    return false;
  }
  // Cache only immutable architecture metadata, never tensor/request state.
  static thread_local int last_device = -1;
  static thread_local bool last_supported = false;
  const int device = input.get_device();
  if (last_device != device) {
    namespace sx = sycl::ext::oneapi::experimental;
    last_supported = c10::xpu::get_raw_device(device)
                         .get_info<sx::info::device::architecture>() ==
                     sx::architecture::intel_gpu_bmg_g31;
    last_device = device;
  }
  return last_supported;
}

at::Tensor unquantized_gemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  // Keep selection and validation inside the existing XPU operator.
  if (!bias.has_value() && fp16_router_supported(input, weight)) {
    return fp16_router_impl(input, weight);
  }
  return at::linear(input, weight, bias);
}

at::Tensor unquantized_gemm_meta(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  return at::linear(input, weight, bias);
}

}  // namespace vllm::fp16_linear_tla

TORCH_LIBRARY_FRAGMENT(_xpu_C, m) {
  m.def(
      "unquantized_gemm(Tensor input, Tensor weight, Tensor? bias=None) -> "
      "Tensor");
  m.impl(
      "unquantized_gemm",
      torch::kXPU,
      &vllm::fp16_linear_tla::unquantized_gemm);
  m.impl(
      "unquantized_gemm",
      torch::kMeta,
      &vllm::fp16_linear_tla::unquantized_gemm_meta);
}

REGISTER_EXTENSION(_fp16_C)
