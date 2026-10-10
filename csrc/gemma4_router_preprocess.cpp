// SPDX-License-Identifier: Apache-2.0
#include "core/registration.h"
#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>

namespace {
constexpr int kHidden = 2816;
constexpr int kThreads = 256;

at::Tensor router_preprocess(
    const at::Tensor& input,
    const at::Tensor& root,
    const at::Tensor& scale,
    double epsilon) {
  TORCH_CHECK(
      input.is_xpu() && input.dim() == 2 && root.device() == input.device() &&
          scale.device() == input.device() && root.numel() == 1 &&
          root.dim() <= 2 && scale.dim() == 1 &&
          scale.numel() == input.size(1) &&
          root.scalar_type() == input.scalar_type() &&
          scale.scalar_type() == input.scalar_type() && epsilon > 0,
      "gemma4_router_preprocess requires XPU [M,H], scalar root, [H] scale "
      "of the same dtype/device and positive epsilon");
  const at::DeviceGuard guard(input.device());
  if (input.scalar_type() != at::kHalf || input.size(1) != kHidden ||
      input.size(0) < 1 || input.size(0) > 8 || !input.is_contiguous() ||
      !root.is_contiguous() || !scale.is_contiguous()) {
    auto x = input.to(at::kFloat);
    auto variance = x.pow(2).mean(-1, true);
    return (x * (variance + epsilon).rsqrt()).to(input.scalar_type()) * root *
           scale;
  }
  auto output = at::empty(input.sizes(), input.options());
  const auto* x =
      reinterpret_cast<const sycl::half*>(input.data_ptr<at::Half>());
  const auto* r =
      reinterpret_cast<const sycl::half*>(root.data_ptr<at::Half>());
  const auto* s =
      reinterpret_cast<const sycl::half*>(scale.data_ptr<at::Half>());
  auto* y = reinterpret_cast<sycl::half*>(output.data_ptr<at::Half>());
  const float eps = static_cast<float>(epsilon);
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  queue.parallel_for(
      sycl::nd_range<1>(input.size(0) * kThreads, kThreads),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int row = item.get_group_linear_id();
        const int tid = item.get_local_linear_id();
        float sum = 0;
        for (int h = tid; h < kHidden; h += kThreads) {
          const float value = static_cast<float>(x[row * kHidden + h]);
          sum += value * value;
        }
        sum =
            sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
        const float inv = sycl::rsqrt(sum / kHidden + eps);
        const float root_value = static_cast<float>(r[0]);
        for (int h = tid; h < kHidden; h += kThreads) {
          // Keep both FP16 rounding points from the original Router input.
          const sycl::half norm =
              sycl::half(static_cast<float>(x[row * kHidden + h]) * inv);
          const sycl::half rooted =
              sycl::half(static_cast<float>(norm) * root_value);
          y[row * kHidden + h] =
              sycl::half(static_cast<float>(rooted) * static_cast<float>(s[h]));
        }
      });
  return output;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_xpu_C, m) {
  m.def(
      "gemma4_router_preprocess(Tensor input, Tensor root, Tensor scale, float "
      "epsilon) -> Tensor");
  m.impl("gemma4_router_preprocess", torch::kXPU, &router_preprocess);
  m.impl(
      "gemma4_router_preprocess",
      torch::kMeta,
      [](const at::Tensor& input,
         const at::Tensor& root,
         const at::Tensor& scale,
         double epsilon) {
        return at::empty_symint(input.sym_sizes(), input.options());
      });
}
REGISTER_EXTENSION(_gemma_compile_C)
