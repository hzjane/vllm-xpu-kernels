// SPDX-License-Identifier: Apache-2.0
#include "moe_ops.h"
#include "core/registration.h"

#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>

namespace {
constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kSubgroup = 16;
constexpr int kWorkgroup = 128;

// One subgroup per token: keep the 128 keys in registers and select only the
// eight required experts. No SLM, workgroup barriers, or intermediate tensors.
template <typename Input>
struct Gemma4BatchTopK {
  const Input* logits;
  const void* expert_scales;
  bool scale_is_half;
  float* weights;
  int32_t* ids;
  int tokens;

  [[sycl::reqd_sub_group_size(kSubgroup)]] void
  operator()(sycl::nd_item<1> item) const {
    auto subgroup = item.get_sub_group();
    const int lane = subgroup.get_local_linear_id();
    const int token = item.get_global_linear_id() / kSubgroup;
    if (token >= tokens) return;
    int64_t keys[kExperts / kSubgroup];
    float maximum = -std::numeric_limits<float>::infinity();
#pragma unroll
    for (int i = 0; i < kExperts / kSubgroup; ++i) {
      const int expert = i * kSubgroup + lane;
      const float value = static_cast<float>(logits[token * kExperts + expert]);
      const uint32_t bits = sycl::bit_cast<uint32_t>(value);
      const uint32_t key =
          (bits & 0x80000000u) ? (bits ^ 0x80000000u) : (bits ^ 0xffffffffu);
      keys[i] = sycl::bit_cast<int64_t>(
          (static_cast<uint64_t>(key) << 32) | static_cast<uint64_t>(expert));
      maximum = sycl::fmax(maximum, value);
    }
    maximum =
        sycl::reduce_over_group(subgroup, maximum, sycl::maximum<float>());
    float exponentials[kTopK];
    int selected_ids[kTopK];
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
      int64_t key = std::numeric_limits<int64_t>::max();
#pragma unroll
      for (int i = 0; i < kExperts / kSubgroup; ++i) {
        key = sycl::min(key, keys[i]);
      }
      key = sycl::reduce_over_group(subgroup, key, sycl::minimum<int64_t>());
      const uint64_t packed = sycl::bit_cast<uint64_t>(key);
      const uint32_t ordered = packed >> 32;
      const uint32_t bits = (ordered & 0x80000000u) ? (ordered ^ 0xffffffffu)
                                                    : (ordered ^ 0x80000000u);
      const float value = sycl::bit_cast<float>(bits);
      selected_ids[rank] = packed & 0xffffffffu;
      exponentials[rank] = sycl::exp2((value - maximum) * 1.4426950408889634f);
#pragma unroll
      for (int i = 0; i < kExperts / kSubgroup; ++i) {
        if (keys[i] == key) keys[i] = std::numeric_limits<int64_t>::max();
      }
    }
    float sum = 0.0f;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank)
      sum += exponentials[rank];
    sum = sum > 0.0f ? sum : 1.0f;
    if (lane < kTopK) {
      const int expert = selected_ids[lane];
      const float scale =
          !expert_scales
              ? 1.0f
              : (scale_is_half
                     ? static_cast<float>(static_cast<const sycl::half*>(
                           expert_scales)[expert])
                     : static_cast<const float*>(expert_scales)[expert]);
      weights[token * kTopK + lane] = exponentials[lane] * (1.0f / sum) * scale;
      ids[token * kTopK + lane] = expert;
    }
  }
};

template <typename Input>
void launch(
    const torch::Tensor& logits,
    const std::optional<torch::Tensor>& scales,
    torch::Tensor& weights,
    torch::Tensor& ids) {
  auto& queue = c10::xpu::getCurrentXPUStream(logits.get_device()).queue();
  const int tokens = logits.size(0);
  const size_t global =
      (tokens * kSubgroup + kWorkgroup - 1) / kWorkgroup * kWorkgroup;
  queue.parallel_for(
      sycl::nd_range<1>(global, kWorkgroup),
      Gemma4BatchTopK<Input>{
          reinterpret_cast<const Input*>(logits.data_ptr()),
          scales ? scales->data_ptr() : nullptr,
          scales && scales->scalar_type() == at::kHalf,
          weights.data_ptr<float>(),
          ids.data_ptr<int32_t>(),
          tokens});
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> gemma4_batch_topk(
    const torch::Tensor& logits,
    const std::optional<torch::Tensor>& per_expert_scale,
    int64_t topk) {
  TORCH_CHECK(
      logits.is_xpu() && logits.dim() == 2 && logits.is_contiguous(),
      "gemma4_batch_topk requires contiguous 2D XPU logits");
  TORCH_CHECK(
      logits.size(0) >= 1 && logits.size(0) <= 2048 &&
          logits.size(1) == kExperts && topk == kTopK,
      "gemma4_batch_topk supports M=1..2048, E=128, topk=8");
  TORCH_CHECK(
      logits.scalar_type() == at::kHalf || logits.scalar_type() == at::kFloat,
      "gemma4_batch_topk requires FP16 or FP32 logits");
  if (per_expert_scale) {
    TORCH_CHECK(
        per_expert_scale->device() == logits.device() &&
            (per_expert_scale->scalar_type() == at::kHalf ||
             per_expert_scale->scalar_type() == at::kFloat) &&
            per_expert_scale->dim() == 1 &&
            per_expert_scale->numel() == kExperts &&
            per_expert_scale->is_contiguous(),
        "per_expert_scale must be contiguous XPU FP16/FP32 [128] on logits "
        "device");
  }
  const at::DeviceGuard guard(logits.device());
  auto weights =
      at::empty({logits.size(0), topk}, logits.options().dtype(at::kFloat));
  auto ids =
      at::empty({logits.size(0), topk}, logits.options().dtype(at::kInt));
  if (logits.scalar_type() == at::kHalf) {
    launch<sycl::half>(logits, per_expert_scale, weights, ids);
  } else {
    launch<float>(logits, per_expert_scale, weights, ids);
  }
  return {weights, ids};
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Meta, m) {
  m.impl(
      "gemma4_batch_topk",
      [](const torch::Tensor& logits,
         const std::optional<torch::Tensor>& per_expert_scale,
         int64_t topk) {
        return std::make_tuple(
            at::empty_symint(
                {logits.sym_size(0), c10::SymInt(topk)},
                logits.options().dtype(at::kFloat)),
            at::empty_symint(
                {logits.sym_size(0), c10::SymInt(topk)},
                logits.options().dtype(at::kInt)));
      });
}
