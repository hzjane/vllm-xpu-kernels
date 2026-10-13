// SPDX-License-Identifier: Apache-2.0
#include "gemma_moe_fused_interface.h"
#include "csrc/utils.h"
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <limits>
#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/gemma_moe_fused_decode_xe2.h"
#endif

void gemma_moe_fused_decode_interface(
    torch::Tensor& output,
    const torch::Tensor& x,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const torch::Tensor& w13,
    const torch::Tensor& s13,
    const torch::Tensor& w2,
    const torch::Tensor& s2,
    torch::Tensor& workspace) {
  TORCH_CHECK(x.is_xpu(), "hidden states must be an XPU tensor");
  const auto device = x.get_device();
  TORCH_CHECK(
      vllm::xpu::is_xe2_arch(device) || vllm::xpu::is_xe3_arch(device),
      "Gemma fused MoE decode requires Xe2/Xe3");
#ifdef VLLM_XPU_ENABLE_XE2
  gemma_moe_fused_decode_xe2(
      output, x, topk_weights, topk_ids, w13, s13, w2, s2, workspace);
#else
  TORCH_CHECK(false, "Xe2 kernels are disabled in this build");
#endif
}

namespace {

constexpr int kExperts = 128;
constexpr int kTopK = 8;

template <typename Input>
struct Gemma4SmallMTopK {
  const Input* logits;
  const void* expert_scales;
  at::ScalarType scale_type;
  float* weights;
  int32_t* ids;
  sycl::local_accessor<int64_t, 1> ordered_keys;
  sycl::local_accessor<float, 1> selected_exp;

  [[sycl::reqd_sub_group_size(16)]] void
  operator()(sycl::nd_item<1> item) const {
    const int expert = item.get_local_linear_id();
    const int token = item.get_group_linear_id();
    const float value = static_cast<float>(logits[token * kExperts + expert]);

    // The model's Triton router sorts this same signed 64-bit key. It orders
    // logits descending and resolves exact ties by ascending expert ID;
    // the float-bit ordering also preserves its +0/-0 behavior.
    const uint32_t bits = sycl::bit_cast<uint32_t>(value);
    const uint32_t key =
        (bits & 0x80000000u) ? (bits ^ 0x80000000u) : (bits ^ 0xffffffffu);
    const int64_t ordered = sycl::bit_cast<int64_t>(
        (static_cast<uint64_t>(key) << 32) | static_cast<uint64_t>(expert));
    ordered_keys[expert] = ordered;
    // Match Triton's maximum: NaN logits do not poison finite lanes.
    // Preserve the original value for sorting and the selected exponential.
    const float maximum = sycl::reduce_over_group(
        item.get_group(),
        sycl::isnan(value) ? -std::numeric_limits<float>::infinity() : value,
        sycl::maximum<float>());
    item.barrier(sycl::access::fence_space::local_space);

    int rank = 0;
    for (int other = 0; other < kExperts; ++other) {
      rank += ordered_keys[other] < ordered;
    }
    if (rank < kTopK) {
      selected_exp[rank] = sycl::exp2((value - maximum) * 1.4426950408889634f);
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (rank < kTopK) {
      float sum = 0.0f;
#pragma unroll
      for (int selected = 0; selected < kTopK; ++selected) {
        sum += selected_exp[selected];
      }
      sum = sum > 0.0f ? sum : 1.0f;
      const float scale =
          scale_type == at::kHalf
              ? static_cast<float>(
                    static_cast<const sycl::half*>(expert_scales)[expert])
          : scale_type == at::kBFloat16
              ? static_cast<float>(
                    static_cast<const sycl::ext::oneapi::bfloat16*>(
                        expert_scales)[expert])
              : static_cast<const float*>(expert_scales)[expert];
      weights[token * kTopK + rank] = selected_exp[rank] * (1.0f / sum) * scale;
      ids[token * kTopK + rank] = expert;
    }
  }
};

template <typename Input>
void launch_topk(
    const torch::Tensor& logits,
    const torch::Tensor& expert_scales,
    torch::Tensor& weights,
    torch::Tensor& ids) {
  auto& queue = c10::xpu::getCurrentXPUStream(logits.get_device()).queue();
  const size_t tokens = logits.size(0);
  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<int64_t, 1> keys(sycl::range<1>(kExperts), handler);
    sycl::local_accessor<float, 1> selected_exp(sycl::range<1>(kTopK), handler);
    handler.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(tokens * kExperts), sycl::range<1>(kExperts)),
        Gemma4SmallMTopK<Input>{
            reinterpret_cast<const Input*>(logits.data_ptr()),
            expert_scales.data_ptr(),
            expert_scales.scalar_type(),
            weights.data_ptr<float>(),
            ids.data_ptr<int32_t>(),
            keys,
            selected_exp});
  });
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> gemma_moe_topk_interface(
    const torch::Tensor& logits,
    const torch::Tensor& per_expert_scale,
    int64_t topk) {
  TORCH_CHECK(
      logits.is_xpu() && logits.dim() == 2 && logits.is_contiguous(),
      "gemma_moe_topk_interface requires contiguous 2D XPU logits");
  TORCH_CHECK(
      logits.size(0) >= 1 && logits.size(0) <= 8 &&
          logits.size(1) == kExperts && topk == kTopK,
      "gemma_moe_topk_interface supports M=1..8, E=128 and topk=8");
  TORCH_CHECK(
      logits.scalar_type() == at::kHalf ||
          logits.scalar_type() == at::kBFloat16 ||
          logits.scalar_type() == at::kFloat,
      "gemma_moe_topk_interface requires FP16/BF16/FP32 logits");
  {
    TORCH_CHECK(
        per_expert_scale.device() == logits.device() &&
            per_expert_scale.is_xpu() &&
            (per_expert_scale.scalar_type() == at::kBFloat16 ||
             per_expert_scale.scalar_type() == at::kFloat ||
             per_expert_scale.scalar_type() == at::kHalf) &&
            per_expert_scale.dim() == 1 &&
            per_expert_scale.numel() == kExperts &&
            per_expert_scale.is_contiguous(),
        "per_expert_scale must be contiguous XPU FP16/BF16/FP32 [128] on the "
        "logits "
        "device");
  }
  const c10::DeviceGuard guard(logits.device());
  auto weights =
      at::empty({logits.size(0), topk}, logits.options().dtype(at::kFloat));
  auto ids =
      at::empty({logits.size(0), topk}, logits.options().dtype(at::kInt));
  if (logits.scalar_type() == at::kHalf) {
    launch_topk<sycl::half>(logits, per_expert_scale, weights, ids);
  } else if (logits.scalar_type() == at::kBFloat16) {
    launch_topk<sycl::ext::oneapi::bfloat16>(
        logits, per_expert_scale, weights, ids);
  } else {
    launch_topk<float>(logits, per_expert_scale, weights, ids);
  }
  return {weights, ids};
}
