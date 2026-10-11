// SPDX-License-Identifier: Apache-2.0
#include "core/registration.h"

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {
constexpr int kHidden = 2816;
constexpr int kExperts = 128;
constexpr int kTopK = 8;

// Each token is owned by one work-group. There is no cross-group counter,
// queue-global scratch, initialization launch, or synchronization on the host.
template <int Threads, bool AddResidual>
struct Gemma4RouterFusion {
  const sycl::half* input;
  const sycl::half* residual;
  const sycl::half* post_weight;
  const sycl::half* dense_weight;
  const sycl::half* moe_weight;
  const sycl::half* router_scale;
  const void* root;
  bool root_is_float;
  const sycl::half* projection;
  const void* expert_scale;
  bool expert_scale_is_half;
  float post_epsilon;
  float dense_epsilon;
  float moe_epsilon;
  float router_epsilon;
  sycl::half* dense_output;
  sycl::half* moe_output;
  sycl::half* residual_output;
  float* logits;
  float* weights;
  int32_t* ids;
  sycl::local_accessor<sycl::half, 1> router_input;
  sycl::local_accessor<int64_t, 1> keys;
  sycl::local_accessor<float, 1> selected_exp;

  [[sycl::reqd_sub_group_size(16)]] void
  operator()(sycl::nd_item<1> item) const {
    const int lane = item.get_local_linear_id();
    const int token = item.get_group_linear_id();
    const int64_t row = int64_t(token) * kHidden;
    float inverse_post = 1.f;
    if constexpr (AddResidual) {
      float sum = 0.f;
      for (int k = lane; k < kHidden; k += Threads) {
        const float x = float(input[row + k]);
        sum += x * x;
      }
      sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
      inverse_post = sycl::rsqrt(sum / kHidden + post_epsilon);
    }
    float sum = 0.f;
    for (int k = lane; k < kHidden; k += Threads) {
      sycl::half x = input[row + k];
      if constexpr (AddResidual) {
        x = sycl::half(float(x) * inverse_post);
        x = sycl::half(float(x) * float(post_weight[k]));
        x = sycl::half(float(x) + float(residual[row + k]));
        residual_output[row + k] = x;
      }
      router_input[k] = x;
      sum += float(x) * float(x);
    }
    sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
    const float mean = sum / kHidden;
    const float inverse_moe = sycl::rsqrt(mean + moe_epsilon);
    const float inverse_router = sycl::rsqrt(mean + router_epsilon);
    const float inverse_dense = sycl::rsqrt(mean + dense_epsilon);
    const sycl::half root_value =
        root_is_float ? sycl::half(static_cast<const float*>(root)[0])
                      : static_cast<const sycl::half*>(root)[0];
    for (int k = lane; k < kHidden; k += Threads) {
      const float x = float(router_input[k]);
      const sycl::half m = sycl::half(x * inverse_moe);
      moe_output[row + k] = sycl::half(float(m) * float(moe_weight[k]));
      if constexpr (AddResidual) {
        const sycl::half d = sycl::half(x * inverse_dense);
        dense_output[row + k] = sycl::half(float(d) * float(dense_weight[k]));
      }
      const sycl::half normalized = sycl::half(x * inverse_router);
      const sycl::half rooted =
          sycl::half(float(normalized) * float(root_value));
      router_input[k] = sycl::half(float(rooted) * float(router_scale[k]));
    }
    item.barrier(sycl::access::fence_space::local_space);

    constexpr int projection_lanes = Threads / kExperts;
    const int expert = lane / projection_lanes;
    const int part = lane % projection_lanes;
    float dot = 0.f;
    for (int k = part * 8; k < kHidden; k += projection_lanes * 8) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dot += float(router_input[k + j]) *
               float(projection[int64_t(expert) * kHidden + k + j]);
      }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int distance = 1; distance < projection_lanes; distance *= 2)
      dot += sycl::permute_group_by_xor(sg, dot, distance);
    // GateLinear stores the FP16 projection before converting to FP32.
    const float value = float(sycl::half(dot));
    if (part == 0) {
      logits[token * kExperts + expert] = value;
      const uint32_t bits = sycl::bit_cast<uint32_t>(value);
      const uint32_t key =
          bits & 0x80000000u ? bits ^ 0x80000000u : bits ^ 0xffffffffu;
      keys[expert] =
          sycl::bit_cast<int64_t>((uint64_t(key) << 32) | uint64_t(expert));
    }
    const float maximum = sycl::reduce_over_group(
        item.get_group(),
        sycl::isnan(value) ? -std::numeric_limits<float>::infinity() : value,
        sycl::maximum<float>());
    item.barrier(sycl::access::fence_space::local_space);
    int rank = 0;
    if (part == 0) {
      const int64_t key = keys[expert];
      for (int other = 0; other < kExperts; ++other)
        rank += keys[other] < key;
      if (rank < kTopK)
        selected_exp[rank] =
            sycl::exp2((value - maximum) * 1.4426950408889634f);
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (part == 0 && rank < kTopK) {
      float denominator = 0.f;
#pragma unroll
      for (int i = 0; i < kTopK; ++i)
        denominator += selected_exp[i];
      denominator = denominator > 0.f ? denominator : 1.f;
      const float scale =
          expert_scale_is_half
              ? float(static_cast<const sycl::half*>(expert_scale)[expert])
              : static_cast<const float*>(expert_scale)[expert];
      weights[token * kTopK + rank] =
          selected_exp[rank] * (1.f / denominator) * scale;
      ids[token * kTopK + rank] = expert;
    }
  }
};

using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
using ResidualOutputs = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

void check_half(const at::Tensor& x, const at::Tensor& input) {
  TORCH_CHECK(
      x.device() == input.device() && x.scalar_type() == at::kHalf &&
          x.is_contiguous(),
      "Expected contiguous FP16 tensors on the input XPU device");
}

void check_router(
    const at::Tensor& input,
    const at::Tensor& router_scale,
    const at::Tensor& root,
    const at::Tensor& projection,
    const at::Tensor& moe_weight,
    const at::Tensor& expert_scale,
    double router_epsilon,
    double moe_epsilon) {
  TORCH_CHECK(input.is_xpu(), "Expected XPU input");
  check_half(input, input);
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 8 &&
          input.size(1) == kHidden,
      "Gemma Router fusion requires [M,2816], M=1..8");
  for (const auto& w : {router_scale, moe_weight}) {
    check_half(w, input);
    TORCH_CHECK(
        w.sizes() == at::IntArrayRef({kHidden}), "Expected weight [2816]");
  }
  check_half(projection, input);
  TORCH_CHECK(
      projection.sizes() == at::IntArrayRef({kExperts, kHidden}),
      "Expected Router projection [128,2816]");
  TORCH_CHECK(
      root.device() == input.device() && root.numel() == 1 && root.dim() <= 2 &&
          root.is_contiguous() &&
          (root.scalar_type() == at::kHalf || root.scalar_type() == at::kFloat),
      "Expected contiguous FP16/FP32 root scalar");
  TORCH_CHECK(
      expert_scale.device() == input.device() &&
          expert_scale.sizes() == at::IntArrayRef({kExperts}) &&
          expert_scale.is_contiguous() &&
          (expert_scale.scalar_type() == at::kHalf ||
           expert_scale.scalar_type() == at::kFloat),
      "Expected contiguous FP16/FP32 expert scale [128]");
  TORCH_CHECK(
      router_epsilon >= 0 && moe_epsilon >= 0,
      "Expected nonnegative norm epsilons");
}

template <int Threads, bool AddResidual>
void launch(
    sycl::queue& queue,
    Gemma4RouterFusion<Threads, AddResidual> kernel,
    int m) {
  queue.submit([&](sycl::handler& h) {
    kernel.router_input = sycl::local_accessor<sycl::half, 1>({kHidden}, h);
    kernel.keys = sycl::local_accessor<int64_t, 1>({kExperts}, h);
    kernel.selected_exp = sycl::local_accessor<float, 1>({kTopK}, h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(m * Threads), sycl::range<1>(Threads)),
        kernel);
  });
}

template <bool AddResidual>
ResidualOutputs router(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& post_weight,
    const at::Tensor& dense_weight,
    const at::Tensor& router_scale,
    const at::Tensor& root,
    const at::Tensor& projection,
    const at::Tensor& moe_weight,
    const at::Tensor& expert_scale,
    double post_epsilon,
    double dense_epsilon,
    double router_epsilon,
    double moe_epsilon) {
  check_router(
      input,
      router_scale,
      root,
      projection,
      moe_weight,
      expert_scale,
      router_epsilon,
      moe_epsilon);
  if constexpr (AddResidual) {
    check_half(residual, input);
    TORCH_CHECK(residual.sizes() == input.sizes(), "Residual shape mismatch");
    for (const auto& w : {post_weight, dense_weight}) {
      check_half(w, input);
      TORCH_CHECK(
          w.sizes() == at::IntArrayRef({kHidden}), "Expected norm [2816]");
    }
    TORCH_CHECK(
        post_epsilon >= 0 && dense_epsilon >= 0, "Negative norm epsilon");
  }
  const at::DeviceGuard guard(input.device());
  const int m = input.size(0);
  auto dense =
      AddResidual ? at::empty_like(input) : at::empty({0}, input.options());
  auto added =
      AddResidual ? at::empty_like(input) : at::empty({0}, input.options());
  auto moe = at::empty_like(input);
  auto logits = at::empty({m, kExperts}, input.options().dtype(at::kFloat));
  auto weights = at::empty({m, kTopK}, input.options().dtype(at::kFloat));
  auto ids = at::empty({m, kTopK}, input.options().dtype(at::kInt));
  auto half = [](const at::Tensor& t) {
    return reinterpret_cast<sycl::half*>(t.data_ptr());
  };
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  auto dispatch = [&](auto threads_tag) {
    constexpr int Threads = decltype(threads_tag)::value;
    launch<Threads, AddResidual>(
        queue,
        Gemma4RouterFusion<Threads, AddResidual>{
            half(input),
            AddResidual ? half(residual) : nullptr,
            AddResidual ? half(post_weight) : nullptr,
            AddResidual ? half(dense_weight) : nullptr,
            half(moe_weight),
            half(router_scale),
            root.data_ptr(),
            root.scalar_type() == at::kFloat,
            half(projection),
            expert_scale.data_ptr(),
            expert_scale.scalar_type() == at::kHalf,
            float(post_epsilon),
            float(dense_epsilon),
            float(moe_epsilon),
            float(router_epsilon),
            half(dense),
            half(moe),
            half(added),
            logits.data_ptr<float>(),
            weights.data_ptr<float>(),
            ids.data_ptr<int32_t>(),
            {},
            {},
            {}},
        m);
  };
  const size_t max_threads =
      queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (max_threads >= 512)
    dispatch(std::integral_constant<int, 512>{});
  else if (max_threads >= 256)
    dispatch(std::integral_constant<int, 256>{});
  else
    dispatch(std::integral_constant<int, 128>{});
  return {dense, moe, logits, weights, ids, added};
}

Outputs plain_router(
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& root,
    const at::Tensor& projection,
    const at::Tensor& moe_weight,
    const at::Tensor& expert_scale,
    double router_epsilon,
    double moe_epsilon) {
  auto [dense, moe, logits, weights, ids, added] = router<false>(
      x,
      {},
      {},
      {},
      scale,
      root,
      projection,
      moe_weight,
      expert_scale,
      0,
      0,
      router_epsilon,
      moe_epsilon);
  return {logits, moe, weights, ids};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_xpu_C, m) {
  m.def(
      "gemma4_norm_router_topk(Tensor input, Tensor router_scale, Tensor root, "
      "Tensor projection, Tensor moe_weight, Tensor expert_scale, "
      "float router_epsilon, float moe_epsilon) -> (Tensor, Tensor, Tensor, "
      "Tensor)");
  m.def(
      "gemma4_residual_norm_router_topk(Tensor input, Tensor residual, "
      "Tensor post_weight, Tensor dense_weight, Tensor router_scale, Tensor "
      "root, "
      "Tensor projection, Tensor moe_weight, Tensor expert_scale, "
      "float post_epsilon, float dense_epsilon, float router_epsilon, "
      "float moe_epsilon) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(_xpu_C, XPU, m) {
  m.impl("gemma4_norm_router_topk", &plain_router);
  m.impl("gemma4_residual_norm_router_topk", &router<true>);
}
TORCH_LIBRARY_IMPL(_xpu_C, Meta, m) {
  m.impl(
      "gemma4_norm_router_topk",
      [](const at::Tensor& x,
         const at::Tensor& scale,
         const at::Tensor& root,
         const at::Tensor& projection,
         const at::Tensor& moe_weight,
         const at::Tensor& expert_scale,
         double e,
         double me) {
        return std::make_tuple(
            at::empty_symint(
                {x.sym_size(0), kExperts}, x.options().dtype(at::kFloat)),
            at::empty_like(x),
            at::empty_symint(
                {x.sym_size(0), kTopK}, x.options().dtype(at::kFloat)),
            at::empty_symint(
                {x.sym_size(0), kTopK}, x.options().dtype(at::kInt)));
      });
  m.impl(
      "gemma4_residual_norm_router_topk",
      [](const at::Tensor& x,
         const at::Tensor& residual,
         const at::Tensor& post_weight,
         const at::Tensor& dense_weight,
         const at::Tensor& scale,
         const at::Tensor& root,
         const at::Tensor& projection,
         const at::Tensor& moe_weight,
         const at::Tensor& expert_scale,
         double pe,
         double de,
         double re,
         double me) {
        return std::make_tuple(
            at::empty_like(x),
            at::empty_like(x),
            at::empty_symint(
                {x.sym_size(0), kExperts}, x.options().dtype(at::kFloat)),
            at::empty_symint(
                {x.sym_size(0), kTopK}, x.options().dtype(at::kFloat)),
            at::empty_symint(
                {x.sym_size(0), kTopK}, x.options().dtype(at::kInt)),
            at::empty_like(x));
      });
}
