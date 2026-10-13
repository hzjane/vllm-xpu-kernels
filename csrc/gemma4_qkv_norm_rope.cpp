// SPDX-License-Identifier: Apache-2.0
#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>

namespace {
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

Outputs outputs(const at::Tensor& qkv, int64_t qh, int64_t kh, int64_t d) {
  return {
      at::empty_symint({qkv.sym_size(0), c10::SymInt(qh * d)}, qkv.options()),
      at::empty_symint({qkv.sym_size(0), c10::SymInt(kh * d)}, qkv.options()),
      at::empty_symint({qkv.sym_size(0), c10::SymInt(kh * d)}, qkv.options())};
}

Outputs qkv_norm_rope(
    const at::Tensor& qkv,
    const at::Tensor& positions,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& cache,
    int64_t qh,
    int64_t kh,
    int64_t d,
    double epsilon,
    bool native_rope) {
  const bool geometry =
      (d == 256 && ((qh == 8 && kh == 4) || (qh == 16 && kh == 8))) ||
      (d == 512 && ((qh == 8 && kh == 1) || (qh == 16 && kh == 2)));
  TORCH_CHECK(
      qkv.is_xpu() && qkv.scalar_type() == at::kHalf && qkv.dim() == 2 &&
          qkv.is_contiguous() && qkv.size(0) >= 1 && qkv.size(0) <= 8 &&
          geometry && qkv.size(1) == (qh + 2 * kh) * d && epsilon > 0,
      "gemma4_qkv_norm_rope requires contiguous FP16 Gemma TP2 QKV, M=1..8");
  TORCH_CHECK(
      positions.device() == qkv.device() && positions.dim() == 1 &&
          positions.is_contiguous() && positions.scalar_type() == at::kLong &&
          positions.size(0) == qkv.size(0),
      "invalid positions");
  for (const auto& weight : {q_weight, k_weight}) {
    TORCH_CHECK(
        weight.device() == qkv.device() && weight.scalar_type() == at::kHalf &&
            weight.dim() == 1 && weight.numel() == d && weight.is_contiguous(),
        "invalid Q/K RMSNorm weight");
  }
  TORCH_CHECK(
      cache.device() == qkv.device() && cache.scalar_type() == at::kHalf &&
          cache.dim() == 2 && cache.is_contiguous() &&
          (cache.size(1) == d || (d == 512 && cache.size(1) == 128)),
      "invalid full/partial NeoX RoPE cache");
  const at::DeviceGuard guard(qkv.device());
  auto [q, k, v] = outputs(qkv, qh, kh, d);
  const auto* src =
      reinterpret_cast<const sycl::half*>(qkv.data_ptr<at::Half>());
  const auto* qw =
      reinterpret_cast<const sycl::half*>(q_weight.data_ptr<at::Half>());
  const auto* kw =
      reinterpret_cast<const sycl::half*>(k_weight.data_ptr<at::Half>());
  const auto* cs =
      reinterpret_cast<const sycl::half*>(cache.data_ptr<at::Half>());
  const auto* pos = positions.data_ptr<int64_t>();
  auto* oq = reinterpret_cast<sycl::half*>(q.data_ptr<at::Half>());
  auto* ok = reinterpret_cast<sycl::half*>(k.data_ptr<at::Half>());
  auto* ov = reinterpret_cast<sycl::half*>(v.data_ptr<at::Half>());
  const int heads = qh + 2 * kh;
  const int rotary = cache.size(1);
  const float eps = epsilon;
  auto& queue = c10::xpu::getCurrentXPUStream(qkv.get_device()).queue();
  queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<sycl::half, 1> norm(sycl::range<1>(d), h);
    h.parallel_for(
        sycl::nd_range<1>(qkv.size(0) * heads * 128, 128),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const int group = item.get_group_linear_id();
          const int token = group / heads;
          const int head = group % heads;
          const int tid = item.get_local_linear_id();
          const int64_t offset = (int64_t(token) * heads + head) * d;
          const bool is_q = head < qh;
          const bool is_v = head >= qh + kh;
          const auto* weight = is_q ? qw : kw;
          float sum = 0;
          for (int j = tid; j < d; j += 128) {
            float value = static_cast<float>(src[offset + j]);
            sum += value * value;
          }
          sum = sycl::reduce_over_group(
              item.get_group(), sum, sycl::plus<float>());
          float inv = sycl::rsqrt(sum / d + eps);
          for (int j = tid; j < d; j += 128) {
            sycl::half normalized =
                sycl::half(static_cast<float>(src[offset + j]) * inv);
            norm[j] = is_v ? normalized
                           : sycl::half(
                                 static_cast<float>(normalized) *
                                 static_cast<float>(weight[j]));
          }
          item.barrier(sycl::access::fence_space::local_space);
          const int local_head =
              is_q ? head : (is_v ? head - qh - kh : head - qh);
          auto* dest = is_q ? oq : (is_v ? ov : ok);
          const int64_t dest_offset =
              (int64_t(token) * (is_q ? qh : kh) + local_head) * d;
          for (int j = tid; j < d; j += 128) {
            float result = static_cast<float>(norm[j]);
            if (!is_v && j < rotary) {
              const int pair = j % (rotary / 2);
              const float c =
                  static_cast<float>(cs[pos[token] * rotary + pair]);
              const float s = static_cast<float>(
                  cs[pos[token] * rotary + pair + rotary / 2]);
              const float left = static_cast<float>(norm[pair]);
              const float right = static_cast<float>(norm[pair + rotary / 2]);
              // Match native FP16 RoPE's product rounding before add/sub.
              const float lc = static_cast<float>(sycl::half(left * c));
              const float rs = static_cast<float>(sycl::half(right * s));
              const float rc = static_cast<float>(sycl::half(right * c));
              const float ls = static_cast<float>(sycl::half(left * s));
              result = native_rope ? (j < rotary / 2 ? lc - rs : rc + ls)
                                   : (j < rotary / 2 ? left * c - right * s
                                                     : right * c + left * s);
            }
            dest[dest_offset + j] = sycl::half(result);
          }
        });
  });
  return {q, k, v};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_xpu_C, m) {
  m.def(
      "gemma4_qkv_norm_rope(Tensor qkv, Tensor positions, Tensor q_weight, "
      "Tensor k_weight, Tensor cache, int q_heads, int kv_heads, int head_dim, "
      "float epsilon, bool native_rope=True) -> (Tensor, Tensor, Tensor)");
  m.impl("gemma4_qkv_norm_rope", torch::kXPU, &qkv_norm_rope);
  m.impl(
      "gemma4_qkv_norm_rope",
      torch::kMeta,
      [](const at::Tensor& qkv,
         const at::Tensor& positions,
         const at::Tensor& qw,
         const at::Tensor& kw,
         const at::Tensor& cache,
         int64_t qh,
         int64_t kh,
         int64_t d,
         double epsilon,
         bool native_rope) { return outputs(qkv, qh, kh, d); });
}
