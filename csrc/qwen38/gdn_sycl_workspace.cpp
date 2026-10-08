// SPDX-License-Identifier: Apache-2.0
#include "qwen38/gdn_sycl_workspace.h"

#include <c10/core/DeviceGuard.h>
#include <c10/core/GradMode.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <cmath>
#include <limits>

#include "qwen38/gdn_sycl.h"
#include "qwen38/gdn_sycl_projection.h"
#include "qwen38/ops.h"

namespace vllm::qwen38 {
namespace {

bool plain(const at::Tensor& t, at::Device device, at::ScalarType dtype) {
  return t.defined() && t.device() == device && t.layout() == at::kStrided &&
         t.scalar_type() == dtype && !t.is_conj() && !t.is_neg();
}

bool dense(
    const at::Tensor& t,
    at::Device device,
    at::ScalarType dtype,
    at::IntArrayRef shape) {
  return plain(t, device, dtype) && t.sizes() == shape && t.is_contiguous();
}

bool convertible(
    const at::Tensor& t, at::Device device, at::IntArrayRef shape) {
  return t.defined() &&
         (t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16 ||
          t.scalar_type() == at::kFloat) &&
         dense(t, device, t.scalar_type(), shape);
}

at::Tensor half_parameter(const at::Tensor& t) {
  return t.detach().to(at::kHalf).contiguous();
}

}  // namespace

std::optional<at::Tensor> GdnSyclM1Workspace::try_run(
    at::Tensor input,
    const std::vector<at::Tensor>& w,
    const std::optional<at::Tensor>& bias,
    at::Tensor conv,
    at::Tensor ssm,
    at::Tensor indices,
    int64_t hv,
    double eps,
    bool sigmoid_gate) {
  if (at::GradMode::is_enabled() || !input.defined() || !input.is_xpu() ||
      w.size() != 10 || (hv != 6 && hv != 12))
    return std::nullopt;
  const auto device = input.device();
  const int64_t h = hv / 3, dim = (2 * h + hv) * 128;
  const int64_t qkvz_width = dim + hv * 128;
  const float epsilon = static_cast<float>(eps);
  if (!std::isfinite(eps) || !std::isfinite(epsilon) || epsilon <= 0.0f ||
      !dense(input, device, at::kHalf, {1, 2560}) ||
      !dense(w[0], device, at::kByte, {qkvz_width, 1280}) ||
      !dense(w[1], device, at::kHalf, {qkvz_width, 20}) ||
      !dense(w[2], device, at::kByte, {2 * hv, 1280}) ||
      !dense(w[3], device, at::kHalf, {2 * hv, 20}) ||
      !dense(w[4], device, at::kByte, {2560, hv * 64}) ||
      !dense(w[5], device, at::kHalf, {2560, hv}) ||
      !convertible(w[6], device, {dim, 1, 4}) ||
      !convertible(w[7], device, {hv}) || !convertible(w[8], device, {hv}) ||
      !convertible(w[9], device, {128}) ||
      (bias.has_value() && !convertible(*bias, device, {dim})) ||
      !dense(indices, device, at::kInt, {1}) ||
      !plain(conv, device, at::kHalf) || !plain(ssm, device, at::kHalf) ||
      conv.dim() != 3 || conv.size(0) <= 0 ||
      conv.size(0) > std::numeric_limits<int>::max() || conv.size(1) != 3 ||
      conv.size(2) != dim || conv.stride(2) != 1 || conv.stride(1) != dim ||
      conv.stride(0) < 3 * dim || ssm.dim() != 4 || ssm.size(0) <= 0 ||
      ssm.size(0) > std::numeric_limits<int>::max() || ssm.size(1) != hv ||
      ssm.size(2) != 128 || ssm.size(3) != 128 || ssm.stride(3) != 1 ||
      ssm.stride(2) != 128 || ssm.stride(1) != 128 * 128 ||
      ssm.stride(0) < hv * 128 * 128)
    return std::nullopt;

  // Raw sources must not be overwritten even when the derived FP16 copy
  // happens to be disjoint. This preserves the Python guard's live contract.
  std::vector<std::optional<at::Tensor>> reads(w.begin(), w.end());
  reads.emplace_back(input);
  reads.emplace_back(indices);
  reads.emplace_back(bias);
  if (!tensors_disjoint_host({conv, ssm}, reads)) return std::nullopt;

  const c10::OptionalDeviceGuard device_guard(device);
  auto stream = c10::xpu::getCurrentXPUStream(device.index());
  if (!stream.queue().is_in_order()) return std::nullopt;
  std::lock_guard<std::mutex> lock(mutex_);
  auto& s = by_stream_[stream.unwrap()];
  if (!s.output.defined() || s.hv != hv) {
    s = Scratch{};
    s.hv = hv;
    const auto options = input.options().requires_grad(false);
    s.qkvz = at::empty({1, qkvz_width}, options);
    s.ba = at::empty({1, 2 * hv}, options);
    s.core = at::empty({1, hv, 128}, options);
    s.z = at::empty({1, hv, 128}, options);
    s.core_rows = s.core.view({hv, 128});
    s.z_rows = s.z.view({hv, 128});
    s.output = at::empty({1, 2560}, options);
  }
  std::vector<at::Tensor> writes = {
      conv, ssm, s.qkvz, s.ba, s.core, s.z, s.output};
  if (!tensors_disjoint_host(writes, reads)) return std::nullopt;

  std::vector<at::Tensor> sources = {w[6], w[7], w[8], w[9]};
  if (bias.has_value()) sources.push_back(*bias);
  const bool refresh = !s.binding || !s.binding->matches(sources);
  if (!refresh) {
    // Check cached derived buffers too, before the first device submission.
    auto derived_reads = reads;
    for (const auto* t :
         {&s.conv_weight, &s.conv_bias, &s.a_log, &s.dt_bias, &s.norm_weight})
      derived_reads.emplace_back(*t);
    if (!tensors_disjoint_host(writes, derived_reads)) return std::nullopt;
  }

  // First possible submission below (parameter conversion/zeroing). Never
  // catch/replay anything after this boundary, including partial failures.
  if (refresh) {
    // A failed conversion must not leave the old witness attached to a
    // partially refreshed set of derived tensors. Any later reuse rebuilds
    // the whole set, even if the caller has restored the previous weights.
    s.binding.reset();
    auto binding = std::make_unique<TensorBindingSnapshotV1>(sources);
    s.conv_weight = half_parameter(w[6]).view({dim, 4});
    s.conv_bias = bias.has_value()
                      ? half_parameter(*bias)
                      : at::zeros({dim}, input.options().requires_grad(false));
    s.a_log = half_parameter(w[7]);
    s.dt_bias = half_parameter(w[8]);
    s.norm_weight = half_parameter(w[9]);
    s.binding = std::move(binding);
  }
  int4_linear_fused2(input, w[0], w[1], s.qkvz, w[2], w[3], s.ba);
  gdn_decode_sycl(
      s.qkvz,
      conv,
      s.conv_weight,
      s.conv_bias,
      indices,
      s.a_log,
      s.dt_bias,
      s.ba,
      ssm,
      indices,
      s.core,
      s.z,
      1.0 / std::sqrt(128.0));
  gdn_norm_int4_sycl(
      s.core_rows,
      s.z_rows,
      s.norm_weight,
      w[4],
      w[5],
      s.output,
      hv,
      128,
      eps,
      sigmoid_gate);
  // A fresh TensorImpl prevents consumer metadata edits from changing the
  // private scratch layout; backing storage follows the current stream.
  return s.output.view({1, 2560});
}

void GdnSyclM1Workspace::invalidate() {
  std::lock_guard<std::mutex> lock(mutex_);
  by_stream_.clear();
}

}  // namespace vllm::qwen38
