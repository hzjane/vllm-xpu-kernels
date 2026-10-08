// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <c10/core/Stream.h>
#include <torch/types.h>

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "qwen38/tensor_binding.h"

namespace vllm::qwen38 {

// One host transaction, not one device kernel. Reuses the existing three
// projection/core/projection implementations without changing their math.
class GdnSyclM1Workspace final {
 public:
  // weights: qkv W/S, ba W/S, out W/S, conv W, A_log, dt_bias, norm W.
  // Python guards exact tensor types/modes and layer/metadata semantics.
  std::optional<at::Tensor> try_run(
      at::Tensor input,
      const std::vector<at::Tensor>& weights,
      const std::optional<at::Tensor>& conv_bias,
      at::Tensor conv_state,
      at::Tensor ssm_state,
      at::Tensor indices,
      int64_t local_v_heads,
      double eps,
      bool sigmoid_gate);

  // Required by the model's explicit inference-weight reload hook.
  void invalidate();

 private:
  struct Scratch {
    int64_t hv = 0;
    at::Tensor qkvz, ba, core, z, core_rows, z_rows, output;
    at::Tensor conv_weight, conv_bias, a_log, dt_bias, norm_weight;
    std::unique_ptr<TensorBindingSnapshotV1> binding;
  };
  std::mutex mutex_;
  std::unordered_map<c10::Stream, Scratch> by_stream_;
};

}  // namespace vllm::qwen38
