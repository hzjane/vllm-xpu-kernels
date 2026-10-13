// SPDX-License-Identifier: Apache-2.0
#include "gemma_moe_fused_interface.h"
#include "csrc/utils.h"
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
