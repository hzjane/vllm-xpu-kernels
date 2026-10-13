// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <torch/all.h>

void gemma_moe_fused_decode_xe2(
    torch::Tensor& output,
    const torch::Tensor& x,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const torch::Tensor& w13,
    const torch::Tensor& s13,
    const torch::Tensor& w2,
    const torch::Tensor& s2,
    torch::Tensor& workspace);
