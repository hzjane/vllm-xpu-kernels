# Gemma 优化接口清单

本清单登记相对 upstream 新增或改变的接口，不是仓库全部算子列表。
2026-09-22 精简以两个仓库的 `upstreamable_gemma` 为准；完整历史实现
保存在 `upstreamable_gemma_full_20260922`。本轮取舍依据为 Gemma-4-26B-A4B
和 Gemma-4-31B 的 AR batch1 整模型测量，不能据此推断 MTP 性能。

## 保留的接口

| 接口 | 模块 / 实现 | 适用范围与回退 |
| --- | --- | --- |
| `_xpu_C.fp8_moe_decode` | 可选 `_moe_decode_C`；`csrc/xpu/moe_decode/` | BMG-G31、FP16 activation、FP8 per-expert scale、E128/topk8/H2816/I352、M1..8。现有 experts 调用边界内融合，不含 routing。无 bias、EP、expert map、激活量化或 clamp；不支持返回 false，调用原 experts。保留分步 FP16 舍入、当前 stream、Meta fallback。 |
| `_moe_C.gemma4_small_m_topk` | `csrc/moe/gemma4_small_m_topk.cpp` | Gemma top8、128 experts、M1..8；输出 FP32 weights、int32 IDs。由 vLLM XPU routing 分派调用，不支持条件继续 Triton。 |
| `_moe_C.gemma4_batch_topk` | `csrc/moe/gemma4_batch_topk.cpp` | 同一 routing 契约的批量实现，原生支持 M1..2048；vLLM 在 M9..2048 使用。Prefill/此前 Diffusion 工作保留，本轮不测 Diffusion。 |
| `_xpu_C.unquantized_gemm` | 可选 `_fp16_C`；`csrc/xpu/fp16_linear/` | **仅** BMG-G31、FP16、M1..8、N128/K2816、连续且 64B 对齐、无 bias 使用 Router 的 SYCL-TLA 实现；其余直接 `at::linear`。不再替换主模型 LM head、draft Linear 或 FP8 Linear。 |
| `_xpu_C.rms_norm_small_m` / `.out` | 可选 `_rms_C`；`csrc/xpu/rms/small_m_rms.cpp` | FP16、H256/512/2816/5376、rank2..4、M1..8、总 rows≤128，允许外维 stride 和未对齐输入，支持可选 weight。旧 `_C.rms_norm` 内按 dtype/shape/stride/alias 条件分派，缺模块或不适用时保留原实现。 |
| `_xpu_C.gelu_tanh_and_mul_small_m` | 可选 `_decode_aux_C`；`csrc/xpu/decode_aux/small_m_aux.cpp` | FP16、D352/1056、rows≤64，保留 GELU FP16 中间舍入。由旧 `_C.gelu_tanh_and_mul` guarded 分派。 |
| `_xpu_C.try_rotary_embedding_small_m` | 同上 | FP16、M1..8、D256/512、NeoX RoPE、partial rotary、外维 stride；原地 Q/K。在提交前检查并对不适用输入返回 false。仅通过旧 `_C.rotary_embedding` 接入。 |

所有保留的独立算子使用当前 XPU stream，有对应 Meta 实现，不保存跨调用 scratch。
可选扩展缺失时使用原入口；扩展存在但 ABI 损坏时加载错误正常传播。
具体边界以源代码检查为准，服务实际命中以 trace 为准。

## 既有 ABI 内的优化

- `_C.gelu_tanh_and_mul`：FP16 M1..8、D4096 使用列分块，其他形状保留原实现或上述辅助路径。本轮主模型 AR 不命中 D4096，不能归因于此次 AR 收益。
- `_C.reshape_and_cache_flash` 的 strided 写入：仅 FP16、非量化 KV、D512、KV heads=2、tokens1..8 使用 256 GRF，减少后续 attention 前的 GRF 模式切换。
- Flash attention：小 query FP16 paged 路径、滑窗 Q64/全局 Q128 调优，以及 chunk prefill 配置。具体条件见 `KERNEL_CONFIGURATION.md`。
- 通用 grouped GEMM 调度已恢复 upstream；Gemma fused experts 独立检查自身适用条件。

## 本轮移除

- 全部 FP8 Linear 替换、`_linear_C` 扩展、相关构建选项和专用测试。
- 主模型 LM head 与 MTP/draft 的 FP16 Linear 特化、直接 `fp16_nt_gemm` API；FP16 模块缩减为上述 Router 和原生 Linear 回退。
- 旧 FP8 ABI 内的 TLA 转接、vLLM FP8 分派；FP8 Linear 使用 upstream 原路径。
- vLLM GELU/RoPE 的额外 Python 包装，以及失去调用方的 GELU 分配式包装、直接 RoPE schema；保留原 `_C` 入口下面的优化。
- 未调用的 `_decode_C` 五种模型融合、独立 `rms_norm_no_weight`、重复的 weightless RMS 源码。
- 被 fused experts 覆盖的 compact MoE tile map、prepare/gather 辅助接口及相关 Python 包装。

## Diffusion 启用代码

`flash_attn_varlen_func` / `_vllm_fa2_C.varlen_fwd` 的可选 `per_seq_causal`
以及配套 mask 实现仍在备份和当前分支。本轮不测试、不改动该功能。与 upstream
PR606 的 `dynamic_causal` 有功能重叠；最终合并时须基于 upstream 接口整理，
不能把当前分支直接称为已消除重复的最终 PR。

## 验证依据

源码、库哈希、独立算子 A/B、服务 E2E、acc5 与 trace 均记录于
`gemma_reduction_20260922` 实验目录。`no_linear` 是全部 Linear 恢复原实现的
消融版；`minimal_final` 是只保留 Router 的最终精简候选，两者不得混称。
旧版本或单算子收益不代替精简后的整模型结果。
