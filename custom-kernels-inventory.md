# Gemma 优化接口清单

本清单登记相对 upstream 新增或改变的接口，不是仓库全部算子列表。
2026-09-22 精简以两个仓库的 `upstreamable_gemma` 为准；完整历史实现
保存在 `upstreamable_gemma_full_20260922`。此前 Linear 取舍依据为两种 Gemma
的 AR batch1 整模型测量，不能据此推断 MTP 性能；本次进一步移除全部新增
Diffusion 专用代码，仅为 Gemma26B 复核服务，Gemma31B 验证 kernel 回归。

## 保留的接口

| 接口 | 模块 / 实现 | 适用范围与回退 |
| --- | --- | --- |
| `_xpu_C.fp8_moe_decode` | 可选 `_moe_decode_C`；`csrc/xpu/moe_decode/` | BMG-G31、FP16 activation、FP8 per-expert scale、E128/topk8/H2816/I352、M1..8。现有 experts 调用边界内融合，不含 routing。无 bias、EP、expert map、激活量化或 clamp；不支持返回 false，调用原 experts。保留分步 FP16 舍入、当前 stream、Meta fallback。 |
| `_moe_C.gemma4_small_m_topk` | `csrc/moe/gemma4_small_m_topk.cpp` | Gemma top8、128 experts、M1..8；输出 FP32 weights、int32 IDs。由 vLLM XPU routing 分派调用，不支持条件继续 Triton。 |
| `_xpu_C.unquantized_gemm` | 可选 `_fp16_C`；`csrc/xpu/fp16_linear/` | **仅** BMG-G31、FP16、M1..8、N128/K2816、连续且 64B 对齐、无 bias 使用 Router 的 SYCL-TLA 实现；其余直接 `at::linear`。不再替换主模型 LM head、draft Linear 或 FP8 Linear。 |
| `_xpu_C.rms_norm_small_m.out` | 可选 `_rms_C`；`csrc/xpu/rms/small_m_rms.cpp` | FP16、H256/512/2816/5376、rank2..4、M1..8、总 rows≤128，允许外维 stride 和未对齐输入，支持可选 weight。旧 `_C.rms_norm` 内按 dtype/shape/stride/alias 条件分派，缺模块或不适用时保留原实现。 |
| `_xpu_C.gelu_tanh_and_mul_small_m` | 可选 `_decode_aux_C`；`csrc/xpu/decode_aux/small_m_aux.cpp` | FP16、D352/1056、rows≤64，保留 GELU FP16 中间舍入。由旧 `_C.gelu_tanh_and_mul` guarded 分派。 |
| `_xpu_C.try_rotary_embedding_small_m` | 同上 | FP16、M1..8、D256/512、NeoX RoPE、partial rotary、外维 stride；原地 Q/K。在提交前检查并对不适用输入返回 false。仅通过旧 `_C.rotary_embedding` 接入。 |

所有保留的独立算子使用当前 XPU stream，有对应 Meta 实现，不保存跨调用 scratch。
可选扩展缺失时使用原入口；扩展存在但 ABI 损坏时加载错误正常传播。
具体边界以源代码检查为准，服务实际命中以 trace 为准。

## 既有 ABI 内的优化

- `_C.gelu_tanh_and_mul`：FP16 M1..8、D4096 使用列分块，其他形状保留原实现或上述辅助路径。本轮主模型 AR 不命中 D4096，不能归因于此次 AR 收益。
- `_C.reshape_and_cache_flash` 的 strided 写入：仅 FP16、非量化 KV、D512、KV heads=2、tokens1..8 使用 256 GRF，减少后续 attention 前的 GRF 模式切换。
- Flash attention：保留 AR/MTP 小 query FP16 paged 路径与 split 调优；使用 upstream attention ABI。Diffusion 的 Q64/Q128 与 M256 特化全部移除。
- 通用 grouped GEMM 调度已恢复 upstream；Gemma fused experts 独立检查自身适用条件。

## 本轮移除

- 全部 FP8 Linear 替换、`_linear_C` 扩展、相关构建选项和专用测试。
- 主模型 LM head 与 MTP/draft 的 FP16 Linear 特化、直接 `fp16_nt_gemm` API；FP16 模块缩减为上述 Router 和原生 Linear 回退。
- 旧 FP8 ABI 内的 TLA 转接、vLLM FP8 分派；FP8 Linear 使用 upstream 原路径。
- vLLM GELU/RoPE 的额外 Python 包装，以及失去调用方的 GELU 分配式包装、直接 RoPE schema；保留原 `_C` 入口下面的优化。
- 未调用的 `_decode_C` 五种模型融合、独立 `rms_norm_no_weight`、重复的 weightless RMS 源码。
- 被 fused experts 覆盖的 compact MoE tile map、prepare/gather 辅助接口及相关 Python 包装。

## 已移除的 Diffusion 启用代码

- vLLM `_xpu_ops.py` 和 `v1/attention/backends/flash_attn.py` 恢复 main。
- 删除自定义 `per_seq_causal` 参数、参数校验、逐序列 causal mask 与双向窗口启用代码。
- 删除为启用 Diffusion 增加的 chunk prefill 配置、测试和说明，恢复 upstream ABI。
- 删除 Q64/Q128 与 M256 特化及其候选调度，`fmha_xe2.cpp` 完整恢复 main。
- 删除本次 Diffusion 开发新增的 `gemma4_batch_topk`、注册、测试及 vLLM 的 M9..2048 接线；M>8 恢复 upstream routing，M1..8 保持 AR/MTP 优化。
- 原 enable 代码可从 `upstreamable_gemma_full_20260922` 找回；当前分支不负责启用 Diffusion attention。
- 同时删除无生产调用方的分配式 `rms_norm_small_m`，仅保留生产使用的 `.out`。

## 验证依据

源码、库哈希、独立算子 A/B、服务 E2E、acc5 与 trace 均记录于
`gemma_reduction_20260922` 实验目录。`no_linear` 是全部 Linear 恢复原实现的
消融版；`minimal_final` 是只保留 Router 的最终精简候选，两者不得混称。
旧版本或单算子收益不代替精简后的整模型结果。

本次删除 Diffusion 的记录见 `gemma_enable_cleanup_20260922`：54 项独立 attention
数学检查、192 项 AR/MTP 回归、25 项 RMS 与 55 项 routing 检查通过；默认
安装已同步，Gemma26B TP2/FP8/FP16/FR=2000 下 acc5=5/5。
本次未重测 TPOT、MTP 服务或 Gemma31B 服务，旧 TPOT 不标为新版本结果。
