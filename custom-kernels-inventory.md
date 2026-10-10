# Gemma compile 融合算子增量清单

本文件记录 `upstreamable_gemma_0.31.0` 新增的接口；已有算子沿用基线。

| 算子 | 计算 | 适用范围 |
| --- | --- | --- |
| `_xpu_C::gemma4_router_preprocess` | 无权重 RMSNorm → root scaling → 逐维 scaling，保留 FP16 舍入 | 快路径 FP16、H2816、M1..8、连续；其他合法输入回退原生组合 |
| `_xpu_C::gemma4_qkv_norm_rope` | QKV split → Q/K 加权 RMSNorm → NeoX full/partial RoPE；V 无权重 RMSNorm | FP16、M1..8，TP2 几何：D256/Q8/KV4、D256/Q16/KV8、D512/Q8/KV1、D512/Q16/KV2 |

两个接口均是 functional op；不修改输入，使用当前 XPU stream，提供 Meta 实现。
QKV 接口返回三个连续输出，D512 支持普通 rotary_dim=128 和 Gemma proportional 的 full-D512 identity cache；D256 为 full RoPE。
`native_rope` 保留原生路径与 custom RoPE 路径各自的乘积舍入方式。

增量库：`_gemma_compile_C.abi3.so`；加载入口：
`vllm_xpu_kernels.gemma4_router_preprocess`。
构建开关 `GEMMA_COMPILE_KERNELS_ENABLED=ON`，可仅构建该 target。
只加回完整融合；不修改普通 GEMM/GEMV、权重布局或通信 kernel。
