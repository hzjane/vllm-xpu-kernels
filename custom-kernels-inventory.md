# Gemma decode 汇总接口

目标分支：gemma_compile_main_0.31.0，base=main；所有vLLM pass默认关闭。

| 接口 | 完整计算与范围 |
| --- | --- |
| gemma4_qkv_norm_rope | QKV split、Q/K weighted RMSNorm、V unweighted RMSNorm、NeoX full/partial/proportional RoPE；FP16/M1..8、26B/31B TP2 |
| gemma4_router_preprocess | 无权重RMSNorm、root/channel scale；原生后续矩阵调用 |
| gemma4_norm_router_moe | 共用平方和的Router/MoE norms、root/channel scale、Router projection；FP16/H2816/E128；输出FP32 logits与FP16 MoE输入；没有独立GEMM接口 |
| gemma4_norm_add_norm | weighted norm、residual add、下一weighted norm；保留FP16舍入，输出norm与新residual |
| gemma4_scaled_add_rms_norm | residual add、设备scalar、下一norm，输出norm与scaled residual |
| gemma4_scaled_add_rms_norm_scalar | 同上，使用host常量scalar；不读GPU scalar到host |
| gemma4_gelu_tanh_and_mul | GELU-tanh与up相乘，保留GELU FP16中间舍入；用于完整dense MLP |
| gemma4_small_m_topk | E128/top8，softmax/renormalization/expert scale；精确signed-key tie-break；FP16/FP32 logits，M1..8 |
| fp8_moe_decode | 完整routed expert gate/up、GELU、down、weighted gather；FP8 per-expert scales、FP16 H2816/I352/E128/top8、M1..8；可选BMG原生try-op，unsupported在enqueue前返回false |

以上为9个native接口，使用当前XPU stream、局部scratch与Meta。增量模块为_gemma_compile_C和_moe_decode_C。

| Kernel-only开启方式 | 计算 |
| --- | --- |
| VLLM_XPU_GEMMA_DECODE_ATTENTION=1 | retained small-query causal metadata、paged decode分派、局部window与split policy；现有ABI |
| VLLM_XPU_GEMMA_CACHE_GRF256=1 | 非量化FP16/D512/KV2、M1..8、page64/128的strided KV writer GRF256；现有ABI |

环境开关默认关闭，shape/dtype/layout检查保持。没有普通FP8/FP16 Linear、LM head/draft纯矩阵替换，也没有旧Diffusion启用。历史独立GELU/ROPE/weightless RMS计算已纳入完整组合；不复制已被组合覆盖的prepare/gather接口或重复upstream global tile256。
