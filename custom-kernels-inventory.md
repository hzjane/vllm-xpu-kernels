# Qwen3.8 普通 SYCL 算子与安装清单

`QWEN38_KERNELS_ENABLED` 控制独立 `_qwen38_C` 扩展，可只构建/安装
此目标，不重编已有扩展。计算采用普通 SYCL、subgroup 与 SYCL-TLA，
不包含 `sycl::ext::intel::esimd`，不链接 `custom_esimd_kernels_vllm`。
Torch dispatcher 与 direct Pybind 共用这一 DSO，不混载独立测试库。

| 基础入口 | 契约 |
| --- | --- |
| `q4_0_quantize` | XPU FP16/BF16 连续 `[N,K]`，K为128倍数；输出 unsigned nibble 的 int32 `[N,K/8]` 和有符号 FP16 scale。保持 GGML Q4_0 位值。 |
| `int4_linear` | FP16 `[M,K]`、uint8 `[N,K/2]` 与 signed FP16 scale；M1按 FP32 dequant，M>1按 FP16 dequant 后 FP32 累加。M2–8 适配形状走 XMX，其余走普通 SYCL。 |
| `int4_linear_fused2` | M1两个独立投影、两个 caller-owned 输出，一次提交。 |
| `ngram_decode_ids` | int64 `[M]`、历史 `[M,2]`；uint64 wrapping hash＋signed remainder，输出 `[M,16]`。 |
| `ngram_host_lookup` / `chunked` | XPU读取 pinned CPU FP16/BF16 `[V,160]` 的原始位值；TP局部词表外输出零。chunked保留TP4的分块加载布局。 |

| 模块族 | 统一模块入口与 provider |
| --- | --- |
| HC | `_qwen38_C` 的 norm/combine/gate/down/up；`qwen38_hc_sycl` workspace 和同一 DSO 的 direct class。TP4维度10240/2560、低秩320，M1/M2–8。 |
| PLE | 13个 `ple_*` dispatcher：NGram、embedding、norm、score/gate、融合 norm、residual 和 decode/prefill/spec short-conv。trusted 元数据只接受生产者证明，不把普通调用伪装 trusted。 |
| GDN | `gdn_decode_sycl`、`gdn_spec_v2_sycl`、`gdn_norm_gate_sycl`、`gdn_norm_int4_sycl`；`qwen38_gdn.py` 保留模型侧参数顺序。TP4/TP8 sequential 几何；不宣告未实现的 interleaved 布局。 |
| MoE | `qwen38_moe_sycl` dispatcher/classes、同一 DSO 的 direct workspace；`qwen38_moe.py` 提供 compact80/160 tuple与prefill DOWN。router是Q4_0，routed权重是signed-S4，不能混淆。 |
| QSA | direct Pybind 的 compression、selection、token-split attention、QKV/norm/RoPE、cache-store/M1 transaction；`qwen38_qsa.py` 保留旧 positional ABI并传真实 host max_seq_len。M≤4096，C++分块复用显式 scratch。 |

所有 native try/fallback 在首次提交前校验支持条件；提交后的异常传播，
不能重放会写同一状态的 fallback。使用输入设备的 current stream，
记录 allocator 生命周期；recordStream 不是跨 stream 数据依赖。
host lookup 同时记录 pinned host allocation，提前释放 Python owner
也不得在队列完成前回收。workspace 按 stream/M 持有，不能跨飞行调用共享。

配套安装文件为 `_qwen38_C.cpython-*.so`、`qwen38.py`、
`qwen38_gdn.py`、`qwen38_moe.py`、`qwen38_qsa.py`。
Pybind direct Tensor ABI需要对应 CPython/Torch，不能标成abi3扩展。
部分构建示例：`cmake --build <build> --target _qwen38_C -j2`；
部分安装用 `cmake --install <build> --component _qwen38_C --prefix <site-packages>`。
安装前备份/hash核对，通过新进程确认真实导入路径，不把build-only环境变量
当作默认安装完成。完整模型 async/MTP、多模态、准确性和 E2E 仍以运行记录
为准，代码/算子单测存在不代表整链验收。
