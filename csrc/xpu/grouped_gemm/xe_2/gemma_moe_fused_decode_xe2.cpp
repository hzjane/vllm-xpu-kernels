// SPDX-License-Identifier: BSD-3-Clause
// Routed FP8 gate/up, GELU-tanh, down and weighted sum for small-M decode.
// The SYCL-TLA GEMM mainloop and ordinary GEMM/GEMV dispatch are unchanged.
#include <torch/all.h>
#include <ATen/MemoryOverlap.h>
#include <c10/xpu/XPUStream.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"
#include "gemma_moe_fused_decode_xe2.h"

namespace gemma_moe_fused {
using namespace cute;
constexpr int H = 2816, I = 352, E = 128, K = 8, MAX_M = 8;
inline int64_t align256(int64_t n) { return (n + 255) / 256 * 256; }
template <typename T>
struct alignas(16) Vec8 {
  T value[8];
};
template <typename T>
class MapName;
template <typename T, bool Gate, bool Grouped>
class GemmName;
template <typename T, bool Grouped>
class GeluName;
template <typename T, bool Grouped>
class GatherName;

// One deterministic map and gather serves both projections. Invalid routes
// are excluded from the compact rows; each invocation initializes its map.
template <typename T>
void prepare(
    sycl::queue& q,
    const T* x,
    const int* ids,
    T* packed,
    int* map,
    int* reverse,
    int rows) {
  q.submit([&](sycl::handler& h) {
    h.parallel_for<MapName<T>>(
        sycl::nd_range<1>((1 + rows) * E, E), [=](sycl::nd_item<1> it) {
          const int group = it.get_group(0), lane = it.get_local_id(0);
          if (group == 0) {
            int count = 0;
            for (int r = 0; r < rows; ++r)
              count += ids[r] == lane;
            const int tiles = (count + 7) / 8;
            const int base = sycl::exclusive_scan_over_group(
                it.get_group(), count, sycl::plus<int>());
            const int tile_base = sycl::exclusive_scan_over_group(
                it.get_group(), tiles, sycl::plus<int>());
            if (lane == E - 1) {
              map[0] = tile_base + tiles;
              map[1] = base + count;
            }
            for (int t = 0; t < tiles; ++t) {
              int* entry = map + 2 + 4 * (tile_base + t);
              entry[0] = lane;
              entry[1] = base;
              entry[2] = count;
              entry[3] = t;
            }
            return;
          }
          const int r = group - 1, expert = ids[r];
          if (expert < 0 || expert >= E) {
            if (lane == 0) reverse[r] = -1;
            return;
          }
          int before = 0;
          for (int j = lane; j < rows; j += E)
            before += ids[j] >= 0 && ids[j] < E &&
                      (ids[j] < expert || (ids[j] == expert && j < r));
          const int dest = sycl::reduce_over_group(
              it.get_group(), before, sycl::plus<int>());
          if (lane == 0) reverse[r] = dest;
          const auto* input = reinterpret_cast<const Vec8<T>*>(x);
          auto* output = reinterpret_cast<Vec8<T>*>(packed);
          for (int col = lane; col < H / 8; col += E)
            output[dest * (H / 8) + col] = input[(r / K) * (H / 8) + col];
        });
  });
}

template <typename T, bool Gate, bool Grouped>
void gemm(
    sycl::queue& q,
    const T* input,
    const void* weights,
    const float* scales,
    const int* ids,
    T* output,
    const int* map,
    int rows) {
  using Policy = MoE::w8a16_policy_m_8;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, T>>,
      Layout<typename Policy::WGTile>,
      typename Policy::SGLayout>::TiledMMA;
  const MMA mma{};
  constexpr int nk = Gate ? H : I, nn = Gate ? 2 * I : H;
  constexpr int tn = get<1>(typename Policy::WGTile{});
  constexpr int threads = size(MMA{});
  namespace sx = sycl::ext::oneapi::experimental;
  namespace ix = sycl::ext::intel::experimental;
  q.submit([&](sycl::handler& h) {
    h.parallel_for<GemmName<T, Gate, Grouped>>(
        sycl::nd_range<3>(
            {1, size_t(rows), size_t((nn / tn) * threads)},
            {1, 1, size_t(threads)}),
        sx::properties{sx::sub_group_size<16>, ix::grf_size<256>},
        [=](sycl::nd_item<3> it) {
          const int route = it.get_group(1), tile = it.get_group(2);
          int expert, base, count, mt;
          if constexpr (Grouped) {
            if (route >= map[0]) return;
            const int* entry = map + 2 + 4 * route;
            expert = entry[0];
            base = entry[1];
            count = entry[2];
            mt = entry[3];
          } else {
            expert = ids[route];
            base = route;
            count = 1;
            mt = 0;
          }
          if (expert < 0 || expert >= E) {
            for (int n = it.get_local_linear_id(); n < tn; n += threads)
              output[route * nn + tile * tn + n] = T(0.f);
            return;
          }
          const int arow = Grouped ? base : (Gate ? route / K : route);
          auto A = MoE::make_moe_tensor<T, 'R'>(
              const_cast<T*>(input) + int64_t(arow) * nk, count, nk);
          auto B = MoE::make_moe_tensor<cutlass::float_e4m3_t, 'C'>(
              reinterpret_cast<cutlass::float_e4m3_t*>(
                  const_cast<void*>(weights)) +
                  int64_t(expert) * nk * nn,
              nn,
              nk);
          auto D = MoE::make_moe_tensor<T, 'R'>(
              output + int64_t(base) * nn, count, nn);
          MoE::xe_gemm<
              typename Policy::GmemTiledCopyA,
              typename Policy::GmemTiledCopyB,
              typename Policy::GmemTiledCopyD>(
              A,
              B,
              scales + expert,
              static_cast<const T*>(nullptr),
              D,
              make_coord(mt, tile, _, 0),
              mma);
        });
  });
}

template <typename T, bool Grouped>
void activate(
    sycl::queue& q, const T* gu, T* activation, const int* map, int rows) {
  q.submit([&](sycl::handler& h) {
    h.parallel_for<GeluName<T, Grouped>>(
        sycl::nd_range<2>({size_t(rows), 96}, {1, 32}),
        [=](sycl::nd_item<2> it) {
          const int row = it.get_group(0), col = 4 * it.get_global_id(1);
          if (col >= I) return;
          if constexpr (Grouped) {
            if (row >= map[1]) {
              for (int j = 0; j < 4; ++j)
                activation[row * I + col + j] = T(0.f);
              return;
            }
          }
          for (int j = 0; j < 4; ++j) {
            const float gate = float(gu[row * (2 * I) + col + j]);
            const float up = float(gu[row * (2 * I) + I + col + j]);
            const float z =
                0.7978845608028654f * (gate + 0.044715f * gate * gate * gate);
            const T gelu(0.5f * gate * (1.f + sycl::tanh(z)));
            activation[row * I + col + j] = T(float(gelu) * up);
          }
        });
  });
}

template <typename T, bool Grouped>
void gather(
    sycl::queue& q,
    const T* partial,
    const float* weights,
    const int* ids,
    const int* reverse,
    T* output,
    int m) {
  q.submit([&](sycl::handler& h) {
    h.parallel_for<GatherName<T, Grouped>>(
        sycl::nd_range<2>({size_t(m), size_t(H / 8)}, {1, 32}),
        [=](sycl::nd_item<2> it) {
          const int row = it.get_group(0), col = it.get_global_id(1);
          float sum[8] = {};
          const auto* input = reinterpret_cast<const Vec8<T>*>(partial);
          auto* out = reinterpret_cast<Vec8<T>*>(output);
          for (int slot = 0; slot < K; ++slot) {
            const int route = row * K + slot;
            int source;
            if constexpr (Grouped)
              source = reverse[route];
            else
              source = ids[route] >= 0 && ids[route] < E ? route : -1;
            if (source < 0) continue;
            const Vec8<T> value = input[source * (H / 8) + col];
            for (int j = 0; j < 8; ++j)
              sum[j] += float(value.value[j]) * weights[route];
          }
          Vec8<T> value;
          for (int j = 0; j < 8; ++j)
            value.value[j] = T(sum[j]);
          out[row * (H / 8) + col] = value;
        });
  });
}

template <typename T, bool Grouped>
void launch(
    torch::Tensor& output,
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const torch::Tensor& ids,
    const torch::Tensor& w13,
    const torch::Tensor& s13,
    const torch::Tensor& w2,
    const torch::Tensor& s2,
    torch::Tensor& ws) {
  const int m = x.size(0), rows = m * K;
  auto* base = ws.data_ptr<uint8_t>();
  auto* map = reinterpret_cast<int*>(base);
  auto* reverse = map + 2 + 4 * rows;
  const int64_t packed_offset = align256(4 * (2 + 5 * rows));
  auto* packed = reinterpret_cast<T*>(base + packed_offset);
  auto* gu = packed + int64_t(rows) * H;
  auto* act = gu + int64_t(rows) * (2 * I);
  auto* partial = act + int64_t(rows) * I;
  const auto* input = reinterpret_cast<const T*>(x.data_ptr());
  auto& q = c10::xpu::getCurrentXPUStream(x.get_device()).queue();
  if constexpr (Grouped)
    prepare(q, input, ids.data_ptr<int>(), packed, map, reverse, rows);
  gemm<T, true, Grouped>(
      q,
      Grouped ? packed : input,
      w13.data_ptr(),
      s13.data_ptr<float>(),
      ids.data_ptr<int>(),
      gu,
      map,
      rows);
  activate<T, Grouped>(q, gu, act, map, rows);
  gemm<T, false, Grouped>(
      q,
      act,
      w2.data_ptr(),
      s2.data_ptr<float>(),
      ids.data_ptr<int>(),
      partial,
      map,
      rows);
  gather<T, Grouped>(
      q,
      partial,
      weights.data_ptr<float>(),
      ids.data_ptr<int>(),
      reverse,
      reinterpret_cast<T*>(output.data_ptr()),
      m);
}

void validate(
    const torch::Tensor& output,
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const torch::Tensor& ids,
    const torch::Tensor& w13,
    const torch::Tensor& s13,
    const torch::Tensor& w2,
    const torch::Tensor& s2,
    const torch::Tensor& ws) {
  const torch::Tensor* tensors[] = {
      &output, &x, &weights, &ids, &w13, &s13, &w2, &s2, &ws};
  for (const auto* t : tensors)
    TORCH_CHECK(
        t->is_xpu() && t->device() == x.device() && t->is_contiguous(),
        "all arguments must be contiguous on the hidden states XPU device");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= MAX_M && x.size(1) == H,
      "Gemma fused decode requires hidden states [M,2816], 1<=M<=8");
  TORCH_CHECK(
      (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
          output.scalar_type() == x.scalar_type() &&
          output.sizes() == x.sizes(),
      "hidden states/output must have the same float16 or bfloat16 dtype and "
      "shape");
  TORCH_CHECK(
      w13.scalar_type() == at::kFloat8_e4m3fn &&
          w2.scalar_type() == at::kFloat8_e4m3fn &&
          w13.sizes() == at::IntArrayRef({E, H, 2 * I}) &&
          w2.sizes() == at::IntArrayRef({E, I, H}),
      "weights must be FP8 E4M3 [128,2816,704] and [128,352,2816]");
  TORCH_CHECK(
      s13.scalar_type() == at::kFloat && s2.scalar_type() == at::kFloat &&
          s13.sizes() == at::IntArrayRef({E}) &&
          s2.sizes() == at::IntArrayRef({E}),
      "weight scales must be float32 [128]");
  TORCH_CHECK(
      ids.scalar_type() == at::kInt && weights.scalar_type() == at::kFloat &&
          ids.sizes() == at::IntArrayRef({x.size(0), K}) &&
          weights.sizes() == ids.sizes(),
      "routing must be int32 IDs and float32 weights [M,8]");
  TORCH_CHECK(ws.scalar_type() == at::kByte, "workspace must be uint8");
  const int64_t rows = x.size(0) * K;
  const int64_t needed =
      align256(4 * (2 + 5 * rows)) + 2 * rows * (2 * H + 3 * I);
  TORCH_CHECK(ws.numel() >= needed, "workspace is too small");
  const auto aligned = [](const torch::Tensor& t, uintptr_t bytes) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0;
  };
  TORCH_CHECK(
      aligned(x, 64) && aligned(w13, 64) && aligned(w2, 64) && aligned(ws, 64),
      "hidden states, matrices and workspace require 64-byte alignment");
  TORCH_CHECK(aligned(output, 16), "output requires 16-byte alignment");
  for (const auto* t : tensors) {
    if (t != &ws) at::assert_no_overlap(ws, *t);
    if (t != &output && t != &ws) at::assert_no_overlap(output, *t);
  }
}
}  // namespace gemma_moe_fused

void gemma_moe_fused_decode_xe2(
    torch::Tensor& output,
    const torch::Tensor& x,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const torch::Tensor& w13,
    const torch::Tensor& s13,
    const torch::Tensor& w2,
    const torch::Tensor& s2,
    torch::Tensor& workspace) {
  using namespace gemma_moe_fused;
  validate(output, x, topk_weights, topk_ids, w13, s13, w2, s2, workspace);
  if (x.scalar_type() == at::kHalf) {
    if (x.size(0) <= 4)
      launch<cutlass::half_t, false>(
          output, x, topk_weights, topk_ids, w13, s13, w2, s2, workspace);
    else
      launch<cutlass::half_t, true>(
          output, x, topk_weights, topk_ids, w13, s13, w2, s2, workspace);
  } else {
    if (x.size(0) <= 4)
      launch<cutlass::bfloat16_t, false>(
          output, x, topk_weights, topk_ids, w13, s13, w2, s2, workspace);
    else
      launch<cutlass::bfloat16_t, true>(
          output, x, topk_weights, topk_ids, w13, s13, w2, s2, workspace);
  }
}
