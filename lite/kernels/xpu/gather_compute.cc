// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/xpu/gather_compute.h"
#include <vector>
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename DataType, typename IndexType>
void GatherCompute<DataType, IndexType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto x = param.X;
  auto index = param.Index;
  auto out = param.Out;
  if (out->numel() == 0) {
    out->set_target(TARGET(kXPU));
    return;
  }
  int axis = 0;
  if (param.Axis != nullptr) {
    CHECK(param.Axis->precision() == PRECISION(kInt32))
        << " xpu only support axis int32 type";
    auto* axis_data = param.Axis->template data<int>();
    axis = axis_data[0];
  }
  std::vector<int> x_dims(x->dims().data().begin(), x->dims().data().end());
  if (axis < 0) {
    axis += x_dims.size();
  }

  int r = xdnn::gather<DataType, IndexType>(
      ctx.GetRawContext(),
      x->template data<DataType>(),
      index->template data<IndexType>(),
      out->template mutable_data<DataType>(TARGET(kXPU)),
      x_dims,
      index->numel(),
      axis);

  CHECK_EQ(r, 0);
}

static void Xpu2Host(const Tensor& src, Tensor* dst) {
  dst->Resize(src.dims());
  dst->set_precision(src.precision());
  auto mem_size = src.memory_size();
  auto* data = dst->mutable_data(TARGET(kHost), mem_size);
  TargetWrapperXPU::MemcpySync(
      data, src.raw_data(), mem_size, IoDirection::DtoH);
}

static void Host2Xpu(const Tensor& src, Tensor* dst) {
  dst->Resize(src.dims());
  dst->set_precision(src.precision());
  auto mem_size = src.memory_size();
  auto* data = dst->mutable_data(TARGET(kXPU), mem_size);
  TargetWrapperXPU::MemcpySync(
      data, src.raw_data(), mem_size, IoDirection::HtoD);
}

template <typename DataType, typename IndexType>
void GatherFunc(const Tensor* x, const Tensor* index, Tensor* out) {
  auto x_dims = x->dims() auto index_size = index->dims()[0];
  auto* p_src = x->data<DataType>();
  const IndexType* p_index = index->data<IndexType>();
  auto* p_output = out->mutable_data<DataType>();

  int slice_size = 1;
  for (size_t i = 1; i < x_dims.size(); ++i) {
    slice_size *= x_dims[i];
  }
  for (int i = 0; i < index_size; ++i) {
    IndexType index_ = p_index[i];
    memcpy(p_output + i * slice_size,
           p_src + index_ * slice_size,
           slice_size * sizeof(DataType));
  }
}

template <>
void GatherCompute<int64_t, int32_t>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto x = param.X;
  auto index = param.Index;
  auto out = param.Out;

  Tensor x_t, index_t, out_t;
  Xpu2Host(*x, &x_t);
  Xpu2Host(*index, &index_t);
  out_t.Resize(out->dims());

  GatherFunc<int64_t, int32_t>(&x_t, &index_t, &out_t);

  Host2Xpu(out_t, out);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(gather, kXPU, kFloat, kNCHW, GatherXPUFloatInt32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(
    gather, kXPU, kFloat, kNCHW, GatherXPUFloatInt64, gather_float_i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(
    gather, kXPU, kFloat, kNCHW, GatherXPUInt32Int32, gather_i32_i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();
REGISTER_LITE_KERNEL(
    gather, kXPU, kFloat, kNCHW, GatherXPUInt32Int64, gather_i32_i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(
    gather, kXPU, kFloat, kNCHW, GatherXPUInt32Int32, gather_i64_i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
REGISTER_LITE_KERNEL(
    gather, kXPU, kFloat, kNCHW, GatherXPUInt64Int32, gather_i64_i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
