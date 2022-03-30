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
#include "lite/kernels/xpu/tile_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include <algorithm>

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void TileCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::TileParam>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x = param.X;
  auto out = param.Out;
  std::vector<int> repeat_times;

  if (param.RepeatTimes) {
    auto repeat_times_size = param.RepeatTimes->data_size();
    for (int64_t i = 0; i < repeat_times_size; i++) {
      repeat_times.push_back(param.RepeatTimes->template data<int>()[i]);
    }
  }
  if (!param.repeat_times_tensor.empty()) {
    size_t repeat_times_tensor_size = param.repeat_times_tensor.size();
    for (size_t i = 0; i < repeat_times_tensor_size; i++) {
      repeat_times.push_back(
          param.repeat_times_tensor[i]->template data<int32_t>()[0]);
    }
  }
  if (!param.repeat_times.empty()) {
    for (auto v : param.repeat_times) {
      repeat_times.push_back(v);
    }
  }

  auto x_shape = x->dims().Vectorize();
  std::vector<int> x_dims(x_shape.begin(), x_shape.end());
  // broadcast for x_dims.size() equal to repeat_times.size()
  if (repeat_times.size() < x_dims.size()) {
    int diff = x_dims.size() - repeat_times.size();
    repeat_times.insert(repeat_times.begin(), diff, 1);
  } else {
    int diff = repeat_times.size() - x_dims.size();
    x_dims.insert(x_dims.begin(), diff, 1);
  }

  // yshape
  std::vector<int> out_dims(x_dims.begin(),x_dims.end());
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    out_dims[i] *= repeat_times[i];
  }

  // broadcast ( x y xshape yshape )
  int r = xdnn::broadcast<T>(ctx.GetRawContext(),
                             x->template data<T>(),
                             out->template mutable_data<T>(TARGET(kXPU)),
                             x_dims,
                             out_dims);
  
  CHECK_EQ(r, 0);
}
}  // namespace xpu

}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using tile_float =
    paddle::lite::kernels::xpu::TileCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kXPU, kFloat, kNCHW, tile_float, def_float)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();
using tile_int32 =
    paddle::lite::kernels::xpu::TileCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kXPU, kFloat, kNCHW, tile_int32, def_int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();
using tile_int64 =
    paddle::lite::kernels::xpu::TileCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kXPU, kFloat, kNCHW, tile_int64, def_int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();

using tile_int8 =
    paddle::lite::kernels::xpu::TileCompute<int8_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(tile, kXPU, kFloat, kNCHW, tile_int8, def_int8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("RepeatTimes",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("repeat_times_tensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .Finalize();

// using tile_bool =
//     paddle::lite::kernels::xpu::TileCompute<bool, PRECISION(kFloat)>;
// REGISTER_LITE_KERNEL(tile, kXPU, kFloat, kNCHW, tile_bool, def_bool)
//     .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kBool))})
//     .BindInput("RepeatTimes",
//                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
//     .BindInput("repeat_times_tensor",
//                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
//     .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kBool))})
//     .Finalize();