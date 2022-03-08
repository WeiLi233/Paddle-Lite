// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/topk_v2_compute.h"
#include "lite/backends/xpu/target_wrapper.h"  // XPUScratchPadGuard
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void TopkV2Compute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  DDim x_dims = param.X->dims();

  int axis = param.axis;
  int dim_size = x_dims.size();
  if(axis < 0) axis += dim_size;
  if(axis < 0 || axis >= dim_size) LOG(FATAL) << "axis should be in range [-R,R), where R is the rank of tensor X";
  int K = 1;
  if(param.k_is_tensor){
    K = param.KTensor->data<int>()[0];
  }else{
    K = param.K;
  }

  if(axis == dim_size - 1){
    int m = x_dims.production() / x_dims[dim_size - 1];
    int n = x_dims[dim_size - 1];

    XPUScratchPadGuard indices_xpu_guard_ =
        TargetWrapperXPU::MallocScratchPad(m * K * sizeof(int));

    int* indices_int32_device = reinterpret_cast<int*>(indices_xpu_guard_->addr_);
    int64_t* indices_int64_device =
        param.Indices->mutable_data<int64_t>(TARGET(kXPU));

    int r = xdnn::sorted_topk(ctx.GetRawContext(),
                                param.X->data<float>(),
                                param.Out->mutable_data<float>(TARGET(kXPU)),
                                indices_int32_device,
                                m,
                                n,
                                K);
    CHECK_EQ(r, 0);

    r = xdnn::cast<int, int64_t>(
        ctx.GetRawContext(), indices_int32_device, indices_int64_device, m * K);

    CHECK_EQ(r, 0);
  }else{
    int n = x_dims[axis];
    int m = x_dims.production() / n;
    std::vector<int> x_shape(dim_size, 0);
    std::vector<int> x_trans_permute(dim_size, 0);

    std::vector<int> out_shape(dim_size, 0);
    std::vector<int> out_trans_permute(dim_size, 0);

    for (int i = 0; i < dim_size; ++i) {
      x_shape[i] = x_dims[i];
      if(i == axis){
        x_trans_permute[i] = i+1;
        out_shape[i] = x_dims[i+1];
        out_trans_permute[i] = dim_size - 1;
      }else if (i < axis){
        x_trans_permute[i] = i;
        out_shape[i] = x_dims[i];
        out_trans_permute[i] = i;
      }else if (i == dim_size - 1){
        x_trans_permute[i] = axis;
        out_shape[i] = K;
        out_trans_permute[i] = i-1;
      }else{
        x_trans_permute[i] = i+1;
        out_shape[i] = x_dims[i+1];
        out_trans_permute[i] = i-1;
      }
    }
    
    XPUScratchPadGuard x_transpose_xpu_guard_ =
        TargetWrapperXPU::MallocScratchPad(x_dims.production() * sizeof(float));

    float* x_transpose_device = reinterpret_cast<float*>(x_transpose_xpu_guard_->addr_);

    XPUScratchPadGuard out_before_trans_xpu_guard_ =
        TargetWrapperXPU::MallocScratchPad(m * K * sizeof(float));

    float* out_before_trans_device = reinterpret_cast<float*>(out_before_trans_xpu_guard_->addr_);

    XPUScratchPadGuard indices_before_trans_xpu_guard_ =
        TargetWrapperXPU::MallocScratchPad(m * K * sizeof(int));

    int* indices_before_trans_device = reinterpret_cast<int*>(indices_before_trans_xpu_guard_->addr_);

    XPUScratchPadGuard indices_xpu_guard_ =
        TargetWrapperXPU::MallocScratchPad(m * K * sizeof(int));

    int* indices_int32_device = reinterpret_cast<int*>(indices_xpu_guard_->addr_);

    int r =
      xdnn::transpose<float>(ctx.GetRawContext(),
                             param.X->data<float>(),
                             x_transpose_device,
                             x_shape,
                             x_trans_permute);
    CHECK_EQ(r, 0);
    
    r = xdnn::sorted_topk(ctx.GetRawContext(),
                                x_transpose_device,
                                out_before_trans_device,
                                indices_before_trans_device,
                                m,
                                n,
                                K);
    CHECK_EQ(r, 0);

    r =
      xdnn::transpose<float>(ctx.GetRawContext(),
                             out_before_trans_device,
                             param.Out->mutable_data<float>(TARGET(kXPU)),
                             out_shape,
                             out_trans_permute);
    CHECK_EQ(r, 0);

    r =
      xdnn::transpose<int>(ctx.GetRawContext(),
                             indices_before_trans_device,
                             indices_int32_device,
                             out_shape,
                             out_trans_permute);
    CHECK_EQ(r, 0);

    r = xdnn::cast<int, int64_t>(
        ctx.GetRawContext(), indices_int32_device, param.Indices->mutable_data<int64_t>(TARGET(kXPU)), m * K);

    CHECK_EQ(r, 0);
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    top_k_v2, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::TopkV2Compute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();