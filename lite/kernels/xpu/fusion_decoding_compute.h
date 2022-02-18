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

#pragma once

#include <vector>
#include <memory>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class FusionDecodingCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
  public:
    using param_t = operators::FusionDecodingParam;

    void PrepareForRun() override;

    void Run() override;

    ~FusionDecodingCompute() override = default;

  private:
    void RunDecodingForward();

    /*
    std::vector<const float*> arg_cross_key_bias_;
    std::vector<const float*> arg_cross_key_weight_;
    std::vector<const float*> arg_cross_layernorm_bias_;
    std::vector<const float*> arg_cross_layernorm_weight_;
    std::vector<const float*> arg_cross_out_bias_;
    std::vector<const float*> arg_cross_out_weight_;
    std::vector<const float*> arg_cross_query_bias_;
    std::vector<const float*> arg_cross_query_weight_;
    std::vector<const float*> arg_cross_value_bias_;
    std::vector<const float*> arg_cross_value_weight_;
    std::vector<const float*> arg_ffn_inter_bias_;
    std::vector<const float*> arg_ffn_inter_weight_;
    std::vector<const float*> arg_ffn_layernorm_bias_;
    std::vector<const float*> arg_ffn_layernorm_weight_;
    std::vector<const float*> arg_ffn_out_bias_;
    std::vector<const float*> arg_ffn_out_weight_;
    std::vector<const float*> arg_self_key_bias_;
    std::vector<const float*> arg_self_key_weight_;
    std::vector<const float*> arg_self_layernorm_bias_;
    std::vector<const float*> arg_self_layernorm_weight_;
    std::vector<const float*> arg_self_out_bias_;
    std::vector<const float*> arg_self_out_weight_;
    std::vector<const float*> arg_self_query_bias_;
    std::vector<const float*> arg_self_query_weight_;
    std::vector<const float*> arg_self_value_bias_;
    std::vector<const float*> arg_self_value_weight_;
    */
    XPUScratchPadGuard weight_max_guard_;
    XPUScratchPadGuard input_max_guard_;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
