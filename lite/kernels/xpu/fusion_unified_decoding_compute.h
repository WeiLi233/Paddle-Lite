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
#include "lite/backends/xpu/xpu_quantizer.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class FusionUnifiedDecodingCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
  public:
    using param_t = operators::FusionUnifiedDecodingParam;

    void PrepareForRun() override;

    void Run() override;

    ~FusionUnifiedDecodingCompute() override = default;

  private:
    void RunDecodingForward();

    std::vector<XPUQuantData> ffn_inter_quant_weight_;
    std::vector<XPUQuantData> ffn_out_quant_weight_;
    std::vector<XPUQuantData> self_key_quant_weight_;
    std::vector<XPUQuantData> self_out_quant_weight_;
    std::vector<XPUQuantData> self_query_quant_weight_;
    std::vector<XPUQuantData> self_value_quant_weight_;
    XPUQuantData trans_quant_weight_;

    std::vector<const float*>  self_ln_weight_ptr_vec_;
    std::vector<const float*>  self_ln_bias_ptr_vec_;

    std::vector<const int16_t*>  self_q_weight_ptr_vec_;
    std::vector<const float*>  self_q_max_ptr_vec_;
    std::vector<const float*>  self_q_bias_ptr_vec_;

    std::vector<const int16_t*>  self_k_weight_ptr_vec_;
    std::vector<const float*>  self_k_max_ptr_vec_;
    std::vector<const float*>  self_k_bias_ptr_vec_;

    std::vector<const int16_t*>  self_v_weight_ptr_vec_;
    std::vector<const float*>  self_v_max_ptr_vec_;
    std::vector<const float*>  self_v_bias_ptr_vec_;

    std::vector<const int16_t*>  self_out_weight_ptr_vec_;
    std::vector<const float*>  self_out_max_ptr_vec_;
    std::vector<const float*>  self_out_bias_ptr_vec_;

    std::vector<const float*>  ffn_ln_weight_ptr_vec_;
    std::vector<const float*>  ffn_ln_bias_ptr_vec_;

    std::vector<const int16_t*>  ffn_inter_weight_ptr_vec_;
    std::vector<const float*>  ffn_inter_max_ptr_vec_;
    std::vector<const float*>  ffn_inter_bias_ptr_vec_;

    std::vector<const int16_t*>  ffn_out_weight_ptr_vec_;
    std::vector<const float*>  ffn_out_max_ptr_vec_;
    std::vector<const float*>  ffn_out_bias_ptr_vec_;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
