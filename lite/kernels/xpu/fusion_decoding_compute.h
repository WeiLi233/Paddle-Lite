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
#include "lite/backends/xpu/xpu_header_sitter.h"
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
    // void RunDecodingForward();
    
    xdnn::FTDecodingParam fd_param_;

    std::vector<XPUQuantData> cross_key_quant_weight_;
    std::vector<XPUQuantData> cross_out_quant_weight_;
    std::vector<XPUQuantData> cross_query_quant_weight_;
    std::vector<XPUQuantData> cross_value_quant_weight_;
    std::vector<XPUQuantData> ffn_inter_quant_weight_;
    std::vector<XPUQuantData> ffn_out_quant_weight_;
    std::vector<XPUQuantData> self_key_quant_weight_;
    std::vector<XPUQuantData> self_out_quant_weight_;
    std::vector<XPUQuantData> self_query_quant_weight_;
    std::vector<XPUQuantData> self_value_quant_weight_;
    XPUQuantData emb_quant_weight_;

    std::vector<const int16_t*> cross_key_weight_ptr_vec_;
    std::vector<const float*> cross_key_max_ptr_vec_;
    std::vector<const float*> cross_key_bias_ptr_vec_;

    std::vector<const float*> cross_layernorm_weight_ptr_vec_;
    std::vector<const float*> cross_layernorm_bias_ptr_vec_;

    std::vector<const int16_t*> cross_out_weight_ptr_vec_;
    std::vector<const float*> cross_out_max_ptr_vec_;
    std::vector<const float*> cross_out_bias_ptr_vec_;

    std::vector<const int16_t*> cross_query_weight_ptr_vec_;
    std::vector<const float*> cross_query_max_ptr_vec_;
    std::vector<const float*> cross_query_bias_ptr_vec_;

    std::vector<const int16_t*> cross_value_weight_ptr_vec_;
    std::vector<const float*> cross_value_max_ptr_vec_;
    std::vector<const float*> cross_value_bias_ptr_vec_;

    std::vector<const int16_t*> ffn_inter_weight_ptr_vec_;
    std::vector<const float*> ffn_inter_max_ptr_vec_;
    std::vector<const float*> ffn_inter_bias_ptr_vec_;

    std::vector<const float*> ffn_layernorm_weight_ptr_vec_;
    std::vector<const float*> ffn_layernorm_bias_ptr_vec_;

    std::vector<const int16_t*> ffn_out_weight_ptr_vec_;
    std::vector<const float*> ffn_out_max_ptr_vec_;
    std::vector<const float*> ffn_out_bias_ptr_vec_;

    std::vector<const int16_t*> self_key_weight_ptr_vec_;
    std::vector<const float*> self_key_max_ptr_vec_;
    std::vector<const float*> self_key_bias_ptr_vec_;

    std::vector<const float*> self_layernorm_weight_ptr_vec_;
    std::vector<const float*> self_layernorm_bias_ptr_vec_;

    std::vector<const int16_t*> self_out_weight_ptr_vec_;
    std::vector<const float*> self_out_max_ptr_vec_;
    std::vector<const float*> self_out_bias_ptr_vec_;

    std::vector<const int16_t*> self_query_weight_ptr_vec_;
    std::vector<const float*> self_query_max_ptr_vec_;
    std::vector<const float*> self_query_bias_ptr_vec_;

    std::vector<const int16_t*> self_value_weight_ptr_vec_;
    std::vector<const float*> self_value_max_ptr_vec_;
    std::vector<const float*> self_value_bias_ptr_vec_;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
