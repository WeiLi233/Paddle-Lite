
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/fusion_decoding_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool FusionDecodingOp::CheckShape() const {
  return true;
}

bool FusionDecodingOp::InferShapeImpl() const {
  return true;
}

bool FusionDecodingOp::AttachImpl(const cpp::OpDesc& op_desc,
                                   lite::Scope* scope) {
  param_.input_ = GetMutableVar<lite::Tensor>(scope, op_desc.Input("Input").front());
  param_.mem_seq_len_ = GetMutableVar<lite::Tensor>(scope, op_desc.Input("MemSeqLen").front());
  param_.cross_key_bias_.clear();
  for(auto& name : op_desc.Input("CrossKeyBias@VECTOR")) {
    param_.cross_key_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.cross_key_weight_.clear();
  for(auto& name : op_desc.Input("CrossKeyWeight@VECTOR")) {
    param_.cross_key_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.cross_layernorm_bias_.clear();
  for(auto& name : op_desc.Input("CrossLayernormBias@VECTOR")) {
    param_.cross_layernorm_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.cross_layernorm_weight_.clear();
  for(auto& name : op_desc.Input("CrossLayernormWeight@VECTOR")) {
    param_.cross_layernorm_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.cross_out_bias_.clear();
  for(auto& name : op_desc.Input("CrossOutBias@VECTOR")) {
    param_.cross_out_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.cross_out_weight_.clear();
  for(auto& name : op_desc.Input("CrossOutWeight@VECTOR")) {
    param_.cross_out_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.cross_query_bias_.clear();
  for(auto& name : op_desc.Input("CrossQueryBias@VECTOR")) {
    param_.cross_query_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.cross_query_weight_.clear();
  for(auto& name : op_desc.Input("CrossQueryWeight@VECTOR")) {
    param_.cross_query_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.cross_value_bias_.clear();
  for(auto& name : op_desc.Input("CrossValueBias@VECTOR")) {
    param_.cross_value_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.cross_value_weight_.clear();
  for(auto& name : op_desc.Input("CrossValueWeight@VECTOR")) {
    param_.cross_value_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.decoder_layernorm_bias_ = GetVar<lite::Tensor>(scope, op_desc.Input("DecoderLayernormBias").front());
  param_.decoder_layernorm_weight_ = GetVar<lite::Tensor>(scope, op_desc.Input("DecoderLayernormWeight").front());
  param_.emb_bias_ = GetVar<lite::Tensor>(scope, op_desc.Input("EmbBias").front());
  param_.emb_weight_ = GetVar<lite::Tensor>(scope, op_desc.Input("EmbWeight").front());
  param_.ffn_inter_bias_.clear();
  for(auto& name : op_desc.Input("FFNInterBias@VECTOR")) {
    param_.ffn_inter_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.ffn_inter_weight_.clear();
  for(auto& name : op_desc.Input("FFNInterWeight@VECTOR")) {
    param_.ffn_inter_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.ffn_layernorm_bias_.clear();
  for(auto& name : op_desc.Input("FFNLayernormBias@VECTOR")) {
    param_.ffn_layernorm_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.ffn_layernorm_weight_.clear();
  for(auto& name : op_desc.Input("FFNLayernormWeight@VECTOR")) {
    param_.ffn_layernorm_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.ffn_out_bias_.clear();
  for(auto& name : op_desc.Input("FFNOutBias@VECTOR")) {
    param_.ffn_out_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.ffn_out_weight_.clear();
  for(auto& name : op_desc.Input("FFNOutWeight@VECTOR")) {
    param_.ffn_out_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.position_enc_emb_ = GetVar<lite::Tensor>(scope, op_desc.Input("PositionEncEmb").front());
  param_.self_key_bias_.clear();
  for(auto& name : op_desc.Input("SelfKeyBias@VECTOR")) {
    param_.self_key_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_key_weight_.clear();
  for(auto& name : op_desc.Input("SelfKeyWeight@VECTOR")) {
    param_.self_key_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_layernorm_bias_.clear();
  for(auto& name : op_desc.Input("SelfLayernormBias@VECTOR")) {
    param_.self_layernorm_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_layernorm_weight_.clear();
  for(auto& name : op_desc.Input("SelfLayernormWeight@VECTOR")) {
    param_.self_layernorm_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_out_bias_.clear();
  for(auto& name : op_desc.Input("SelfOutBias@VECTOR")) {
    param_.self_out_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_out_weight_.clear();
  for(auto& name : op_desc.Input("SelfOutWeight@VECTOR")) {
    param_.self_out_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_query_bias_.clear();
  for(auto& name : op_desc.Input("SelfQueryBias@VECTOR")) {
    param_.self_query_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_query_weight_.clear();
  for(auto& name : op_desc.Input("SelfQueryWeight@VECTOR")) {
    param_.self_query_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_value_bias_.clear();
  for(auto& name : op_desc.Input("SelfValueBias@VECTOR")) {
    param_.self_value_bias_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.self_value_weight_.clear();
  for(auto& name : op_desc.Input("SelfValueWeight@VECTOR")) {
    param_.self_value_weight_.push_back(GetVar<lite::Tensor>(scope, name));
  }
  param_.word_embedding_ = GetVar<lite::Tensor>(scope, op_desc.Input("WordEmbedding").front());
  param_.output_ids_ = GetMutableVar<lite::Tensor>(scope, op_desc.Output("OutputIds").front());
  param_.parent_ids_ = GetMutableVar<lite::Tensor>(scope, op_desc.Output("ParentIds").front());
  param_.sequence_length_ = GetMutableVar<lite::Tensor>(scope, op_desc.Output("SequenceLength").front());
  param_.decoding_strategy_ = op_desc.GetAttr<std::string>("decoding_strategy");
  param_.beam_size_ = op_desc.GetAttr<int32_t>("beam_size");
  param_.topk_ = op_desc.GetAttr<int32_t>("topk");
  param_.topp_ = op_desc.GetAttr<float>("topp");
  param_.n_head_ = op_desc.GetAttr<int32_t>("n_head");
  param_.size_per_head_ = op_desc.GetAttr<int32_t>("size_per_head");
  param_.num_layer_ = op_desc.GetAttr<int32_t>("num_layer");
  param_.bos_id_ = op_desc.GetAttr<int32_t>("bos_id");
  param_.eos_id_ = op_desc.GetAttr<int32_t>("eos_id");
  param_.max_len_ = op_desc.GetAttr<int64_t>("max_len");
  param_.beam_search_diversity_rate_ = op_desc.GetAttr<float>("beam_search_diversity_rate");
  param_.alpha_ = op_desc.GetAttr<float>("alpha");
  param_.rel_len_ = op_desc.GetAttr<bool>("rel_len");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fusion_decoding,
                 paddle::lite::operators::FusionDecodingOp);

