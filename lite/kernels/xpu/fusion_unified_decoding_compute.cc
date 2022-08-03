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

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath> // TOOD: remove in the future
#include <chrono>
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/kernels/xpu/fusion_unified_decoding_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"
#include "xpu/refactor/math.h"

using std::cout;
using std::endl;
using std::vector;

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void FusionUnifiedDecodingCompute::PrepareForRun() {
  // auto& ctx = this->ctx_->As<XPUContext>();
  auto& param = this->Param<param_t>();
  VLOG(2) << "Prepare for fusion_unified_decoding";
  
  fud_param_.decoding_strategy = param.decoding_strategy_.c_str();
  fud_param_.beam_size = param.beam_size_;
  fud_param_.topk = param.topk_;
  fud_param_.topp = param.topp_;
  fud_param_.n_head = param.n_head_;
  fud_param_.size_per_head = param.size_per_head_;
  fud_param_.num_layer = param.num_layer_;
  fud_param_.bos_id = param.bos_id_;
  fud_param_.eos_id = param.eos_id_;
  fud_param_.max_len = param.max_len_;
  fud_param_.beam_search_diversity_rate = param.beam_search_diversity_rate_;
  fud_param_.unk_id = param.unk_id_;
  fud_param_.mask_id = param.mask_id_;
  fud_param_.temperature = param.temperature_;
  fud_param_.len_penalty = param.len_penalty_;
  fud_param_.normalize_before = param.normalize_before_;
  fud_param_.pos_bias = param.pos_bias_;
  if(param.hidden_act_ == "gelu") {
    fud_param_.hidden_act = xdnn::Activation_t::GELU;
  } else {
    CHECK(false) << "hidden_act is ilegal: " << param.hidden_act_;
  }
  fud_param_.rel_len = param.rel_len_;
  fud_param_.early_stopping = param.early_stopping_;
  fud_param_.min_length = param.min_length_;
  fud_param_.vocab_size = static_cast<int32_t>(param.word_embedding_->dims()[0]);

  self_ln_weight_ptr_vec_.resize(param.num_layer_);
  self_ln_bias_ptr_vec_.resize(param.num_layer_);
  
  self_q_weight_ptr_vec_.resize(param.num_layer_);
  self_q_max_ptr_vec_.resize(param.num_layer_);
  self_q_bias_ptr_vec_.resize(param.num_layer_);

  self_k_weight_ptr_vec_.resize(param.num_layer_, nullptr);
  self_k_max_ptr_vec_.resize(param.num_layer_, nullptr);
  self_k_bias_ptr_vec_.resize(param.num_layer_, nullptr);

  self_v_weight_ptr_vec_.resize(param.num_layer_, nullptr);
  self_v_max_ptr_vec_.resize(param.num_layer_, nullptr);
  self_v_bias_ptr_vec_.resize(param.num_layer_, nullptr);

  self_out_weight_ptr_vec_.resize(param.num_layer_);
  self_out_max_ptr_vec_.resize(param.num_layer_);
  self_out_bias_ptr_vec_.resize(param.num_layer_);
  
  ffn_ln_weight_ptr_vec_.resize(param.num_layer_);
  ffn_ln_bias_ptr_vec_.resize(param.num_layer_);

  ffn_inter_weight_ptr_vec_.resize(param.num_layer_);
  ffn_inter_max_ptr_vec_.resize(param.num_layer_);
  ffn_inter_bias_ptr_vec_.resize(param.num_layer_);

  ffn_out_weight_ptr_vec_.resize(param.num_layer_);
  ffn_out_max_ptr_vec_.resize(param.num_layer_);
  ffn_out_bias_ptr_vec_.resize(param.num_layer_);

  ffn_inter_quant_weight_.resize(param.num_layer_);
  ffn_out_quant_weight_.resize(param.num_layer_);
  self_key_quant_weight_.resize(param.num_layer_);
  self_out_quant_weight_.resize(param.num_layer_);
  self_query_quant_weight_.resize(param.num_layer_);
  self_value_quant_weight_.resize(param.num_layer_);

  for(int i=0; i<param.num_layer_; i++) {
    ffn_inter_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.ffn_inter_weight_[i]->data<float>(),
        param.ffn_inter_weight_[i]->dims(),
        true);
    ffn_inter_weight_ptr_vec_[i] = \
      reinterpret_cast<const int16_t*>(ffn_inter_quant_weight_[i].data_ptr_);
    ffn_inter_max_ptr_vec_[i] = ffn_inter_quant_weight_[i].max_ptr_;
    ffn_inter_bias_ptr_vec_[i] = param.ffn_inter_bias_[i]->data<float>();

    ffn_out_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.ffn_out_weight_[i]->data<float>(),
        param.ffn_out_weight_[i]->dims(),
        true);
    ffn_out_weight_ptr_vec_[i] = \
      reinterpret_cast<const int16_t*>(ffn_out_quant_weight_[i].data_ptr_);
    ffn_out_max_ptr_vec_[i] = ffn_out_quant_weight_[i].max_ptr_;
    ffn_out_bias_ptr_vec_[i] = param.ffn_out_bias_[i]->data<float>();
    
    /*
    TODO: why self_k dim is [1]??
    self_key_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.self_k_weight_[i]->data<float>(),
        param.self_k_weight_[i]->dims(),
        true);
    self_k_weight_ptr_vec_[i] = \
      reinterpret_cast<const int16_t*>(self_key_quant_weight_[i].data_ptr_);
    self_k_max_ptr_vec_[i] = self_key_quant_weight_[i].max_ptr_;
    self_k_bias_ptr_vec_[i] = param.self_k_bias_[i]->data<float>();
    */
    
    self_out_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.self_out_weight_[i]->data<float>(),
        param.self_out_weight_[i]->dims(),
        true);
    self_out_weight_ptr_vec_[i] = \
      reinterpret_cast<const int16_t*>(self_out_quant_weight_[i].data_ptr_);
    self_out_max_ptr_vec_[i] = self_out_quant_weight_[i].max_ptr_;
    self_out_bias_ptr_vec_[i] = param.self_out_bias_[i]->data<float>();

    self_query_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.self_q_weight_[i]->data<float>(),
        param.self_q_weight_[i]->dims(),
        true);
    self_q_weight_ptr_vec_[i] = \
      reinterpret_cast<const int16_t*>(self_query_quant_weight_[i].data_ptr_);
    self_q_max_ptr_vec_[i] = self_query_quant_weight_[i].max_ptr_;
    self_q_bias_ptr_vec_[i] = param.self_q_bias_[i]->data<float>();
    
    /*
    self_value_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.self_v_weight_[i]->data<float>(),
        param.self_v_weight_[i]->dims(),
        true);
    self_v_weight_ptr_vec_[i] = \
      reinterpret_cast<const int16_t*>(self_value_quant_weight_[i].data_ptr_);
    self_v_max_ptr_vec_[i] = self_value_quant_weight_[i].max_ptr_;
    self_v_bias_ptr_vec_[i] = param.self_v_bias_[i]->data<float>();
    */

    self_ln_weight_ptr_vec_[i] = param.self_ln_weight_[i]->data<float>();
    self_ln_bias_ptr_vec_[i] = param.self_ln_bias_[i]->data<float>();

    ffn_ln_weight_ptr_vec_[i] = param.ffn_ln_weight_[i]->data<float>();
    ffn_ln_bias_ptr_vec_[i] = param.ffn_ln_bias_[i]->data<float>();
  }

  trans_quant_weight_ = \
    TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
      param.trans_weight_->data<float>(), 
      param.trans_weight_->dims(), 
      true);
  
  embedding_quant_weight_ = \
    TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
      param.embedding_weight_->data<float>(), 
      param.embedding_weight_->dims(), 
      true);

  VLOG(2) << "End for fusion_unified_decoding";
  return;
}

void FusionUnifiedDecodingCompute::RunDecodingForward() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  VLOG(2) << "A GOOD START";
  VLOG(2) << "input dim " << param.input_ids_->dims();
  VLOG(2) << "max len " << fud_param_.max_len; //(20,60)

  const int32_t batch_size = param.input_ids_->dims()[0];
  const int32_t max_out_len = param.rel_len_ ? \
                  param.max_len_ + param.input_ids_->dims()[1] : param.max_len_;
  
  if(param.decoding_strategy_ != "topk_sampling") {
    CHECK(false) \
      << "Ony 'topk_sampling' is supported now, your strategy is " \
      << param.decoding_strategy_;
  }
  
  // fud_param_.max_len = max_out_len;

  param.output_ids_->Resize({max_out_len, batch_size});
  param.parent_ids_->Resize({1});
  param.output_scores_->Resize({batch_size});
  param.sequence_length_->Resize({batch_size});
    
  // cout << "type emb shape is " << param.type_embedding_weight_->dims() << endl;
  // cout << "pos embedding table is " << param.positional_embedding_weight_->dims() << endl;
  // cout << "attn dims is " << param.attn_mask_->dims() << endl;
  // cout << "mem_seq size is " << param.mem_seq_len_->dims() << endl;
  // cout << "decoder pos ids is " << param.decoder_position_ids_->dims() << endl;
  /*
  cout << "LOGITS MASK " << endl;
  vector<float> mask_cpu(param.logits_mask_->numel());
  TargetWrapperXPU::MemcpySync(
            mask_cpu.data(), param.logits_mask_->data<float>(), \
            mask_cpu.size() * sizeof(float), IoDirection::DtoH); 
  for(int i=0; i<mask_cpu.size(); i++) {
    cout << mask_cpu[i] << ' ';
  }
  cout << endl;
  */
  int32_t ret = xdnn::fasttransformer_unified_decoding<float, int16_t, int16_t>(
            ctx.GetRawContext(), 
            param.input_ids_->data<int32_t>(),
            param.attn_mask_->data<float>(),
            param.mem_seq_len_->data<int32_t>(),
            param.type_id_->data<int32_t>(),
            param.decoder_type_id_->data<int32_t>(),
            param.logits_mask_->data<float>(),
            param.word_embedding_->data<float>(),
            self_ln_weight_ptr_vec_,
            self_ln_bias_ptr_vec_,
            self_q_weight_ptr_vec_,
            self_q_max_ptr_vec_,
            self_q_bias_ptr_vec_,
            self_k_weight_ptr_vec_,
            self_k_max_ptr_vec_,
            self_k_bias_ptr_vec_,
            self_v_weight_ptr_vec_,
            self_v_max_ptr_vec_,
            self_v_bias_ptr_vec_,
            self_out_weight_ptr_vec_,
            self_out_max_ptr_vec_,
            self_out_bias_ptr_vec_,
            ffn_ln_weight_ptr_vec_,
            ffn_ln_bias_ptr_vec_,
            ffn_inter_weight_ptr_vec_,
            ffn_inter_max_ptr_vec_,
            ffn_inter_bias_ptr_vec_,
            ffn_out_weight_ptr_vec_,
            ffn_out_max_ptr_vec_,
            ffn_out_bias_ptr_vec_,
            param.decoder_ln_weight_->data<float>(),
            param.decoder_ln_bias_->data<float>(),
            reinterpret_cast<const int16_t*>(trans_quant_weight_.data_ptr_),
            trans_quant_weight_.max_ptr_,
            param.trans_bias_->data<float>(),
            param.lm_ln_weight_->data<float>(),
            param.lm_ln_bias_->data<float>(),
            reinterpret_cast<const int16_t*>(embedding_quant_weight_.data_ptr_),
            embedding_quant_weight_.max_ptr_,
            param.embedding_bias_->data<float>(),
            param.positional_embedding_weight_->data<float>(),
            param.type_embedding_weight_->data<float>(),
            param.role_id_->data<int32_t>(),
            param.decoder_role_id_->data<int32_t>(),
            param.role_embedding_table_->data<float>(),
            param.position_ids_->data<int32_t>(),
            param.decoder_position_ids_->data<int32_t>(),
            param.output_ids_->mutable_data<int32_t>(),
            param.output_scores_->mutable_data<float>(),
            param.parent_ids_->mutable_data<int32_t>(),
            param.sequence_length_->mutable_data<int32_t>(),
            batch_size,
            param.input_ids_->dims()[1], 
            fud_param_);
  CHECK_EQ(ret, 0) << "Calling fusion_unified_decoding error";
  
  // cout << "MYOUTID ";
  // for(int i=0; i<param.sequence_length_->data<int32_t>()[0]; i++) {
  //   cout << param.output_ids_->data<int32_t>()[i*batch_size] << ' ';
  // }
  // cout << endl;
  return;
}

void FusionUnifiedDecodingCompute::Run() {
  this->RunDecodingForward();
  return;
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fusion_unified_decoding,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::FusionUnifiedDecodingCompute,
                     def)
    .BindInput("AttnMask", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("DecPositionIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("DecRoleIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("DecTypeIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("DecoderLayernormBias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("DecoderLayernormWeight", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("EmbBias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("EmbWeight", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("FFNInterBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNInterWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("FFNLayernormBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNLayernormWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNOutBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNOutWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("InputIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("LMLayernormBias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("LMLayernormWeight", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("LogitsMask", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("MemSeqLen", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("PositionEncEmb", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("PositionIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("RoleEmbedding", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("RoleIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("SelfKeyBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfKeyWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("SelfLayernormBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfLayernormWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfOutBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfOutWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("SelfQueryBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfQueryWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("SelfValueBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfValueWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("TransBias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("TransWeight", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("TypeEmb", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("TypeIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("WordEmbedding", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("OutputIds", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("OutputScores", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("ParentIds", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("SequenceLength", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
