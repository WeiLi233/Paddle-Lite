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
// #include <fstream> // TODO: remove in the future
// #include <memory>
// #include <algorithm>
// #include <array>
// #include <cmath> // TOOD: remove in the future
#include <chrono>
#include "lite/core/target_wrapper.h"
#include "lite/kernels/xpu/fusion_decoding_compute.h"
#include "lite/utils/log/logging.h"
#include "xpu/refactor/fusion.h"

using namespace std::chrono;
using std::cout;
using std::endl;
using std::vector;
using std::array;
using std::ifstream;

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void FusionDecodingCompute::PrepareForRun() {
  auto param = this->Param<param_t>();
  VLOG(2) << "Prepare for fusion decoding";
  fd_param_.decoding_strategy   = param.decoding_strategy_.c_str();
  fd_param_.beam_size           = param.beam_size_;
  fd_param_.topk                = param.topk_;
  fd_param_.topp                = param.topp_;
  fd_param_.n_head              = param.n_head_;
  fd_param_.size_per_head       = param.size_per_head_;
  fd_param_.num_layer           = param.num_layer_;
  fd_param_.bos_id              = param.bos_id_;
  fd_param_.eos_id              = param.eos_id_;
  fd_param_.max_len             = param.max_len_;
  fd_param_.vocab_size          = static_cast<int32_t>(param.word_embedding_->dims()[0]);
  fd_param_.pos_enc_emb_dim     = static_cast<int32_t>(param.position_enc_emb_->dims()[0]);
  fd_param_.beam_search_diversity_rate  = param.beam_search_diversity_rate_;
  fd_param_.rel_len             = param.rel_len_;
  fd_param_.alpha               = param.alpha_;
  // fd_param_.is_fuse_qkv         = true; // TODO

  cross_key_quant_weight_.resize(param.num_layer_);
  cross_out_quant_weight_.resize(param.num_layer_);
  cross_query_quant_weight_.resize(param.num_layer_);
  cross_value_quant_weight_.resize(param.num_layer_);
  ffn_inter_quant_weight_.resize(param.num_layer_);
  ffn_out_quant_weight_.resize(param.num_layer_);
  self_key_quant_weight_.resize(param.num_layer_);
  self_out_quant_weight_.resize(param.num_layer_);
  self_query_quant_weight_.resize(param.num_layer_);
  self_value_quant_weight_.resize(param.num_layer_);

  cross_key_weight_ptr_vec_.resize(param.num_layer_);
  cross_key_max_ptr_vec_.resize(param.num_layer_);
  cross_key_bias_ptr_vec_.resize(param.num_layer_);

  cross_layernorm_weight_ptr_vec_.resize(param.num_layer_);
  cross_layernorm_bias_ptr_vec_.resize(param.num_layer_);

  cross_out_weight_ptr_vec_.resize(param.num_layer_);
  cross_out_max_ptr_vec_.resize(param.num_layer_);
  cross_out_bias_ptr_vec_.resize(param.num_layer_);

  cross_query_weight_ptr_vec_.resize(param.num_layer_);
  cross_query_max_ptr_vec_.resize(param.num_layer_);
  cross_query_bias_ptr_vec_.resize(param.num_layer_);

  cross_value_weight_ptr_vec_.resize(param.num_layer_);
  cross_value_max_ptr_vec_.resize(param.num_layer_);
  cross_value_bias_ptr_vec_.resize(param.num_layer_);

  ffn_inter_weight_ptr_vec_.resize(param.num_layer_);
  ffn_inter_max_ptr_vec_.resize(param.num_layer_);
  ffn_inter_bias_ptr_vec_.resize(param.num_layer_);

  ffn_layernorm_weight_ptr_vec_.resize(param.num_layer_);
  ffn_layernorm_bias_ptr_vec_.resize(param.num_layer_);

  ffn_out_weight_ptr_vec_.resize(param.num_layer_);
  ffn_out_max_ptr_vec_.resize(param.num_layer_);
  ffn_out_bias_ptr_vec_.resize(param.num_layer_);

  self_key_weight_ptr_vec_.resize(param.num_layer_);
  self_key_max_ptr_vec_.resize(param.num_layer_);
  self_key_bias_ptr_vec_.resize(param.num_layer_);

  self_layernorm_weight_ptr_vec_.resize(param.num_layer_);
  self_layernorm_bias_ptr_vec_.resize(param.num_layer_);

  self_out_weight_ptr_vec_.resize(param.num_layer_);
  self_out_max_ptr_vec_.resize(param.num_layer_);
  self_out_bias_ptr_vec_.resize(param.num_layer_);

  self_query_weight_ptr_vec_.resize(param.num_layer_);
  self_query_max_ptr_vec_.resize(param.num_layer_);
  self_query_bias_ptr_vec_.resize(param.num_layer_);

  self_value_weight_ptr_vec_.resize(param.num_layer_);
  self_value_max_ptr_vec_.resize(param.num_layer_);
  self_value_bias_ptr_vec_.resize(param.num_layer_);

  for(int32_t i=0; i<param.num_layer_; i++) {
    cross_key_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.cross_key_weight_[i]->data<float>(),
        param.cross_key_weight_[i]->dims(),
        true);
    cross_key_weight_ptr_vec_[i] = reinterpret_cast<const int16_t*>(cross_key_quant_weight_[i].data_ptr_);
    cross_key_max_ptr_vec_[i] = cross_key_quant_weight_[i].max_ptr_;
    cross_key_bias_ptr_vec_[i] = param.cross_key_bias_[i]->data<float>();

    cross_layernorm_weight_ptr_vec_[i] = param.cross_layernorm_weight_[i]->data<float>();
    cross_layernorm_bias_ptr_vec_[i] = param.cross_layernorm_bias_[i]->data<float>();

    cross_out_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.cross_out_weight_[i]->data<float>(),
        param.cross_out_weight_[i]->dims(),
        true);
    cross_out_weight_ptr_vec_[i] = reinterpret_cast<const int16_t*>(cross_out_quant_weight_[i].data_ptr_);
    cross_out_max_ptr_vec_[i] = cross_out_quant_weight_[i].max_ptr_;
    cross_out_bias_ptr_vec_[i] = param.cross_out_bias_[i]->data<float>();

    cross_query_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.cross_query_weight_[i]->data<float>(),
        param.cross_query_weight_[i]->dims(),
        true);
    cross_query_weight_ptr_vec_[i] = reinterpret_cast<const int16_t*>(cross_query_quant_weight_[i].data_ptr_);
    cross_query_max_ptr_vec_[i] = cross_query_quant_weight_[i].max_ptr_;
    cross_query_bias_ptr_vec_[i] = param.cross_query_bias_[i]->data<float>();

    cross_value_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.cross_value_weight_[i]->data<float>(),
        param.cross_value_weight_[i]->dims(),
        true);
    cross_value_weight_ptr_vec_[i] = reinterpret_cast<const int16_t*>(cross_value_quant_weight_[i].data_ptr_);
    cross_value_max_ptr_vec_[i] = cross_value_quant_weight_[i].max_ptr_;
    cross_value_bias_ptr_vec_[i] = param.cross_value_bias_[i]->data<float>();

    ffn_inter_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.ffn_inter_weight_[i]->data<float>(),
        param.ffn_inter_weight_[i]->dims(),
        true);
    ffn_inter_weight_ptr_vec_[i] = reinterpret_cast<const int16_t*>(ffn_inter_quant_weight_[i].data_ptr_);
    ffn_inter_max_ptr_vec_[i] = ffn_inter_quant_weight_[i].max_ptr_;
    ffn_inter_bias_ptr_vec_[i] = param.ffn_inter_bias_[i]->data<float>();

    ffn_layernorm_weight_ptr_vec_[i] = param.ffn_layernorm_weight_[i]->data<float>();
    ffn_layernorm_bias_ptr_vec_[i] = param.ffn_layernorm_bias_[i]->data<float>();

    ffn_out_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.ffn_out_weight_[i]->data<float>(),
        param.ffn_out_weight_[i]->dims(),
        true);
    ffn_out_weight_ptr_vec_[i] = reinterpret_cast<const int16_t*>(ffn_out_quant_weight_[i].data_ptr_);
    ffn_out_max_ptr_vec_[i] = ffn_out_quant_weight_[i].max_ptr_;
    ffn_out_bias_ptr_vec_[i] = param.ffn_out_bias_[i]->data<float>();

    self_key_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.self_key_weight_[i]->data<float>(),
        param.self_key_weight_[i]->dims(),
        true);
    self_key_weight_ptr_vec_[i] = reinterpret_cast<const int16_t*>(self_key_quant_weight_[i].data_ptr_);
    self_key_max_ptr_vec_[i] = self_key_quant_weight_[i].max_ptr_;
    self_key_bias_ptr_vec_[i] = param.self_key_bias_[i]->data<float>();

    self_layernorm_weight_ptr_vec_[i] = param.self_layernorm_weight_[i]->data<float>();
    self_layernorm_bias_ptr_vec_[i] = param.self_layernorm_bias_[i]->data<float>();

    self_out_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.self_out_weight_[i]->data<float>(),
        param.self_out_weight_[i]->dims(),
        true);
    self_out_weight_ptr_vec_[i] = reinterpret_cast<const int16_t*>(self_out_quant_weight_[i].data_ptr_);
    self_out_max_ptr_vec_[i] = self_out_quant_weight_[i].max_ptr_;
    self_out_bias_ptr_vec_[i] = param.self_out_bias_[i]->data<float>();

    self_query_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.self_query_weight_[i]->data<float>(),
        param.self_query_weight_[i]->dims(),
        true);
    self_query_weight_ptr_vec_[i] = reinterpret_cast<const int16_t*>(self_query_quant_weight_[i].data_ptr_);
    self_query_max_ptr_vec_[i] = self_query_quant_weight_[i].max_ptr_;
    self_query_bias_ptr_vec_[i] = param.self_query_bias_[i]->data<float>();

    self_value_quant_weight_[i] = \
      TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
        param.self_value_weight_[i]->data<float>(),
        param.self_value_weight_[i]->dims(),
        true);
    self_value_weight_ptr_vec_[i] = reinterpret_cast<const int16_t*>(self_value_quant_weight_[i].data_ptr_);
    self_value_max_ptr_vec_[i] = self_value_quant_weight_[i].max_ptr_;
    self_value_bias_ptr_vec_[i] = param.self_value_bias_[i]->data<float>();
  }

  emb_quant_weight_ = \
    TargetWrapperXPU::ConvertCPUWeightToXPUQuantWeight<float, int16_t>(
      param.emb_weight_->data<float>(), 
      param.emb_weight_->dims(), 
      true);

  VLOG(2) << "Finish preparing for fusion decoding";
  return;
}

// void FusionDecodingCompute::RunDecodingForward() {
//   return;
// }

void FusionDecodingCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  // cout << "INPUT DIM IS " << param.input_->dims() << endl;
  int32_t batch_size = param.input_->dims()[0];
  const int32_t max_out_len = param.rel_len_ ?  param.max_len_ + param.input_->dims()[1] : param.max_len_;
  
  std::vector<int64_t> output_dims;
  std::vector<int64_t> parent_ids_dims;
  std::vector<int64_t> sequence_length_dims({batch_size});
  if (param.decoding_strategy_ == "beam_search") {
    // TODO
    sequence_length_dims = {batch_size};
    batch_size /= param.beam_size_;
    output_dims = {max_out_len, batch_size, param.beam_size_};
    parent_ids_dims = output_dims;
  } else if (param.decoding_strategy_ == "beam_search_v2") {
    sequence_length_dims = {batch_size * 2};
    batch_size /= param.beam_size_;
    output_dims = {max_out_len, batch_size, param.beam_size_ * 2};
    parent_ids_dims = output_dims;
  } else if (param.decoding_strategy_ == "topk_sampling" ||
             param.decoding_strategy_ == "topp_sampling") {
    CHECK(false) << "\"topk_sampling\" or \"topp_sampling\" not supported! "; 
  } else {
    CHECK(false) << "Not supported decoding strategy. ";
  }

  param.output_ids_->Resize(output_dims);
  param.parent_ids_->Resize(parent_ids_dims);
  param.sequence_length_->Resize(sequence_length_dims);
  VLOG(2) << "DEBUG: EMB WEIGHT DIM IS " << param.emb_weight_->dims();
  auto s = steady_clock::now();
  int32_t ret = xdnn::fasttransformer_decoding<float, int16_t, int16_t>(
                  ctx.GetRawContext(),
                  param.input_->data<float>(), param.mem_seq_len_->data<int32_t>(), param.word_embedding_->data<float>(),
                  self_layernorm_weight_ptr_vec_, self_layernorm_bias_ptr_vec_,
                  self_query_weight_ptr_vec_, self_query_max_ptr_vec_, self_query_bias_ptr_vec_,
                  self_key_weight_ptr_vec_, self_key_max_ptr_vec_, self_key_bias_ptr_vec_,
                  self_value_weight_ptr_vec_, self_value_max_ptr_vec_, self_value_bias_ptr_vec_,
                  self_out_weight_ptr_vec_, self_out_max_ptr_vec_, self_out_bias_ptr_vec_,
                  cross_layernorm_weight_ptr_vec_, cross_layernorm_bias_ptr_vec_,
                  cross_query_weight_ptr_vec_, cross_query_max_ptr_vec_, cross_query_bias_ptr_vec_,
                  cross_key_weight_ptr_vec_, cross_key_max_ptr_vec_, cross_key_bias_ptr_vec_,
                  cross_value_weight_ptr_vec_, cross_value_max_ptr_vec_, cross_value_bias_ptr_vec_,
                  cross_out_weight_ptr_vec_, cross_out_max_ptr_vec_, cross_out_bias_ptr_vec_,
                  ffn_layernorm_weight_ptr_vec_, ffn_layernorm_bias_ptr_vec_,
                  ffn_inter_weight_ptr_vec_, ffn_inter_max_ptr_vec_, ffn_inter_bias_ptr_vec_,
                  ffn_out_weight_ptr_vec_, ffn_out_max_ptr_vec_, ffn_out_bias_ptr_vec_,
                  param.decoder_layernorm_weight_->data<float>(), param.decoder_layernorm_bias_->data<float>(),
                  reinterpret_cast<const int16_t*>(emb_quant_weight_.data_ptr_), emb_quant_weight_.max_ptr_,
                  param.emb_bias_->data<float>(), param.position_enc_emb_->data<float>(),
                  param.output_ids_->mutable_data<int32_t>(),
                  param.parent_ids_->mutable_data<int32_t>(),
                  param.sequence_length_->mutable_data<int32_t>(),
                  batch_size, param.input_->dims()[1], fd_param_);
  
  CHECK_EQ(ret, 0) << "CALLING fasttransformer_decoding error.";
  auto t = steady_clock::now();
  cout << "fusiondecoding cost: " << duration_cast<milliseconds>(t-s).count() << endl;
  return;


}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fusion_decoding,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::FusionDecodingCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("MemSeqLen", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("CrossKeyBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossKeyWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("CrossLayernormBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossLayernormWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossOutBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossOutWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("CrossQueryBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossQueryWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("CrossValueBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossValueWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
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
    .BindInput("PositionEncEmb", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
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
    .BindInput("WordEmbedding", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("OutputIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("ParentIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("SequenceLength", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();
