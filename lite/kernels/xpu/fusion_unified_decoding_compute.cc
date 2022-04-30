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
#include <fstream> // TODO: remove in the future
#include <memory>
#include <algorithm>
#include <array>
#include <cmath> // TOOD: remove in the future
#include <vector> // TODO: ...
#include <chrono>
#include "lite/kernels/xpu/fusion_unified_decoding_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

using std::cout;
using std::endl;
using std::vector;
using std::array;
using std::ifstream;

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template<typename T>
struct DenseWeight{
    const T* kernel{};
    const T* bias{};
};

template<typename T>
struct LayerNormWeight{
    const T* gamma{};
    const T* beta{};
};

template<typename T>
struct AttentionWeight{
    DenseWeight<T> query_weight;
    DenseWeight<T> key_weight;
    DenseWeight<T> value_weight;
    DenseWeight<T> attention_output_weight;
};

template<typename T>
struct FFNWeight{
    DenseWeight<T> intermediate_weight;
    DenseWeight<T> output_weight;
};

template <typename T>
class DecodingInitParam {
public:
  /* weights for masked_multi_head_attention */
  const T *embedding_table{};
  const T *embedding_kernel{};
  const T *embedding_bias{};

  // Used for unilm.
  const T *trans_kernel{};
  const T *trans_bias{};

  const T *memory_tensor{};
  const int32_t *type_id{};
  const int32_t *memory_sequence_length{};

  // Used for force decoding.
  /// const int32_t *trg_word = nullptr; // TODO: temporarily unseles
  /// const int32_t *trg_length = nullptr;

  const T *position_encoding_table{};

  // segment table
  const T *type_table{};

  LayerNormWeight<T> layernorm;

  const T *logits_mask{};
  int32_t *output_ids{};
  int32_t *parent_ids{};
  int32_t *sequence_length{};
};

template <typename T>
class DecoderInitParam {
public:

  /* weights for masked_multi_head_attention */
  LayerNormWeight<T> self_layernorm;
  AttentionWeight<T> self_attention;

  LayerNormWeight<T> cross_layernorm;
  AttentionWeight<T> cross_attention;

  LayerNormWeight<T> ffn_layernorm;
  FFNWeight<T> ffn;

  int32_t request_batch_size = -1;
  int32_t request_max_mem_seq_len = -1;

  //const float *k_cache = nullptr;
  //const float *v_cache = nullptr;
};

struct TransformerArguments {
  int32_t batch_size_;
  int32_t seq_len_;
  int32_t head_num_;
  int32_t size_per_head_;
  int32_t hidden_units_;
};

struct DecodingArguments : public TransformerArguments {
  int32_t decoder_layers_;
  int32_t vocab_size_;
  int32_t start_id_;
  int32_t end_id_;
  int32_t vocab_size_padded_;
};

struct DecodingBeamsearchArguments : public DecodingArguments{
  int32_t beam_width_;
  int32_t temp_storage_size_;
  float beam_search_diversity_rate_;
  float alpha_;  // power number for length penalty in beam search v2
  bool normalization_before_{true};
  int32_t pos_offset_{0};     // For BART position embedding
  bool pos_bias_{false};  // For Unified position embedding
  // ActivationType act_{ActivationType::RELU};

  int32_t memory_max_seq_len_{0};
  bool prefix_lm_{false};
  int32_t finished_candidate_num_{-1};
  bool early_stopping_{false};
  bool is_mbart_{false};
};

struct TensorParallelParam {
  int32_t local_head_num_{0};
  int32_t local_hidden_units_{0};
};

template <typename T>
struct TopKFinish {
  T u;
  int32_t idx;
  int32_t len;
};



void FusionUnifiedDecodingCompute::PrepareForRun() {
  auto& ctx = this->ctx_->As<XPUContext>();
  int32_t maxptr_size = xdnn::get_max_ptr_size(ctx.GetRawContext());
  input_max_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(maxptr_size * sizeof(float));
  weight_max_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(maxptr_size * sizeof(float));
  output_max_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(maxptr_size * sizeof(float));
  return;
}

void FusionUnifiedDecodingCompute::RunDecodingForward() {
  // auto& param = this->Param<param_t>();
  // auto& ctx = this->ctx_->As<XPUContext>();
  // auto* xpu_ctx = ctx.GetRawContext();
  cout << "A GOOD START" << endl;

  // param.output_ids_->Resize(output_dims);
  // param.output_ids_->mutable_data<int32_t>(TARGET(kHost));
  // param.parent_ids_->Resize(parent_ids_dims);
  // param.parent_ids_->mutable_data<int32_t>(TARGET(kHost));
  // param.sequence_length_->Resize(sequence_length_dims);
  // param.sequence_length_->mutable_data<int32_t>(TARGET(kHost));
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
    .BindInput("AttnMask", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("DecPositionIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("DecRoleIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("DecTypeIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("DecoderLayernormBias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("DecoderLayernormWeight", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("EmbBias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("EmbWeight", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("FFNInterBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("FFNInterWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("FFNLayernormBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("FFNLayernormWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("FFNOutBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("FFNOutWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("InputIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("LMLayernormBias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("LMLayernormWeight", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("LogitsMask", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("MemSeqLen", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("PositionEncEmb", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("PositionIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("RoleEmbedding", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("RoleIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("SelfKeyBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("SelfKeyWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("SelfLayernormBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("SelfLayernormWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("SelfOutBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("SelfOutWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("SelfQueryBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("SelfQueryWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("SelfValueBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("SelfValueWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("TransBias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("TransWeight", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFP16))})
    .BindInput("TypeEmb", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFP16))})
    .BindInput("TypeIds", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("WordEmbedding", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFP16))})
    .BindOutput("OutputIds", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("OutputScores", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("ParentIds", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("SequenceLength", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
