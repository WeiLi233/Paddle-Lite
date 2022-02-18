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
#include <fstream>
#include <memory>
#include "lite/kernels/xpu/fusion_decoding_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template<typename T>
struct DenseWeight{
    const T* kernel = nullptr;
    const T* bias = nullptr;
};

template<typename T>
struct LayerNormWeight{
    const T* gamma = nullptr;
    const T* beta = nullptr;
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
  const T *embedding_table = nullptr;
  const T *embedding_kernel = nullptr;
  const T *embedding_bias = nullptr;

  // Used for unilm.
  const T *trans_kernel = nullptr;
  const T *trans_bias = nullptr;

  const T *memory_tensor = nullptr;
  const int32_t *type_id = nullptr;
  const int32_t *memory_sequence_length = nullptr;

  // Used for force decoding.
  const int32_t *trg_word = nullptr;
  const int32_t *trg_length = nullptr;

  const T *position_encoding_table = nullptr;

  // segment table
  const T *type_table = nullptr;

  LayerNormWeight<T> layernorm;
  LayerNormWeight<T> lm_layernorm;
  LayerNormWeight<T> mbart_layernorm;

  const T *logits_mask = nullptr;

  int32_t *output_ids = nullptr;
  int32_t *parent_ids = nullptr;
  int32_t *sequence_length = nullptr;
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
  /*
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;
  cudaStream_t stream;
  */

  int request_batch_size = -1;
  int request_max_mem_seq_len = -1;

  const float *k_cache = nullptr;
  const float *v_cache = nullptr;
};

struct TransformerArguments {
  size_t batch_size_;
  size_t seq_len_;
  size_t head_num_;
  size_t size_per_head_;
  size_t hidden_units_;
};

struct DecodingArguments : public TransformerArguments {
  int decoder_layers_;
  int vocab_size_;
  int start_id_;
  int end_id_;
  int vocab_size_padded_;
};

struct DecodingBeamsearchArguments : public DecodingArguments{
  int beam_width_;
  int temp_storage_size_;
  float beam_search_diversity_rate_;
  float alpha_;  // power number for length penalty in beam search v2
  bool normalization_before_{true};
  int pos_offset_{0};     // For BART position embedding
  bool pos_bias_{false};  // For Unified position embedding
  // ActivationType act_{ActivationType::RELU};

  int memory_max_seq_len_{0};
  bool prefix_lm_{false};
  int finished_candidate_num_{-1};
  bool early_stopping_{false};
  bool is_mbart_{false};
};

template <typename T>
static inline std::vector<const T*> prepare_weight(
    const std::vector<const lite::Tensor*>& tensor_weight) {
  std::vector<const T*> xdnn_weight;
  for (auto* t: tensor_weight) {
    xdnn_weight.push_back(t->data<float>());
  }
  return xdnn_weight;
}


template<typename T>
class DecodingBeamsearch {
private:
  DecodingBeamsearchArguments args_;
  baidu::xpu::api::Context* ctx_;
  bool is_fuse_topk_softMax_;
  bool keep_alive_beam_;

  T **K_cache_;
  T **V_cache_;
  T **K_mem_cache_;
  T **V_mem_cache_;
  T *from_tensor_[2];
  T *decoder_buf_;

public:
  DecodingBeamsearch& operator=(const DecodingBeamsearch& ) = delete;

  DecodingBeamsearch& operator=(DecodingBeamsearch&& ) = delete;

  DecodingBeamsearch(baidu::xpu::api::Context *ctx,
                     const int32_t batch_size,
                     const int32_t beam_width,
                     const int32_t seq_len,
                     const int32_t head_num,
                     const int32_t size_per_head,
                     const int32_t vocab_size,
                     const int32_t decoder_layers,
                     const int32_t memory_hidden_units,
                     const int32_t memory_max_seq_len,
                     const int32_t start_id,
                     const int32_t end_id,
                     const float beam_search_diversity_rate = -0.0f,
                     const bool is_fuse_topk_softMax = false,
                     const bool is_fuse_qkv = false,
                     const bool keep_alive_beam = false,
                     const float alpha = 0.6,
                     const bool normalization_before = true,
                     const int32_t pos_offset = 0,
                     const bool pos_bias = false,
                     const bool prefix_lm = false,
                     const int32_t finished_candidate_num = -1,
                     const bool early_stopping = false,
                     const bool is_mbart = false)
      : ctx_(ctx),
        is_fuse_topk_softMax_(is_fuse_topk_softMax),
        keep_alive_beam_(keep_alive_beam) {
    args_.batch_size_ = batch_size;
    args_.beam_width_ = beam_width;
    args_.seq_len_ = seq_len;
    args_.memory_max_seq_len_ = memory_max_seq_len;
    args_.head_num_ = head_num;
    args_.size_per_head_ = size_per_head;
    args_.hidden_units_ = head_num * size_per_head;
    args_.decoder_layers_ = decoder_layers;
    args_.vocab_size_ = vocab_size;
    args_.start_id_ = start_id;
    args_.end_id_ = end_id;
    args_.beam_search_diversity_rate_ = beam_search_diversity_rate;
    //if (args_.beam_width_ > 16 || args_.beam_width_ > MAX_K)
    //  CHECK(false) << "Not Suported!";
    if (std::is_same<T, float>::value)
      args_.vocab_size_padded_ = vocab_size;
    else 
      CHECK(false) << "half not supported now";
    //if (std::is_same<T, half>::value)
    //  args_.vocab_size_padded_ = (int)(ceil(vocab_size / 8.)) * 8;

    args_.alpha_ = alpha;
    args_.normalization_before_ = normalization_before;
    args_.pos_offset_ = pos_offset;
    args_.pos_bias_ = pos_bias;
    // args_.act_ = act;

    args_.prefix_lm_ = prefix_lm;
    args_.is_mbart_ = is_mbart;

    args_.finished_candidate_num_ = (finished_candidate_num == -1)
                                        ? beam_width * 2
                                        : finished_candidate_num;
    args_.early_stopping_ = early_stopping;
    // TODO: OpenDecoder

    size_t from_tensor_size =
        args_.batch_size_ * args_.beam_width_ * args_.hidden_units_;  // type T
    /// TODO size_t decoder_workspace_size = decoder_->getWorkspaceSize();     // type T
    size_t decoder_normed_result_buffer_size =
        args_.batch_size_ * args_.beam_width_ * args_.hidden_units_;  // type T
    size_t cache_size = (prefix_lm)
                            ? (args_.batch_size_ * args_.beam_width_ *
                               (args_.seq_len_ + args_.memory_max_seq_len_) *
                               args_.hidden_units_)
                            : (args_.batch_size_ * args_.beam_width_ *
                               args_.seq_len_ * args_.hidden_units_);  // type T
    size_t mem_cache_size =
        (prefix_lm) ? 0 : (args_.batch_size_ * args_.beam_width_ *
                           memory_max_seq_len * args_.hidden_units_);  // type T

    size_t logits_buf_size = args_.batch_size_ * args_.beam_width_ *
                             args_.vocab_size_padded_;  // type float
    size_t cum_log_buf_size =
        args_.batch_size_ * args_.beam_width_;  // type float
    size_t word_ids_buf_size =
        args_.batch_size_ * args_.beam_width_;  // type int
    size_t parent_ids_buf_size =
        keep_alive_beam_ ? word_ids_buf_size : 0;  // type int
    size_t finished_buf_size =
        args_.batch_size_ * args_.beam_width_;  // type bool
    size_t alive_finished_buf_size = keep_alive_beam_ ? finished_buf_size : 0;
    size_t finished_count_size = (size_t)(ceil(1 / 32.)) * 32;  // type int
    /*
    size_t storage_size_per_beam =
        2 * args_.beam_width_ +
        SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K + 2);
    args_.temp_storage_size_ = args_.batch_size_ * args_.beam_width_ *
                               storage_size_per_beam;  // type float
    args_.temp_storage_size_ = (size_t)(
        ceil(args_.batch_size_ * args_.beam_width_ * args_.beam_width_ / 4.) *
            4 * 2 +
        ceil(args_.batch_size_ * args_.beam_width_ *
             SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K + 2) / 4.) *
            4);
    */
    size_t padded_embedding_kernel_size =
        args_.hidden_units_ * args_.vocab_size_padded_;
    size_t padded_embedding_bias_size = args_.vocab_size_padded_;
    if (args_.vocab_size_padded_ == args_.vocab_size_)) {
      padded_embedding_kernel_size = 0;
      padded_embedding_bias_size = 0;
    }


        // When using separated alive and finish beam queues, some buffers size need
    // to be doubled to restore beam search intermedia results of both alive and
    // finish beams.
    if (keep_alive_beam_ == true) {
      // cumulated log-probs of finish beams and alive beams
      cum_log_buf_size += cum_log_buf_size;
      finished_buf_size += finished_buf_size;
      // Double the size of topk_tmp_id_buf, topk_tmp_val_buf, since we need
      // select the top 2*beam_width.
      args_.temp_storage_size_ +=
          ceil(args_.batch_size_ * args_.beam_width_ * args_.beam_width_ / 4.) *
          4 * 2;
    }

    // prevent memory misalinged address
    logits_buf_size = (size_t)(ceil(logits_buf_size / 4.)) * 4;
    cum_log_buf_size = (size_t)(ceil(cum_log_buf_size / 4.)) * 4;
    word_ids_buf_size = (size_t)(ceil(word_ids_buf_size / 4.)) * 4;
    parent_ids_buf_size = (size_t)(ceil(parent_ids_buf_size / 4.)) * 4;
    finished_buf_size = (size_t)(ceil(finished_buf_size / 32.)) * 32;
    alive_finished_buf_size =
        (size_t)(ceil(alive_finished_buf_size / 32.)) * 32;
    const size_t tmp_logits_buf_size = logits_buf_size;

    // get workspace size of topk kernel
    if (keep_alive_beam_ == true) {
      topK_update_kernelLauncher(topK_kernel_workspace,
                                 topk_workspace_size_,
                                 logits_buf_,
                                 finished_buf_,
                                 alive_finished_buf_,
                                 nullptr,
                                 word_ids_buf_,
                                 parent_ids_buf_,
                                 nullptr,
                                 nullptr,
                                 cum_log_buf_,
                                 0,
                                 args_,
                                 0);
    } else {
      topK_kernelLauncher(topK_kernel_workspace,
                          topk_workspace_size_,
                          logits_buf_,
                          word_ids_buf_,
                          finished_buf_,
                          args_,
                          0);
    }
  }
  
  virtual ~DecodingBeamsearch() {
    /*
    delete[] K_cache_;
    delete[] V_cache_;
    delete[] K_mem_cache_;
    delete[] V_mem_cache_;
    delete[] h_finished_buf_;
    delete[] h_trg_length_;
    delete decoder_;
    allocator_.free(buf_);
    */
  }
};

template <typename T>
static void DecodingKernel(
    baidu::xpu::api::Context *ctx,
    const lite::Tensor* input,
    const lite::Tensor* mem_seq_len,
    const lite::Tensor* word_embedding,
    const std::vector<const lite::Tensor*>& self_ln_weight,
    const std::vector<const lite::Tensor*>& self_ln_bias,
    const std::vector<const lite::Tensor*>& self_q_weight,
    const std::vector<const lite::Tensor*>& self_q_bias,
    const std::vector<const lite::Tensor*>& self_k_weight,
    const std::vector<const lite::Tensor*>& self_k_bias,
    const std::vector<const lite::Tensor*>& self_v_weight,
    const std::vector<const lite::Tensor*>& self_v_bias,
    const std::vector<const lite::Tensor*>& self_out_weight,
    const std::vector<const lite::Tensor*>& self_out_bias,
    const std::vector<const lite::Tensor*>& cross_ln_weight,
    const std::vector<const lite::Tensor*>& cross_ln_bias,
    const std::vector<const lite::Tensor*>& cross_q_weight,
    const std::vector<const lite::Tensor*>& cross_q_bias,
    const std::vector<const lite::Tensor*>& cross_k_weight,
    const std::vector<const lite::Tensor*>& cross_k_bias,
    const std::vector<const lite::Tensor*>& cross_v_weight,
    const std::vector<const lite::Tensor*>& cross_v_bias,
    const std::vector<const lite::Tensor*>& cross_out_weight,
    const std::vector<const lite::Tensor*>& cross_out_bias,
    const std::vector<const lite::Tensor*>& ffn_ln_weight,
    const std::vector<const lite::Tensor*>& ffn_ln_bias,
    const std::vector<const lite::Tensor*>& ffn_inter_weight,
    const std::vector<const lite::Tensor*>& ffn_inter_bias,
    const std::vector<const lite::Tensor*>& ffn_out_weight,
    const std::vector<const lite::Tensor*>& ffn_out_bias,
    const lite::Tensor* decoder_ln_weight,
    const lite::Tensor* decoder_ln_bias,
    const lite::Tensor* embedding_weight,
    const lite::Tensor* embedding_bias,
    const lite::Tensor* positional_embedding_weight,
    lite::Tensor* output_ids,
    lite::Tensor* parent_ids,
    lite::Tensor* sequence_length,
    const std::string& decoding_strategy,
    const int32_t beam_size,
    const int32_t topk,
    const float topp,
    const int32_t n_head,
    const int32_t size_per_head,
    const int32_t num_layer,
    const int32_t bos_id,
    const int32_t eos_id,
    const int64_t max_len,
    const float beam_search_diversity_rate,
    const float alpha) {
  CHECK_EQ(input->numel(), 32*39*1024) << "input dimmension mismatch!";
  vector<float> cpu_input(32*39*1024);
  vector<int32_t> cpu_mem_seq_len(32);
  ifstream ifs("../debug_tensor_bin");
  if(! ifs.is_open()) {
    throw std::runtime_error("can not open debug_bin_tensor");
  }
  ifs.read(reinterpret_cast<char*>(cpu_input.data()), 32*39*1024*sizeof(float));
  ifs.read(reinterpret_cast<char*>(cpu_mem_seq_len.data()), 32*sizeof(int32_t));
  ifs.close();

  TargetWrapperXPU::MemcpySync(
            const_cast<lite::Tensor*>(input)->mutable_data(TARGET(kXPU), input->memory_size()), 
            cpu_input.data(), 
            32*39*1024*sizeof(float), 
            IoDirection::HtoD);
  TargetWrapperXPU::MemcpySync(
           const_cast<lite::Tensor*>(mem_seq_len)->mutable_data(TARGET(kXPU), mem_seq_len->memory_size()), 
           cpu_mem_seq_len.data(), 
           32*sizeof(int32_t), 
           IoDirection::HtoD);

  const int32_t beam_width = (decoding_strategy == "beam_search" ||
                    decoding_strategy == "beam_search_v2") ? 
                    beam_size : 1;
  /*
  const int32_t candidate_num = (decoding_strategy == "topk_sampling" ||
                    decoding_strategy == "topp_sampling") ? 
                    topk : 1;
  const float probability_threshold = (decoding_strategy == "topk_sampling" ||
                    decoding_strategy == "topp_sampling") ? 
                    topp : 0.0;
  */
  auto input_dims = input->dims();
  const int32_t batch_size = (decoding_strategy == "beam_search" ||
                     decoding_strategy == "beam_search_v2")
                        ? input_dims[0] / beam_width
                        : input_dims[0];
  const int32_t memory_max_seq_len = input_dims[1];
  const int32_t memory_hidden_dim = input_dims[2];
  const int vocab_size = word_embedding->dims()[0];
  
  DecodingInitParam<float> decoding_params;
  decoding_params.output_ids = output_ids->mutable_data<int32_t>(
                                  TARGET(kXPU), output_ids->memory_size());
  decoding_params.parent_ids = parent_ids->mutable_data<int32_t>(
                                  TARGET(kXPU), parent_ids->memory_size());
  decoding_params.sequence_length = sequence_length->mutable_data<int32_t>(
                                  TARGET(kXPU), sequence_length->memory_size());
  decoding_params.memory_tensor = input->data<float>();
  decoding_params.memory_sequence_length = mem_seq_len->data<int32_t>();

  vector<DecoderInitParam<float>> params(num_layer);
  auto q_weight_dims = self_q_weight[0]->dims();
  auto k_weight_dims = self_k_weight[0]->dims();
  bool fuse_qkv = (q_weight_dims[1] == k_weight_dims[1]) ? false : true;

  for(int32_t i=0; i<num_layer; i++) {
    if (decoding_strategy == "beam_search" || decoding_strategy == "beam_search_v2") {
      params[i].request_batch_size = batch_size * beam_width;
      params[i].request_max_mem_seq_len = memory_max_seq_len;
    } else if (decoding_strategy == "sampling" ||
               decoding_strategy == "topk_sampling" ||
               decoding_strategy == "topp_sampling") {
      CHECK(false) << "not supported now";
    }

    // self attn
    params[i].self_layernorm.gamma = self_ln_weight[i]->data<float>();
    params[i].self_layernorm.beta = self_ln_bias[i]->data<float>();
    // query
    params[i].self_attention.query_weight.kernel = self_q_weight[i]->data<float>();
    params[i].self_attention.query_weight.bias = self_q_bias[i]->data<float>();
    // key
    params[i].self_attention.key_weight.kernel = self_k_weight[i]->data<float>();
    params[i].self_attention.key_weight.bias = self_k_bias[i]->data<float>();
    // value
    params[i].self_attention.value_weight.kernel = self_v_weight[i]->data<float>();
    params[i].self_attention.value_weight.bias = self_v_bias[i]->data<float>();
    // out proj
    params[i].self_attention.attention_output_weight.kernel = self_out_weight[i]->data<float>();
    params[i].self_attention.attention_output_weight.bias = self_out_bias[i]->data<float>();
    // cross
    params[i].cross_layernorm.gamma = cross_ln_weight[i]->data<float>();
    params[i].cross_layernorm.beta = cross_ln_bias[i]->data<float>();
    // query
    params[i].cross_attention.query_weight.kernel = cross_q_weight[i]->data<float>();
    params[i].cross_attention.query_weight.bias = cross_q_bias[i]->data<float>();
    // key
    params[i].cross_attention.key_weight.kernel = cross_k_weight[i]->data<float>();
    params[i].cross_attention.key_weight.bias = cross_k_bias[i]->data<float>();
    // value
    params[i].cross_attention.value_weight.kernel = cross_v_weight[i]->data<float>();
    params[i].cross_attention.value_weight.bias = cross_v_bias[i]->data<float>();
    // out proj
    params[i].cross_attention.attention_output_weight.kernel = cross_out_weight[i]->data<float>();
    params[i].cross_attention.attention_output_weight.bias = cross_out_bias[i]->data<float>();
    // ffn
    params[i].ffn_layernorm.gamma = ffn_ln_weight[i]->data<float>();
    params[i].ffn_layernorm.beta = ffn_ln_bias[i]->data<float>();
    // intermediate proj
    params[i].ffn.intermediate_weight.kernel = ffn_inter_weight[i]->data<float>();
    params[i].ffn.intermediate_weight.bias = ffn_inter_bias[i]->data<float>();
    // out proj
    params[i].ffn.output_weight.kernel = ffn_out_weight[i]->data<float>();
    params[i].ffn.output_weight.bias = ffn_out_bias[i]->data<float>();
  }

  decoding_params.layernorm.gamma = decoder_ln_weight->data<float>();
  decoding_params.layernorm.beta = decoder_ln_bias->data<float>();
  // for embedding
  decoding_params.embedding_table = word_embedding->data<float>();

  // for weight sharing matmul
  decoding_params.embedding_kernel = embedding_weight->data<float>();
  // for matmul bias
  decoding_params.embedding_bias = embedding_bias->data<float>();

  decoding_params.position_encoding_table = positional_embedding_weight->data<float>();

  if(decoding_strategy == "beam_search_v2") {
    auto decoding_beam_search = std::unique_ptr<DecodingBeamsearch<float>>(
      new DecodingBeamsearch<float>(
          ctx,
          batch_size,
          beam_width,
          max_len,
          n_head,
          size_per_head,
          vocab_size,
          num_layer,
          memory_hidden_dim,
          memory_max_seq_len,
          bos_id,
          eos_id,
          beam_search_diversity_rate,
          true,   // is_fuse_topk_softMax
          fuse_qkv,
          true,   // keep_alive_beam
          alpha));
  } else {
    CHECK(false) << "not supported now";
  }
  



  return;
}

void FusionDecodingCompute::PrepareForRun() {
  /*
  auto& param = this->Param<param_t>();
  arg_cross_key_bias_ = prepare_weight<float>(param.cross_key_bias_);
  arg_cross_key_weight_ = prepare_weight<float>(param.cross_key_weight_); 
  arg_cross_layernorm_bias_ = prepare_weight<float>(param.cross_layernorm_bias_);
  arg_cross_layernorm_weight_ = prepare_weight<float>(param.cross_layernorm_weight_);
  arg_cross_out_bias_ = prepare_weight<float>(param.cross_out_bias_);
  arg_cross_out_weight_ = prepare_weight<float>(param.cross_out_weight_);
  arg_cross_query_bias_ = prepare_weight<float>(param.cross_query_bias_);
  arg_cross_query_weight_ = prepare_weight<float>(param.cross_query_weight_);
  arg_cross_value_bias_ = prepare_weight<float>(param.cross_value_bias_);
  arg_cross_value_weight_ = prepare_weight<float>(param.cross_value_weight_);
  arg_ffn_inter_bias_ = prepare_weight<float>(param.ffn_inter_bias_);
  arg_ffn_inter_weight_ = prepare_weight<float>(param.ffn_inter_weight_);
  arg_ffn_layernorm_bias_ = prepare_weight<float>(param.ffn_layernorm_bias_);
  arg_ffn_layernorm_weight_ = prepare_weight<float>(param.ffn_layernorm_weight_);
  arg_ffn_out_bias_ = prepare_weight<float>(param.ffn_out_bias_);
  arg_ffn_out_weight_ = prepare_weight<float>(param.ffn_out_weight_);
  arg_self_key_bias_ = prepare_weight<float>(param.self_key_bias_);
  arg_self_key_weight_ = prepare_weight<float>(param.self_key_weight_);
  arg_self_layernorm_bias_ = prepare_weight<float>(param.self_layernorm_bias_);
  arg_self_layernorm_weight_ = prepare_weight<float>(param.self_layernorm_weight_);
  arg_self_out_bias_ = prepare_weight<float>(param.self_out_bias_);
  arg_self_out_weight_ = prepare_weight<float>(param.self_out_weight_);
  arg_self_query_bias_ = prepare_weight<float>(param.self_query_bias_);
  arg_self_query_weight_ = prepare_weight<float>(param.self_query_weight_);
  arg_self_value_bias_ = prepare_weight<float>(param.self_value_bias_);
  arg_self_value_weight_ = prepare_weight<float>(param.self_value_weight_);
  */
  return;
}

void FusionDecodingCompute::RunDecodingForward() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  auto* xpu_ctx = ctx.GetRawContext();

  int32_t batch_size = param.input_->dims()[0];
  int32_t max_out_len = param.rel_len_ ?  param.max_len_ + param.input_->dims()[1] : param.max_len_;

  std::vector<int64_t> output_dims;
  std::vector<int64_t> parent_ids_dims;
  std::vector<int64_t> sequence_length_dims({batch_size});
  if (param.decoding_strategy_ == "beam_search") {
    CHECK(false) << "\"beam_search\" not supported! ";
  } else if (param.decoding_strategy_ == "beam_search_v2") {
    // Use separated alive and finish beam queues to avoid the decrease of alive
    // beams. The outputs must include both the finish and alive to trace full
    // path.
    if (batch_size != -1) {
      sequence_length_dims = {batch_size * 2};
      batch_size /= param.beam_size_;
    } else {
      sequence_length_dims = {batch_size};
    }
    output_dims = {max_out_len, batch_size, param.beam_size_ * 2};
    parent_ids_dims = output_dims;
  } else if (param.decoding_strategy_ == "topk_sampling" ||
             param.decoding_strategy_ == "topp_sampling") {
    CHECK(false) << "\"topk_sampling\" or \"topp_sampling\" not supported! "; 
  } else {
    CHECK(false) << "Not supported decoding strategy. ";
  }

  param.output_ids_->Resize(output_dims);
  param.output_ids_->mutable_data(TARGET(kXPU), param.output_ids_->dims().production()*sizeof(int32_t));
  param.parent_ids_->Resize(parent_ids_dims);
  param.parent_ids_->mutable_data(TARGET(kXPU), param.parent_ids_->dims().production()*sizeof(int32_t));
  param.sequence_length_->Resize(sequence_length_dims);
  param.sequence_length_->mutable_data(TARGET(kXPU), param.sequence_length_->dims().production()*sizeof(int32_t));

  DecodingKernel<float>(
    xpu_ctx,
    param.input_,
    param.mem_seq_len_,
    param.word_embedding_,
    param.self_layernorm_weight_,
    param.self_layernorm_bias_,
    param.self_query_weight_,
    param.self_query_bias_,
    param.self_key_weight_,
    param.self_key_bias_,
    param.self_value_weight_,
    param.self_value_bias_,
    param.self_out_weight_,
    param.self_out_bias_,
    param.cross_layernorm_weight_,
    param.cross_layernorm_bias_,
    param.cross_query_weight_,
    param.cross_query_bias_,
    param.cross_key_weight_,
    param.cross_key_bias_,
    param.cross_value_weight_,
    param.cross_value_bias_,
    param.cross_out_weight_,
    param.cross_out_bias_,
    param.ffn_layernorm_weight_,
    param.ffn_layernorm_bias_,
    param.ffn_inter_weight_,
    param.ffn_inter_bias_,
    param.ffn_out_weight_,
    param.ffn_out_bias_,
    param.decoder_layernorm_weight_,
    param.decoder_layernorm_bias_,
    param.emb_weight_,
    param.emb_bias_,
    param.position_enc_emb_,
    param.output_ids_,
    param.parent_ids_,
    param.sequence_length_,
    param.decoding_strategy_,
    param.beam_size_,
    param.topk_,
    param.topp_,
    param.n_head_,
    param.size_per_head_,
    param.num_layer_,
    param.bos_id_,
    param.eos_id_,
    param.max_len_,
    param.beam_search_diversity_rate_,
    param.alpha_
  );

  return;
}

void FusionDecodingCompute::Run() {
  this->RunDecodingForward();
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
    .BindInput("CrossKeyWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossLayernormBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossLayernormWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossOutBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossOutWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossQueryBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossQueryWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossValueBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("CrossValueWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("DecoderLayernormBias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("DecoderLayernormWeight", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("EmbBias", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("EmbWeight", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNInterBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNInterWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNLayernormBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNLayernormWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNOutBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("FFNOutWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("PositionEncEmb", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfKeyBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfKeyWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfLayernormBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfLayernormWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfOutBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfOutWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfQueryBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfQueryWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfValueBias@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("SelfValueWeight@VECTOR", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("WordEmbedding", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("OutputIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("ParentIds", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("SequenceLength", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();
