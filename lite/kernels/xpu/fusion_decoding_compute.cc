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
#include "lite/kernels/xpu/fusion_decoding_compute.h"
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

static constexpr int32_t MAX_K = 4;
static constexpr int32_t SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS = 128;
static constexpr float PLITE_FLT_MIN = -1e20;

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


template <typename T>
class OpenDecoder {
  baidu::xpu::api::Context *ctx_;
  const DecoderInitParam<T> *param_;

  int32_t max_batch_size_ = -1;
  int32_t head_num_{};
  int32_t size_per_head_{};
  int32_t hidden_units_{};
  int32_t memory_hidden_units_{};
  int32_t od_mem_len_{}; // od is short for OpenDecoder
  int32_t od_seq_len_{};
  bool is_fuse_QKV_in_batched_gemm_{};
  bool is_fuse_QKV_in_normal_gemm_{};
  bool normalization_before_{};
  /// ActivationType act_;

  T *norm_from_tensor_buf_{}, *query_buf_{};
  T *context_buf_{}, *masked_output_buf_{};
  T *norm_masked_output_buf_{}, *cross_output_buf_{};
  T *norm_cross_output_buf_{}, *ffn_inner_buf_{};
  // T *ffn_out_buf_{};
  T *key_buf_{}, *value_buf_{}; //useless...
  T *sliced_q_{}, *sliced_k_{}, *sliced_v_{}; // [max_batch, hidden_dim]
  T *broadcast_q_{}; // [max_batch, max_seq, hidden_dim]
  T *transpose_kv_buf_{}; // [max_batch, max_seq, hidden_dim]
  T *qk_buffer_{}; // [max_batch, head_num, 1, max_seq], self 和cross 可以共用

  float *self_mask_{}; // [max_batch, max_seq]
  float *cross_mask_{}; // [max_batch, max_mem_seq]
  // self_mask_只是为了适配qk_attention api，fill all 0就可以，但cross_mask_需要根据输入更新，因此不能复用

public:
  OpenDecoder(baidu::xpu::api::Context *ctx,
              int32_t head_num,
              int32_t size_per_head,
              int32_t memory_hidden_units,
              bool is_fuse_QKV_in_normal_gemm = false,
              bool normalization_before = true)
      : ctx_(ctx),
        head_num_(head_num),
        size_per_head_(size_per_head),
        hidden_units_(head_num * size_per_head),
        memory_hidden_units_(memory_hidden_units),
        is_fuse_QKV_in_normal_gemm_(is_fuse_QKV_in_normal_gemm),
        normalization_before_(normalization_before) {};
  
  inline void set_max_batch_size(const int32_t batch_size) {
    max_batch_size_ = batch_size;
  }
  inline void set_max_mem_length(const int32_t memory_length) {
    od_mem_len_ = memory_length;
  }
  inline void set_max_seq_length(const int32_t sequence_length) {
    od_seq_len_ = sequence_length;
  }

  int32_t getWorkspaceSize() {
    CHECK_NE(od_mem_len_, 0) << "od_mem_len_ can not be 0, bug occurs.";
    CHECK_NE(od_seq_len_, 0) << "od_seq_len_ can not be 0, bug occurs.";
    // Check OpenDecoder::initialize() to see why magic number 13 appears
    // return 13 * max_batch_size_ * hidden_units_ + sizeof(T*) * 9;
    return (13 + 3 + od_seq_len_* 2) * max_batch_size_ * hidden_units_ 
            + ((head_num_ + 1) * od_seq_len_ + od_mem_len_) * max_batch_size_;
    
  }

  void initialize(const DecoderInitParam<T>& param, T* buf) {
    param_ = &param;
    const int32_t buf_size = max_batch_size_ * hidden_units_;

    norm_from_tensor_buf_ = buf;
    // ffn_out_buf_          = buf;
    query_buf_            = buf + buf_size;  // store the query values (from_tensor * Q) in
                                  // both masked and multi-head attention
    key_buf_      = query_buf_ + buf_size;
    value_buf_    = key_buf_ + buf_size;
    context_buf_  = value_buf_ + buf_size;  // store the context result
                                           // (softmax(qk)v) in both masked and
                                           // multi-head attention

    masked_output_buf_      = context_buf_ + buf_size;  // masked_attention_output
    norm_masked_output_buf_ = masked_output_buf_ + buf_size;  // norm(masked_attention_output)
    cross_output_buf_       = norm_masked_output_buf_ + buf_size;  // mutli-head attention_output
    norm_cross_output_buf_  = cross_output_buf_ + buf_size;  // norm(multi-head attention_output)
    ffn_inner_buf_          = norm_cross_output_buf_ + buf_size;  // 4 buf size to store inner product
    
    sliced_q_               = ffn_inner_buf_ + 4 * buf_size;
    sliced_k_               = sliced_q_ + buf_size;
    sliced_v_               = sliced_k_ + buf_size;
    broadcast_q_            = sliced_v_ + buf_size;
    transpose_kv_buf_       = broadcast_q_ + buf_size * od_seq_len_;
    qk_buffer_              = transpose_kv_buf_ + buf_size * od_seq_len_;
    self_mask_              = qk_buffer_ + max_batch_size_ * head_num_ * od_seq_len_;
    cross_mask_             = self_mask_ + max_batch_size_ * od_seq_len_;
  }

  void forward(const T* from_tensor,
               const T* memory_tensor,
               T* key_cache_,
               T* value_cache_,
               T* key_mem_cache_,
               T* value_mem_cache_,
               const int* memory_sequence_length,
               T* decoder_output,
               const int32_t step,
               const int32_t decoder_max_seq_len,
               const bool is_cross_attention,
               const bool* finished = nullptr,
               const int32_t memory_max_seq_len = -1) {
    /*
    if(step == 2) {
        const int32_t dbg_len = max_batch_size_ * 1024;
        vector<float> mask_cpu(dbg_len);
        TargetWrapperXPU::MemcpySync(
            mask_cpu.data(), from_tensor, dbg_len*sizeof(float), IoDirection::DtoH); 
        std::cout << "from tensor step = " << step << std::endl;
        for(int32_t x=0; x<dbg_len; x+=200) {
          std::cout << x << '\t' << mask_cpu[x] << std::endl;
        }
        std::cout << std::endl;
    }
    */
    int32_t xdnn_ret;
    xdnn_ret = xdnn::layer_norm<float>(ctx_,    
                           from_tensor, 
                           norm_from_tensor_buf_,
                           max_batch_size_,
                           hidden_units_,
                           1e-6f,
                           param_->self_layernorm.gamma,
                           param_->self_layernorm.beta,
                           nullptr,
                           nullptr);
    CHECK_EQ(xdnn_ret, 0) << "Error in layernorm";
    if (memory_max_seq_len == -1) {
          masked_multi_head_attention(norm_from_tensor_buf_,
                                      key_cache_,
                                      value_cache_,
                                      masked_output_buf_,
                                      finished,
                                      step,
                                      decoder_max_seq_len);
    } else {
      CHECK(false) << "Not implemented.";
    }

    if(is_cross_attention) {
      xdnn_ret = xdnn::add<T>(ctx_, 
                          from_tensor, 
                          masked_output_buf_, 
                          masked_output_buf_,
                          max_batch_size_ * hidden_units_);
      xdnn_ret = xdnn::layer_norm<T>(ctx_,    
                           masked_output_buf_, 
                           norm_masked_output_buf_,
                           max_batch_size_,
                           hidden_units_,
                           1e-6f,
                           param_->cross_layernorm.gamma,
                           param_->cross_layernorm.beta,
                           nullptr,
                           nullptr);
      CHECK_EQ(xdnn_ret, 0) << "norm masked layer norm error.";
      /*
      if(step == 1) {
        const int32_t dbg_len = max_batch_size_ * 1024;
        vector<float> mask_cpu(dbg_len);
        TargetWrapperXPU::MemcpySync(
            mask_cpu.data(), norm_masked_output_buf_, dbg_len*sizeof(float), IoDirection::DtoH); 
        std::cout << "norm mask out" << std::endl;
        for(int32_t x=0; x<dbg_len; x+=1000) {
          std::cout << mask_cpu[x] << std::endl;
        }
        std::cout << std::endl;
      }
      */
      cross_multi_head_attention(norm_masked_output_buf_,
                                  memory_tensor,
                                  key_mem_cache_,
                                  value_mem_cache_,
                                  cross_output_buf_,
                                  memory_sequence_length,
                                  finished,
                                  param_->request_max_mem_seq_len,
                                  step);
      xdnn_ret = xdnn::add<T>(ctx_, 
                          masked_output_buf_, 
                          cross_output_buf_, 
                          cross_output_buf_,
                          max_batch_size_ * hidden_units_);
      xdnn_ret = xdnn::layer_norm<T>(ctx_,    
                           cross_output_buf_, 
                           norm_cross_output_buf_,
                           max_batch_size_,
                           hidden_units_,
                           1e-6f,
                           param_->ffn_layernorm.gamma,
                           param_->ffn_layernorm.beta,
                           nullptr,
                           nullptr);
      CHECK_EQ(xdnn_ret, 0) << "norm_cross_output_buf_ layer norm error.";

      // start ffn()
      int32_t m = max_batch_size_;
      int32_t n = hidden_units_;
      xdnn_ret = xdnn::fc_fusion<T, T, T, float>(
          ctx_, /* context */
          norm_cross_output_buf_,          /* x */
          param_->ffn.intermediate_weight.kernel,
          ffn_inner_buf_,                      /* y */
          m,                  /* m */
          n * 4,                        /* n */
          n,                        /* k */
          false,                       /* x_trans */
          false,                        /* w_trans */
          nullptr,              /* x_max */
          nullptr,             /* w_max */
          nullptr,             /* y_max */
          n,                        /* ldx */
          n * 4,                        /* ldw */
          n * 4,                        /* ldy */
          1.0f,                        /* alpha */
          0.0f,                        /* beta */
          param_->ffn.intermediate_weight.bias, /* bias */
          xdnn::Activation_t::RELU); /* act_type */
          CHECK_EQ(xdnn_ret, 0) << "calling ffn fc_fusion error!.";

      xdnn_ret = xdnn::fc_fusion<T, T, T, float>(
          ctx_, /* context */
          ffn_inner_buf_,          /* x */
          param_->ffn.output_weight.kernel,
          decoder_output,                      /* y */
          m,                  /* m */
          n,                        /* n */
          n * 4,                        /* k */
          false,                       /* x_trans */
          false,                        /* w_trans */
          nullptr,              /* x_max */
          nullptr,             /* w_max */
          nullptr,             /* y_max */
          n * 4,                        /* ldx */
          n,                        /* ldw */
          n,                        /* ldy */
          1.0f,                        /* alpha */
          0.0f,                        /* beta */
          param_->ffn.output_weight.bias, /* bias */
          xdnn::Activation_t::LINEAR); /* act_type */
      CHECK_EQ(xdnn_ret, 0) << "calling ffn fc_fusion error!.";

      xdnn_ret = xdnn::add<T>(ctx_, 
                          decoder_output, 
                          cross_output_buf_, 
                          decoder_output,
                          max_batch_size_ * hidden_units_);
      CHECK_EQ(xdnn_ret, 0) << "calling ffn add error!.";
      /*
      if(step == 1) {
        const int32_t dbg_len = max_batch_size_ * 1024;
        vector<float> m_cpu(dbg_len);
        TargetWrapperXPU::MemcpySync(
            m_cpu.data(), decoder_output, dbg_len*sizeof(float), IoDirection::DtoH);
        std::cout << "decoder_output out" << std::endl;
        for(int32_t x=0; x<dbg_len; x+=500) {
          std::cout << m_cpu[x] << ' ';
        }
        std::cout << std::endl;
      }
      */
      // end ffn()
    } else {
      CHECK(false) << "is_cross_attention -> false not implemented.";
    }

  }

  void masked_multi_head_attention(const T *from_tensor,
                                   T *key_cache,
                                   T *value_cache,
                                   T *decoder_output,
                                   const bool *finished,
                                   const int32_t step,
                                   const int32_t max_seq_len) {
    int32_t m = max_batch_size_;
    int32_t n = hidden_units_;

    int32_t xdnn_ret;
    if (is_fuse_QKV_in_normal_gemm_ == true) {
      xdnn_ret = xdnn::fc_fusion<T, T, T, float>(
      ctx_, /* context */
      from_tensor,          /* x */
      param_->self_attention.query_weight.kernel,
      query_buf_,                      /* y */
      m,                  /* m */
      n * 3,                        /* n */
      n,                        /* k */
      false,                       /* x_trans */
      false,                        /* w_trans */
      nullptr,              /* x_max */
      nullptr,             /* w_max */
      nullptr,             /* y_max */
      n,                        /* ldx */
      n * 3,                        /* ldw */
      n * 3,                        /* ldy */
      1.0f,                        /* alpha */
      0.0f,                        /* beta */
      param_->self_attention.query_weight.bias, /* bias */
      xdnn::Activation_t::LINEAR); /* act_type */
      CHECK_EQ(xdnn_ret, 0) << "calling fc_fusion error!.";
      
      xdnn_ret = xdnn::slice<T>(ctx_,
                                query_buf_,
                                sliced_q_,
                                {m, n*3},
                                {0, 0},
                                {m, n});
      CHECK_EQ(xdnn_ret, false) << "slice query error!";
      xdnn_ret = xdnn::slice<T>(ctx_,
                                query_buf_,
                                sliced_k_,
                                {m, n*3},
                                {0, n},
                                {m, n*2});
      CHECK_EQ(xdnn_ret, false) << "slice key error!";
      xdnn_ret = xdnn::slice<T>(ctx_,
                                query_buf_,
                                sliced_v_,
                                {m, n*3},
                                {0, n*2},
                                {m, n*3});
      CHECK_EQ(xdnn_ret, false) << "slice value error!";

      if(step == 1) {
        xdnn_ret = xdnn::copy<T>(ctx_,
                          sliced_k_,
                          key_cache,
                          m*n);
        xdnn_ret = xdnn::copy<T>(ctx_,
                          sliced_v_,
                          value_cache,
                          m*n);
      } else {
        // update K_cache
        xdnn_ret = xdnn::transpose<T>(ctx_,
                            key_cache,
                            transpose_kv_buf_,
                            {m,step-1,n},
                            {1,0,2});
        xdnn_ret = xdnn::copy<T>(ctx_,
                          sliced_k_,
                          transpose_kv_buf_ + m*n*(step-1),
                          m*n);
        xdnn_ret = xdnn::transpose<T>(ctx_,
                            transpose_kv_buf_,
                            key_cache,
                            {step,m,n},
                            {1,0,2});
        // update V_cache
        xdnn_ret = xdnn::transpose<T>(ctx_,
                            value_cache,
                            transpose_kv_buf_,
                            {m,step-1,n},
                            {1,0,2});
        xdnn_ret = xdnn::copy<T>(ctx_,
                          sliced_v_,
                          transpose_kv_buf_ + m*n*(step-1),
                          m*n);
        xdnn_ret = xdnn::transpose<T>(ctx_,
                            transpose_kv_buf_,
                            value_cache,
                            {step,m,n},
                            {1,0,2});
      }
      /*
      if(step == 2) {
        cout << "V_CACHE STEP = " << step << endl;
        vector<float> tmp_v_cache_cpu(2 * m * n);
        TargetWrapperXPU::MemcpySync(
                        tmp_v_cache_cpu.data(),
                        value_cache, 
                        2*m*n*sizeof(float),
                        IoDirection::DtoH);
        for(int i=0; i<2*m*n; i++) {
          cout << i << '\t' << tmp_v_cache_cpu[i] << endl;
        }
      }
      */

      xdnn_ret = xdnn::broadcast<T>(ctx_, 
                            sliced_q_,
                            broadcast_q_,
                            {m, 1, n},
                            {m, step, n});
      CHECK_EQ(xdnn_ret, 0) << "broad ccast error.";

      xdnn::QKVAttnParam qkv_attn_param(m, 
                                      step, 
                                      head_num_,
                                      size_per_head_,
                                      {m,1,1,step},
                                      xdnn::Activation_t::RELU,
                                      0,
                                      false);

      xdnn_ret = xdnn::constant<T>(ctx_,
                                    self_mask_,
                                    m * step,
                                    static_cast<T>(0));
              
      xdnn_ret = xdnn::qk_attention<T, T, T, int32_t>(
          ctx_,
          broadcast_q_,
          key_cache,
          qk_buffer_,
          nullptr,
          nullptr,
          nullptr,
          qkv_attn_param,
          self_mask_);
      /*
      if(step == 2 || step == 1) {
        cout << "SELF_MASK STEP = " << step << endl;
        vector<float> self_mask_cpu(m * step);
        TargetWrapperXPU::MemcpySync(
                        self_mask_cpu.data(),
                        self_mask_, 
                        m*step*sizeof(float),
                        IoDirection::DtoH);
        for(int i=0; i<m*step; i++) {
          cout << i << '\t' << self_mask_cpu[i] << endl;
        }
      }
      */
      xdnn_ret = xdnn::qk_v_attention<T, T, T, int32_t>(
          ctx_,
          qk_buffer_,
          value_cache,
          context_buf_,
          nullptr,
          nullptr,
          nullptr,
          qkv_attn_param);
    } else {
      CHECK(false) << "NOT IMPLEMENTEDD.";
    }

    /*
    if(step == 2) {
      // vector<float> tmp_v_cache(m*step*n);
      vector<float> tmp_query(m*n);
      vector<float> tmp_context(m*n);
      TargetWrapperXPU::MemcpySync(
                        tmp_context.data(),
                        context_buf_, 
                        m*n*sizeof(float), 
                        IoDirection::DtoH); 
      TargetWrapperXPU::MemcpySync(
                        tmp_query.data(),
                        query_buf_, 
                        m*n*sizeof(float), 
                        IoDirection::DtoH); 
      cout << "QUERY/CONTEXT STEP = " << step << endl;
      for(int32_t xx=0; xx<tmp_context.size(); xx+=200) {
        cout << xx << '\t' << tmp_query[xx] << '\t' << tmp_context[xx] << endl;
      }
      cout << endl;
    }
    */
    
    xdnn_ret = xdnn::fc_fusion<T, T, T, float>(
      ctx_, /* context */
      context_buf_,          /* x */
      param_->self_attention.attention_output_weight.kernel,
      decoder_output,                      /* y */
      m,                  /* m */
      n,                        /* n */
      n,                        /* k */
      false,                       /* x_trans */
      false,                        /* w_trans */
      nullptr,              /* x_max */
      nullptr,             /* w_max */
      nullptr,             /* y_max */
      n,                        /* ldx */
      n,                        /* ldw */
      n,                        /* ldy */
      1.0f,                        /* alpha */
      0.0f,                        /* beta */
      param_->self_attention.attention_output_weight.bias, /* bias */
      xdnn::Activation_t::LINEAR); /* act_type */
      CHECK_EQ(xdnn_ret, 0) << "calling fc_fusion error2!.";
  } // end masked_multi_head_attention

  void cross_multi_head_attention(const T *from_tensor,
                                  const T *memory_tensor,
                                  T *key_mem_cache,
                                  T *value_mem_cache,
                                  T *decoder_output,
                                  const int32_t *memory_sequence_length,
                                  const bool *finished,
                                  const int32_t max_seq_len,
                                  const int32_t step) {
    int32_t m = max_batch_size_;
    int32_t n = hidden_units_;

    int32_t xdnn_ret;
    xdnn_ret = xdnn::fc_fusion<T, T, T, float>(
      ctx_, /* context */
      from_tensor,          /* x */
      param_->cross_attention.query_weight.kernel,
      query_buf_,                      /* y */
      m,                  /* m */
      n,                        /* n */
      n,                        /* k */
      false,                       /* x_trans */
      false,                        /* w_trans */
      nullptr,              /* x_max */
      nullptr,             /* w_max */
      nullptr,             /* y_max */
      n,                        /* ldx */
      n,                        /* ldw */
      n,                        /* ldy */
      1.0f,                        /* alpha */
      0.0f,                        /* beta */
      param_->cross_attention.query_weight.bias, /* bias */
      xdnn::Activation_t::LINEAR); /* act_type */
      CHECK_EQ(xdnn_ret, 0) << "calling fc_fusion error!.";
    
    if(step == 1) {
      xdnn_ret = xdnn::fc_fusion<T, T, T, float>(
        ctx_, /* context */
        memory_tensor,          /* x */
        param_->cross_attention.key_weight.kernel,
        key_mem_cache,                      /* y */
        m * max_seq_len,                  /* m */
        n,                        /* n */
        n,                        /* k */
        false,                       /* x_trans */
        false,                        /* w_trans */
        nullptr,              /* x_max */
        nullptr,             /* w_max */
        nullptr,             /* y_max */
        n,                        /* ldx */
        n,                        /* ldw */
        n,                        /* ldy */
        1.0f,                        /* alpha */
        0.0f,                        /* beta */
        param_->cross_attention.key_weight.bias, /* bias */
        xdnn::Activation_t::LINEAR); /* act_type */
      CHECK_EQ(xdnn_ret, 0) << "calling fc_fusion error!.";

      xdnn_ret = xdnn::fc_fusion<T, T, T, float>(
        ctx_, /* context */
        memory_tensor,          /* x */
        param_->cross_attention.value_weight.kernel,
        value_mem_cache,                      /* y */
        m * max_seq_len,                  /* m */
        n,                        /* n */
        n,                        /* k */
        false,                       /* x_trans */
        false,                        /* w_trans */
        nullptr,              /* x_max */
        nullptr,             /* w_max */
        nullptr,             /* y_max */
        n,                        /* ldx */
        n,                        /* ldw */
        n,                        /* ldy */
        1.0f,                        /* alpha */
        0.0f,                        /* beta */
        param_->cross_attention.value_weight.bias, /* bias */
        xdnn::Activation_t::LINEAR); /* act_type */
      CHECK_EQ(xdnn_ret, 0) << "calling fc_fusion error!.";
    
      std::vector<int32_t> memory_sequence_length_cpu(m);
      std::vector<float> tmp_float_finish_cpu(m * max_seq_len, PLITE_FLT_MIN);
      TargetWrapperXPU::MemcpySync(
                        memory_sequence_length_cpu.data(),
                        memory_sequence_length, 
                        m*sizeof(int32_t), 
                        IoDirection::DtoH); 
      for(int32_t i=0; i<m; i++) {
        std::fill_n(tmp_float_finish_cpu.begin() + i*max_seq_len, 
                      memory_sequence_length_cpu[i], 
                      0.0f);
      }

      TargetWrapperXPU::MemcpySync(
                          cross_mask_, 
                          tmp_float_finish_cpu.data(), 
                          m * max_seq_len * sizeof(float), 
                          IoDirection::HtoD);
    }

    xdnn::QKVAttnParam qkv_attn_param(m, 
                                      max_seq_len, 
                                      head_num_,
                                      size_per_head_,
                                      {m, 1, 1, max_seq_len},
                                      xdnn::Activation_t::RELU,
                                      0, // important!
                                      false);
    
    xdnn_ret = xdnn::broadcast<T>(ctx_, 
                            query_buf_,
                            broadcast_q_,
                            {m, 1, n},
                            {m, max_seq_len, n});
    CHECK_EQ(xdnn_ret, 0) << "broadcast error!.";

    xdnn_ret = xdnn::qk_attention<T, T, T, int32_t>(
          ctx_,
          broadcast_q_,
          key_mem_cache,
          qk_buffer_,
          nullptr,
          nullptr,
          nullptr,
          qkv_attn_param,
          cross_mask_);
    CHECK_EQ(xdnn_ret, 0) << "cross attention qk_attention error.";


    xdnn_ret = xdnn::qk_v_attention<T, T, T, int32_t>(
          ctx_,
          qk_buffer_,
          value_mem_cache,
          context_buf_,
          nullptr,
          nullptr,
          nullptr,
          qkv_attn_param);
    CHECK_EQ(xdnn_ret, 0) << "cross attention qk_v_attention error.";
    
    if(step == 2) {
      /*
      vector<float> tmp_v_out(m*n);
      TargetWrapperXPU::MemcpySync(
                        tmp_v_out.data(),
                        query_buf_, 
                        m*n*sizeof(float), 
                        IoDirection::DtoH);
      cout << "STEP QUERY = " << step << endl;
      for(int i=0; i<m*n; i+=200) {
        cout << i << '\t' << tmp_v_out[i] << endl;
      }
      */
      /*
      vector<float> tmp_v_out(m*n);
      TargetWrapperXPU::MemcpySync(
                        tmp_v_out.data(),
                        context_buf_, 
                        m*n*sizeof(float), 
                        IoDirection::DtoH);
      cout << "STEP CROSS = " << step << endl;
      for(int i=0; i<m*n; i+=200) {
        cout << i << '\t' << tmp_v_out[i] << endl;
      }
      */
    }

    xdnn_ret = xdnn::fc_fusion<T, T, T, float>(
        ctx_, /* context */
        context_buf_,          /* x */
        param_->cross_attention.attention_output_weight.kernel,
        decoder_output,                      /* y */
        m,                  /* m */
        n,                        /* n */
        n,                        /* k */
        false,                       /* x_trans */
        false,                        /* w_trans */
        nullptr,              /* x_max */
        nullptr,             /* w_max */
        nullptr,             /* y_max */
        n,                        /* ldx */
        n,                        /* ldw */
        n,                        /* ldy */
        1.0f,                        /* alpha */
        0.0f,                        /* beta */
        param_->cross_attention.attention_output_weight.bias, /* bias */
        xdnn::Activation_t::LINEAR); /* act_type */
    CHECK_EQ(xdnn_ret, 0) << "calling fc_fusion error!.";
  } // end cross_multi_head_attention()
};

template<typename T>
class DecodingBeamsearch {
private:
  DecodingBeamsearchArguments args_;
  baidu::xpu::api::Context* ctx_{};
  bool is_fuse_topk_softMax_{};
  bool keep_alive_beam_{};

  std::unique_ptr<OpenDecoder<T>> decoder_{};
  //T **K_cache_{};
  //T **V_cache_{};
  array<T*, 2> K_cache_;
  array<T*, 2> V_cache_;
  //T **K_mem_cache_{};
  //T **V_mem_cache_{};
  vector<T*> K_mem_cache_;
  vector<T*> V_mem_cache_;

  T *from_tensor_[2]{};
  T *decoder_buf_{};

  // Prefix LM
  /// T *trans_out_buf_;
  /// T *lm_normed_result_buf_;

  T *decoder_normed_result_buf_{};
  // T *embedding_buf_{};
  float *logits_buf_{};
  float *cum_log_buf_{};
  int32_t *word_ids_buf_{};
  int32_t *parent_ids_buf_{};
  bool *finished_buf_{};
  bool *alive_finished_buf_{};

  lite::Tensor buf_tensor_;
  void *buf_{};
  int32_t *finished_count_buf_{};
  bool *h_finished_buf_{};
  int32_t *h_trg_length_{};
  float *temp_storage_{};

  void *topK_kernel_workspace = nullptr;
  size_t topk_workspace_size_ = 0;

  T *padded_embedding_kernel{};
  T *padded_embedding_bias{};
  T *tmp_logits_buf_{};

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
                     const float beam_search_diversity_rate = -0.0f, // 0
                     const bool is_fuse_topk_softMax = false, // true
                     const bool is_fuse_qkv = false, // true
                     const bool keep_alive_beam = false, // true
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
    
    VLOG(2) << "DecodingBeamsearch INFO: \n" 
            << "- batch_size\t" << batch_size << '\n'
            << "- beam_width\t" << beam_width << '\n'
            << "- seq_len\t" << seq_len << '\n'
            << "- head_num\t" << head_num << '\n'
            << "- size_per_head\t" << size_per_head << '\n'
            << "- vocab_size\t" << vocab_size << '\n'
            << "- decoder_layers\t" << decoder_layers << '\n'
            << "- memory_hidden_units\t" << memory_hidden_units << '\n'
            << "- momory_max_seq_len\t" << memory_max_seq_len << '\n'
            << "- start_id:\t" << args_.start_id_ << '\n';

    if (std::is_same<T, float>::value)
      args_.vocab_size_padded_ = vocab_size;
    else 
      CHECK(false) << "half not supported now";

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

    //K_cache_ = new T *[2];
    //V_cache_ = new T *[2];

    //K_mem_cache_ = new T *[args_.decoder_layers_];
    //V_mem_cache_ = new T *[args_.decoder_layers_];
    K_mem_cache_.resize(args_.decoder_layers_);
    V_mem_cache_.resize(args_.decoder_layers_);

    // TODO: OpenDecoder
    decoder_ = std::unique_ptr<OpenDecoder<T>>(new OpenDecoder<T>(ctx,
                                                        head_num,
                                                        size_per_head,
                                                        memory_hidden_units,
                                                        is_fuse_qkv,
                                                        normalization_before));
    decoder_->set_max_batch_size(batch_size * beam_width);
    decoder_->set_max_mem_length(memory_max_seq_len);
    decoder_->set_max_seq_length(seq_len);

    size_t from_tensor_size =
        args_.batch_size_ * args_.beam_width_ * args_.hidden_units_; 
    size_t decoder_workspace_size = decoder_->getWorkspaceSize();
    size_t decoder_normed_result_buffer_size =
        args_.batch_size_ * args_.beam_width_ * args_.hidden_units_;
    size_t cache_size = (prefix_lm)
          ? (args_.batch_size_ * args_.beam_width_ 
            * (args_.seq_len_ + args_.memory_max_seq_len_) 
            * args_.hidden_units_)
          : (args_.batch_size_ * args_.beam_width_ 
            * args_.seq_len_ * args_.hidden_units_);

    size_t mem_cache_size =
        (prefix_lm) ? 0 : (args_.batch_size_ * args_.beam_width_ *
                           memory_max_seq_len * args_.hidden_units_);  // type T
        // 8 * 4 * 256 * 1024 = 8388608
    size_t logits_buf_size      = args_.batch_size_ * args_.beam_width_ *
                                  args_.vocab_size_padded_;
    size_t cum_log_buf_size     = args_.batch_size_ * args_.beam_width_;
    size_t word_ids_buf_size    = args_.batch_size_ * args_.beam_width_;
    size_t parent_ids_buf_size  = keep_alive_beam_ ? word_ids_buf_size : 0;
    size_t finished_buf_size    = args_.batch_size_ * args_.beam_width_;  // type bool

    size_t alive_finished_buf_size = keep_alive_beam_ ? finished_buf_size : 0;
    size_t finished_count_size = (size_t)(ceil(1 / 32.)) * 32;  // type int
    
    size_t storage_size_per_beam = 2 * args_.beam_width_ + SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K + 2);
    args_.temp_storage_size_ = args_.batch_size_ * args_.beam_width_ * storage_size_per_beam;  // type float
    args_.temp_storage_size_ = (size_t)(
        ceil(args_.batch_size_ * args_.beam_width_ * args_.beam_width_ / 4.) *
            4 * 2 +
        ceil(args_.batch_size_ * args_.beam_width_ *
             SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K + 2) / 4.) *
            4);

    size_t padded_embedding_kernel_size =
        args_.hidden_units_ * args_.vocab_size_padded_;
    size_t padded_embedding_bias_size = args_.vocab_size_padded_;
    if (args_.vocab_size_padded_ == args_.vocab_size_) {
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
          ceil(args_.batch_size_ * args_.beam_width_ * args_.beam_width_ / 4.) * 4 * 2;
    } else {
      CHECK(false) << "keep_alive_beam_ == false not implemented.";
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
    
    // TODO: omit topK_kernel_laucher
    { 
      // implement topK_update_kernelLauncher()
      const int32_t max_block_per_beam = 8;
      int32_t temp_log_probs_buf_size = batch_size * beam_width * vocab_size;  // type float
      // select top beam_width*2 for topk_tmp_id_buf and topk_tmp_val_buf
      int32_t topk_tmp_ids_buf_size = batch_size * beam_width * beam_width * 2 * max_block_per_beam;  // type int
      int32_t topk_tmp_val_buf_size = batch_size * beam_width * beam_width * 2 * max_block_per_beam;  // type float
      // // to save tmp output_cum_log_probs results of the alive beams
      // topk_tmp_val_buf_size += batch_size * beam_width;

      // prevent memory misalinged address
      temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
      topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
      topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

      topk_workspace_size_ = sizeof(float) * temp_log_probs_buf_size +
                              sizeof(int) * topk_tmp_ids_buf_size +
                              sizeof(float) * topk_tmp_val_buf_size;
    }

    size_t lm_head_buffer_size = (prefix_lm)
                          ? decoder_normed_result_buffer_size
                          : decoder_normed_result_buffer_size * 3;


    size_t datatype_buf_size =
        from_tensor_size * 2 + 
        (cache_size * 4 + mem_cache_size * 2) * args_.decoder_layers_ +
        decoder_workspace_size + 
        lm_head_buffer_size;

    // 为了兼容qk_attention只能接受float类型的mask，所以datatype_buf_size分配sizeof(float)而不是size(T)
    buf_ = buf_tensor_.mutable_data(TARGET(kXPU), 
                sizeof(float) * datatype_buf_size + 
                sizeof(float) * (logits_buf_size + cum_log_buf_size) + 
                sizeof(T) * tmp_logits_buf_size + 
                sizeof(T) * padded_embedding_kernel_size + 
                sizeof(float) * padded_embedding_bias_size + 
                sizeof(int32_t) * (word_ids_buf_size + parent_ids_buf_size) + 
                sizeof(bool) * (finished_buf_size + alive_finished_buf_size) + 
                topk_workspace_size_ + 
                sizeof(float) * args_.temp_storage_size_ + 
                sizeof(int32_t) * finished_count_size); // TODO: not accurate!


    from_tensor_[0] = (T*)(buf_);
    from_tensor_[1] = (T*)(from_tensor_[0] + from_tensor_size);

    for (int32_t i = 0; i < args_.decoder_layers_; ++i) {
      K_mem_cache_[i] = from_tensor_[1] + from_tensor_size + i * mem_cache_size * 2;
      V_mem_cache_[i] = from_tensor_[1] + from_tensor_size + i * mem_cache_size * 2 + mem_cache_size;
    }
    if (args_.beam_width_ > 1) {
      /* We use two-way buffer since we have to update KV buf at the end of each
       * step. */
      K_cache_[0] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                    0 * cache_size * args_.decoder_layers_;
      K_cache_[1] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                    1 * cache_size * args_.decoder_layers_;
      V_cache_[0] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                    2 * cache_size * args_.decoder_layers_;
      V_cache_[1] = V_mem_cache_[decoder_layers - 1] + mem_cache_size +
                    3 * cache_size * args_.decoder_layers_;
    } else {
      // if beam width is 1, we only need one buffer
      CHECK(false) << "not implemented.";
    }

    decoder_buf_ = V_cache_[1] + cache_size * args_.decoder_layers_;
    
    if (prefix_lm) {
        CHECK(false) << "not implemented!";
    } else {
      decoder_normed_result_buf_ = (decoder_buf_ + decoder_workspace_size);
      // Used for post-norm.
      // embedding_buf_ = (decoder_buf_ + decoder_workspace_size);
    }

    logits_buf_ = (float *)(decoder_normed_result_buf_ +
                            decoder_normed_result_buffer_size);
    cum_log_buf_ = (float *)(logits_buf_ + logits_buf_size);
    word_ids_buf_ = (int32_t *)(cum_log_buf_ + cum_log_buf_size);
    parent_ids_buf_ = (int32_t *)(word_ids_buf_ + word_ids_buf_size);
    finished_buf_ = (bool *)(parent_ids_buf_ + parent_ids_buf_size);
    alive_finished_buf_ = (bool *)(finished_buf_ + finished_buf_size);
    temp_storage_ = (float *)(alive_finished_buf_ + alive_finished_buf_size);
    finished_count_buf_ = (int32_t *)(temp_storage_ + args_.temp_storage_size_);
    topK_kernel_workspace = (void *)(finished_count_buf_ + finished_count_size);
    padded_embedding_kernel =
        (T *)((char *)topK_kernel_workspace + topk_workspace_size_);
    padded_embedding_bias =
        (T *)(padded_embedding_kernel + padded_embedding_kernel_size);
    tmp_logits_buf_ =
        (T *)(padded_embedding_bias + padded_embedding_bias_size);

    // h_finished_buf_ = new bool[finished_buf_size]; // TODO
    // h_trg_length_ = new int[args_.batch_size_]; // TODO
  }
  
  void forward(const vector<DecoderInitParam<T>>& param, DecodingInitParam<T>& decoding_params) {
    const int32_t m = args_.batch_size_ * args_.beam_width_;
    const int32_t k = args_.hidden_units_;
    const int32_t n = args_.vocab_size_padded_;
    const T *embedding_kernel_ptr = nullptr;
    const T *embedding_bias_ptr = nullptr;
    embedding_kernel_ptr = (const T*)decoding_params.embedding_kernel;
    embedding_bias_ptr = (const T*)decoding_params.embedding_bias;
    int32_t min_trg_len = 0;
    // int32_t max_trg_len = 0;

    // call init_kernelLauncher_v2()
    int32_t xdnn_ret;
    xdnn_ret = xdnn::constant<bool>(ctx_, 
                            finished_buf_, 
                            args_.batch_size_ * args_.beam_width_ * 2, 
                            false);
    CHECK_EQ(xdnn_ret, 0) << "calling xdnn::constant error 1!.";
    xdnn_ret = xdnn::constant<bool>(ctx_, 
                            alive_finished_buf_, 
                            args_.batch_size_ * args_.beam_width_, 
                            false);
    CHECK_EQ(xdnn_ret, 0) << "calling xdnn::constant error 2!.";

    xdnn_ret = xdnn::constant<int32_t>(ctx_, 
                            word_ids_buf_, 
                            args_.batch_size_ * args_.beam_width_, 
                            args_.start_id_);
    CHECK_EQ(xdnn_ret, 0) << "calling xdnn::constant error 3!.";
    
    vector<float> h_cum_log_buf_0(args_.batch_size_ * args_.beam_width_, -1e20f);
    vector<float> h_cum_log_buf_1(args_.batch_size_ * args_.beam_width_, -1e20f);
    for(int32_t tmp_i=0; tmp_i<args_.batch_size_; tmp_i++) {
      h_cum_log_buf_1.at(tmp_i * args_.beam_width_) = 0.0f;
    }

    // end init_kernelLauncher_v2()
    
    int32_t cache_size = (args_.prefix_lm_) ? 
            (m * (args_.seq_len_ + args_.memory_max_seq_len_) * args_.hidden_units_)
            : (m * args_.seq_len_ * args_.hidden_units_);  // type T
            // 4 * 8 * 256 * 1024 = 8388608


    for (uint32_t step = 1; step <= args_.seq_len_; ++step) {
      // we use two-way buffer
      int32_t kv_cache_id = step & 0x1;
      /*
      if(step == 2) {
        vector<int32_t> h_word_buf(args_.batch_size_ * args_.beam_width_);
        TargetWrapperXPU::MemcpySync(
            h_word_buf.data(), 
            word_ids_buf_, 
            args_.batch_size_ * args_.beam_width_ * sizeof(int32_t), 
            IoDirection::DtoH); 
        cout << "STEP2 WORD IND " << endl;
        for(auto w: h_word_buf) {
          cout << w << '\t';
        }
        cout << endl;
      }
      */
      // call embedding_lookup_sine_position_encoding_kernel_launcher()
      xdnn_ret = xdnn::embedding<T, int32_t>(
                                ctx_, /* context */
                                decoding_params.embedding_table,
                                word_ids_buf_,
                                from_tensor_[0],
                                args_.vocab_size_ ,
                                args_.hidden_units_,
                                m,
                                0);
      CHECK_EQ(xdnn_ret, 0) << "calling xdnn::constant error 5!.";                            
      const float scale = std::sqrt(args_.hidden_units_);
      xdnn_ret = xdnn::scale<T>(
                        ctx_,
                        from_tensor_[0],
                        from_tensor_[0],
                        m * args_.hidden_units_,
                        0.0f, /* bias_after_scale */
                        scale,            /* alpha */
                        0.0f);            /* beta */
      CHECK_EQ(xdnn_ret, 0) << "calling xdnn::constant error 6!.";
      xdnn_ret = xdnn::broadcast_add<T>(ctx_,
                    from_tensor_[0],
                    decoding_params.position_encoding_table 
                    + (step - 1 + args_.pos_offset_) * args_.hidden_units_,
                    from_tensor_[0],
                    {m, static_cast<int32_t>(args_.hidden_units_)},
                    {1, static_cast<int32_t>(args_.hidden_units_)});
      CHECK_EQ(xdnn_ret, 0) << "calling xdnn::constant error 7!.";
      // end embedding_lookup_sine_position_encoding_kernel_launcher()

      int32_t from_id, out_id;
      for (int32_t layer = 0; layer < args_.decoder_layers_; ++layer) {
        /*
         * For the first layer (layer-0), from_id  is 0. We also stored the embedding lookup
         * result in from_tensor_[0]
         */
        from_id = layer & 0x1;
        out_id = 1 - from_id;

        /*
         * We use one decoder_ object to process multiple decoder layers.
         *
         * At the beginning of each decoder layer, we initialize the decoder object
         * with corresponding weights and decoder_buf_.
         * The decoder_buf_ is reused.
         */
        decoder_->initialize(param[layer], decoder_buf_);
    
        if (args_.prefix_lm_) {
          CHECK(false) << "Not IMPLEMENTED!";
        } else {
          decoder_->forward(
              from_tensor_[from_id],
              decoding_params.memory_tensor,
              K_cache_[kv_cache_id] + layer * cache_size,
              V_cache_[kv_cache_id] + layer * cache_size,
              K_mem_cache_[layer],
              V_mem_cache_[layer],
              decoding_params.memory_sequence_length,
              from_tensor_[out_id],
              step,
              args_.seq_len_,
              true, // is_cross_attention
              keep_alive_beam_ ? alive_finished_buf_ : finished_buf_);
        }
      } // end layer loop
      
      if(step == 2) {
          vector<float> attention_out(args_.batch_size_ * args_.beam_width_ * args_.hidden_units_);
          TargetWrapperXPU::MemcpySync(
              attention_out.data(), 
              from_tensor_[0], 
              args_.batch_size_ * args_.beam_width_ * args_.hidden_units_ * sizeof(float), 
              IoDirection::DtoH); 
          cout << "STEP = " << step << endl;
          for(size_t i=0; i<attention_out.size(); i+=200) {
            cout << i << '\t' << attention_out[i] << endl;
          }
      }
      
      if(step > min_trg_len) {
        xdnn_ret = xdnn::layer_norm<float>(ctx_,    
                           from_tensor_[out_id], 
                           decoder_normed_result_buf_,
                           m,
                           k,
                           1e-6f,
                           decoding_params.layernorm.gamma,
                           decoding_params.layernorm.beta,
                           nullptr,
                           nullptr);
        
        xdnn_ret = xdnn::fc_fusion<float, float, float, float>(
            ctx_, /* context */
            decoder_normed_result_buf_,          /* x */
            embedding_kernel_ptr,
            tmp_logits_buf_,                      /* y */
            m,                  /* m */
            n,                        /* n */
            k,                        /* k */
            false,                       /* x_trans */
            false,                        /* w_trans */
            nullptr,              /* x_max */
            nullptr,             /* w_max */
            nullptr,             /* y_max */
            k,                        /* ldx */
            n,                        /* ldw */
            n,                        /* ldy */
            1.0f,                        /* alpha */
            0.0f,                        /* beta */
            embedding_bias_ptr, /* bias */
            xdnn::Activation_t::LINEAR); /* act_type */
        CHECK_EQ(xdnn_ret, 0) << "calling embedding_kernel fc_fusion error!.";  
      
        if(is_fuse_topk_softMax_ == true) {
          if(keep_alive_beam_ == true) {
            // Use separated alive and finish beam queues to avoid the decrease
            // of alive beams.
            topK_softMax_update(tmp_logits_buf_,
                                finished_buf_,
                                alive_finished_buf_,
                                decoding_params.sequence_length,
                                word_ids_buf_,
                                parent_ids_buf_,
                                decoding_params.output_ids + (step - 1) * m * 2,
                                decoding_params.parent_ids + (step - 1) * m * 2,
                                &h_cum_log_buf_0,
                                &h_cum_log_buf_1,
                                reinterpret_cast<void *>(temp_storage_),
                                step,
                                args_);
          } else {
            CHECK(false) << "keep_alive_beam == false not implemented.";
          }
        } else {
          CHECK(false) << "is_fuse_topk_softMax == false not implemented.";
        }
      }

      if(args_.beam_width_ > 1) {
        int32_t decoder_max_seq_len = args_.seq_len_;
        update_KV_cache_kernelLauncher_v2(
            &K_cache_[0],
            &V_cache_[0],
            keep_alive_beam_ ? parent_ids_buf_ : decoding_params.parent_ids + (step - 1) * m,
            keep_alive_beam_ ? alive_finished_buf_ : finished_buf_,
            args_.batch_size_,
            args_.beam_width_,
            args_.head_num_,
            args_.size_per_head_,
            step,
            decoder_max_seq_len,
            cache_size,
            args_.decoder_layers_,
            (args_.prefix_lm_) ? args_.memory_max_seq_len_ : -1);
      }
    } // end step loop
  }

  void topK_softMax_update(
            const T* log_probs,
            bool* finished,
            bool* alive_finished,
            int32_t* h_sequence_length,
            int32_t* word_ids_buf,
            int32_t* parent_ids_buf,
            int32_t* h_output_word_ids,
            int32_t* h_output_parent_ids,  // for gather tree, include both alive and finish beams
            vector<float>* h_output_cum_log_probs_ptr0,  // NOTE: cum_log_probs is T in V3.1
            vector<float>* h_output_cum_log_probs_ptr1,
            void* temp_storage,
            const int32_t step,
            DecodingBeamsearchArguments args) {
    
    const int32_t temp_storage_size = args.temp_storage_size_;
    const int32_t batch_size = args.batch_size_;
    const int32_t beam_width = args.beam_width_;
    const int32_t vocab_size = args.vocab_size_padded_;
    const int32_t end_id = args.end_id_;
    const T diversity_rate = args.beam_search_diversity_rate_;
    const int32_t max_out_len = args.seq_len_;
    const float alpha = args.alpha_;
    const int32_t finished_candidate_num = args.finished_candidate_num_;
    const bool early_stopping = args.early_stopping_;
    if(step == 1) {
      VLOG(2) << "[MYDEBUGXPU] temp_storage_size = " << temp_storage_size;
      VLOG(2) << "[MYDEBUGXPU] batch_size = " << batch_size;
      VLOG(2) << "[MYDEBUGXPU] beam_width = " << beam_width;
      VLOG(2) << "[MYDEBUGXPU] vocab_size = " << vocab_size;
      VLOG(2) << "[MYDEBUGXPU] end_id = " << end_id;
      VLOG(2) << "[MYDEBUGXPU] diversity_rate = " << diversity_rate;
      VLOG(2) << "[MYDEBUGXPU] max_out_len = " << max_out_len;
      VLOG(2) << "[MYDEBUGXPU] alpha = " << alpha;
      VLOG(2) << "[MYDEBUGXPU] finished_candidate_num = " << finished_candidate_num;
      VLOG(2) << "[MYDEBUGXPU] early_stopping = " << early_stopping;
    }

    lite::Tensor sorted_beam_value;
    sorted_beam_value.Resize({batch_size*beam_width, beam_width*2});
    lite::Tensor sorted_ind_beam_topk;
    sorted_ind_beam_topk.Resize({batch_size*beam_width, beam_width*2});
    int32_t xdnn_ret;
    xdnn_ret = xdnn::sorted_softmax_topk<float, int32_t>(
                                        ctx_,
                                        log_probs,
                                        sorted_beam_value.mutable_data<float>(TARGET(kXPU), sorted_beam_value.dims().production()*sizeof(float)),
                                        sorted_ind_beam_topk.mutable_data<int32_t>(TARGET(kXPU), sorted_ind_beam_topk.dims().production()*sizeof(int32_t)),
                                        {batch_size*beam_width, vocab_size},
                                        1,
                                        beam_width*2);
    CHECK_EQ(xdnn_ret, 0) << "calling sorted_softmax_topk error.";

    lite::Tensor sub_cum_log_buf_1; // TODO: remove in the future
    sub_cum_log_buf_1.Resize({batch_size, beam_width});
    sub_cum_log_buf_1.mutable_data<float>(TARGET(kXPU), sub_cum_log_buf_1.dims().production() * sizeof(float));
    TargetWrapperXPU::MemcpySync(
            sub_cum_log_buf_1.mutable_data<float>(), 
            h_output_cum_log_probs_ptr1->data(), 
            batch_size*beam_width*sizeof(float), 
            IoDirection::HtoD);
    
    xdnn_ret = xdnn::broadcast_add<float>(ctx_,
                              sorted_beam_value.data<float>(),
                              sub_cum_log_buf_1.data<float>(),
                              sorted_beam_value.mutable_data<float>(),
                              {batch_size*beam_width, beam_width*2},
                              {batch_size*beam_width, 1});
    CHECK_EQ(xdnn_ret, 0) << "calling broadcast_add error.";

    // implement batch_topk_update_kernel() online_softmax_beamsearch_kernels.cu L1294
    lite::Tensor sorted_value_total_topk;
    sorted_value_total_topk.Resize({batch_size, beam_width*2});
    lite::Tensor sorted_ind_total_topk;
    sorted_ind_total_topk.Resize({batch_size, beam_width*2});
    xdnn_ret = xdnn::sorted_topk<float>(ctx_,
                                        sorted_beam_value.data<float>(),
                                        sorted_value_total_topk.mutable_data<float>(TARGET(kXPU), sorted_value_total_topk.dims().production()*sizeof(float)),
                                        sorted_ind_total_topk.mutable_data<int32_t>(TARGET(kXPU), sorted_ind_total_topk.dims().production()*sizeof(int32_t)),
                                        batch_size,
                                        beam_width * beam_width * 2,
                                        beam_width * 2);
    CHECK_EQ(xdnn_ret, 0) << "calling sorted_topk error.";

    vector<int32_t> h_sorted_ind_beam_topk(batch_size * beam_width * beam_width * 2);
    vector<int32_t> h_sorted_ind_total_topk(batch_size * beam_width * 2);
    vector<float> h_sorted_value_total_topk(batch_size * beam_width * 2);
    TargetWrapperXPU::MemcpySync(
            h_sorted_ind_beam_topk.data(), 
            sorted_ind_beam_topk.data<int32_t>(), 
            batch_size * beam_width * beam_width * 2 * sizeof(int32_t), 
            IoDirection::DtoH); 
    TargetWrapperXPU::MemcpySync(
            h_sorted_ind_total_topk.data(), 
            sorted_ind_total_topk.data<int32_t>(), 
            batch_size * beam_width * 2 * sizeof(int32_t), 
            IoDirection::DtoH); 
    TargetWrapperXPU::MemcpySync(
            h_sorted_value_total_topk.data(), 
            sorted_value_total_topk.data<float>(), 
            batch_size * beam_width * 2 * sizeof(float), 
            IoDirection::DtoH);
    
    vector<int32_t> h_tmp_word_ids_buf(batch_size * beam_width);
    vector<int32_t> h_tmp_parent_ids_buf(batch_size * beam_width);
    vector<int8_t> h_temp_finished(batch_size * beam_width * 2);
    vector<int8_t> h_temp_alive_finished(batch_size * beam_width);
    TargetWrapperXPU::MemcpySync(
            h_tmp_word_ids_buf.data(), 
            word_ids_buf, 
            batch_size * beam_width * sizeof(int32_t), 
            IoDirection::DtoH);
    TargetWrapperXPU::MemcpySync(
            h_tmp_parent_ids_buf.data(), 
            parent_ids_buf, 
            batch_size * beam_width * sizeof(int32_t), 
            IoDirection::DtoH);
    TargetWrapperXPU::MemcpySync(
            h_temp_finished.data(), 
            finished, 
            batch_size * beam_width * sizeof(bool) * 2, 
            IoDirection::DtoH);
    TargetWrapperXPU::MemcpySync(
            h_temp_alive_finished.data(), 
            alive_finished, 
            batch_size * beam_width * sizeof(bool), 
            IoDirection::DtoH);

    float length_penalty = std::pow((5. + step + 1) / 6., alpha);
    float max_length_penalty = std::pow((5. + max_out_len + 1) / 6., alpha);

    for(int32_t bs=0; bs<batch_size; bs++) {
      T* output_cum_log_probs_ptr0 = h_output_cum_log_probs_ptr0->data() + bs * beam_width;
      T* output_cum_log_probs_ptr1 = h_output_cum_log_probs_ptr1->data() + bs * beam_width;
      int32_t* output_word_ids_ptr = h_output_word_ids + bs * (beam_width * 2);
      int32_t* output_parent_ids_ptr = h_output_parent_ids + bs * (beam_width * 2);
      int8_t* finished_ptr = h_temp_finished.data() + bs * (beam_width * 2);
      int8_t* alive_finished_ptr = h_temp_alive_finished.data() + bs * beam_width;
      float* h_sorted_value_total_topk_ptr = h_sorted_value_total_topk.data() + bs * (beam_width * 2);
      int32_t* h_sorted_ind_total_topk_ptr = h_sorted_ind_total_topk.data() + bs * (beam_width * 2);
      int32_t* h_sorted_ind_beam_topk_ptr = h_sorted_ind_beam_topk.data() + bs * (beam_width * beam_width * 2);
      int32_t* h_tmp_word_ids_buf_ptr = h_tmp_word_ids_buf.data() + bs * beam_width;
      int32_t* h_tmp_parent_ids_buf_ptr = h_tmp_parent_ids_buf.data() + bs * beam_width;
      int32_t* sequence_length_ptr = h_sequence_length + bs * (beam_width * 2);

      int32_t finish_num = 0;
      vector<TopKFinish<T>> finish_candidate(beam_width);
      finish_candidate.reserve(beam_width * 2);

      if(step == 1) {
        for(int32_t i=0; i<beam_width; i++) {
          finish_candidate[i].u = PLITE_FLT_MIN;
          finish_candidate[i].idx = -1;
          finish_candidate[i].len = 0;
        }
      } else {
        for(int32_t i=0; i<beam_width; i++) {
          finish_candidate[i].u = output_cum_log_probs_ptr0[i];
          finish_candidate[i].idx = i;
          finish_candidate[i].len = output_parent_ids_ptr[i];
          if(finished_ptr[i] != 0) ++finish_num;
        }
      }

      int32_t alive_num = 0;
      for(int32_t i=0; i<beam_width*2; i++) {
        int32_t word_id = h_sorted_ind_beam_topk_ptr[h_sorted_ind_total_topk_ptr[i]];
        float cum_log_prob = h_sorted_value_total_topk_ptr[i];
        int32_t beam_id = h_sorted_ind_total_topk_ptr[i] / (beam_width * 2) + bs * beam_width;

        int32_t beam_id_in_output = bs * (beam_width * 2) + beam_id % beam_width + beam_width;
        if(word_id == end_id) {
          finish_candidate.push_back(TopKFinish<T>{cum_log_prob / length_penalty, beam_id_in_output, step});
          if (finish_num != beam_width) finish_num++;
        } else if (alive_num < beam_width) {
          h_tmp_parent_ids_buf_ptr[alive_num] = beam_id;
          h_tmp_word_ids_buf_ptr[alive_num] = word_id;
          // Also put alive candidates after finish candidates, since output
          // must include both the finish and alive to trace full path
          output_word_ids_ptr[beam_width + alive_num] = word_id;
          output_parent_ids_ptr[beam_width + alive_num] = beam_id_in_output;
          output_cum_log_probs_ptr1[alive_num] = cum_log_prob;
          sequence_length_ptr[beam_width + alive_num] = step;
          finished_ptr[beam_width + alive_num] = 0;
          alive_finished_ptr[alive_num] = 0;
          alive_num++;
        }
      }

      std::sort(finish_candidate.begin(), finish_candidate.end(), 
                [](TopKFinish<T> &a, TopKFinish<T>& b) {return a.u > b.u;});
      
      for(int32_t i=0; i<beam_width; i++) {
        output_word_ids_ptr[i] = end_id;
        output_cum_log_probs_ptr0[i] = finish_candidate[i].u;
        output_parent_ids_ptr[i] = finish_candidate[i].idx;
        sequence_length_ptr[i] = finish_candidate[i].len;
        finished_ptr[i] = finish_candidate[i].u > (PLITE_FLT_MIN + static_cast<T>(10.0f)) ? 1 : 0;
      }

      // early finish
      float lowest_finish = finish_num == 0 ? PLITE_FLT_MIN : output_cum_log_probs_ptr0[finish_num - 1];
      // The best possible score of the most likely alive sequence
      
      float lower_bound = (float)output_cum_log_probs_ptr1[0] / max_length_penalty;

      if (step == max_out_len || lowest_finish > lower_bound) {  // when finishing
        for (int32_t i = 0; finish_num < beam_width; ++finish_num, ++i) {
          output_word_ids_ptr[finish_num] = h_tmp_word_ids_buf_ptr[i];
          output_cum_log_probs_ptr0[finish_num] = output_cum_log_probs_ptr1[i] / length_penalty;
          output_parent_ids_ptr[finish_num] = output_parent_ids_ptr[i + beam_width];
          sequence_length_ptr[finish_num] = step;
          finished_ptr[finish_num] = 1;
        }
        // If early stop, also mark the alive beams finished.
        for (int32_t i = beam_width; i < beam_width*2; ++i) {
          finished_ptr[i] = 1;
          alive_finished_ptr[i - beam_width] = 1;
        }
      }
    } // end batch
    /*
    if(step == 1) {
      std::cout << "parent id " << std::endl;
      for(auto v : h_tmp_parent_ids_buf) {
        std::cout << v << '\t';
      }
      std::cout << std::endl;
    }
    */
    // sync cpu data to xpu
    TargetWrapperXPU::MemcpySync(
            word_ids_buf,
            h_tmp_word_ids_buf.data(),
            batch_size * beam_width * sizeof(int32_t), 
            IoDirection::HtoD);
    TargetWrapperXPU::MemcpySync(
            parent_ids_buf,
            h_tmp_parent_ids_buf.data(),
            batch_size * beam_width * sizeof(int32_t), 
            IoDirection::HtoD);
    TargetWrapperXPU::MemcpySync(
            finished,
            h_temp_finished.data(),
            batch_size * beam_width * sizeof(bool) * 2, 
            IoDirection::HtoD);
    TargetWrapperXPU::MemcpySync(
            alive_finished,
            h_temp_alive_finished.data(), 
            batch_size * beam_width * sizeof(bool), 
            IoDirection::HtoD);
  } // end topK_softMax_update


  void update_KV_cache_kernelLauncher_v2(T** key_cache,
                                       T** value_cache,
                                       const int* beam_ids,
                                       const bool* finished,
                                       const int32_t batch_size,
                                       const int32_t beam_width,
                                       const int32_t head_num,
                                       const int32_t size_per_head,
                                       const int32_t step,
                                       const int32_t decoder_max_seq_len,
                                       const int32_t cache_size,
                                       const int32_t decoder_layers,
                                       const int32_t memory_max_seq_len = -1) {
    if(memory_max_seq_len != -1) {
      CHECK(false) << "memory_max_seq_len = -1 not implemented.";
    }
    /*
    std::vector<int32_t>  h_beam(batch_size*beam_width);
    TargetWrapperXPU::MemcpySync(
            h_beam.data(), 
            beam_ids, 
            batch_size * beam_width * sizeof(int32_t), 
            IoDirection::DtoH); 
    */
    int32_t src_id = step & 0x1;
    int32_t tgt_id = 1 - src_id;
    int32_t xdnn_ret;
    for(int32_t layer=0; layer<decoder_layers; layer++) {
      xdnn_ret = xdnn::gather<float, int32_t>(ctx_,
              key_cache[src_id] + layer * cache_size,
              beam_ids,
              key_cache[tgt_id] + layer * cache_size,
              {batch_size*beam_width, step, head_num*size_per_head},
              batch_size*beam_width,
              0);
      CHECK_EQ(xdnn_ret, false) << "Gather k_cache error.";
      xdnn_ret = xdnn::gather<float, int32_t>(ctx_,
              value_cache[src_id] + layer * cache_size,
              beam_ids,
              value_cache[tgt_id] + layer * cache_size,
              {batch_size*beam_width, step, head_num*size_per_head},
              batch_size*beam_width,
              0);
      CHECK_EQ(xdnn_ret, false) << "Gather v_cache error.";
    }
  } // end update_KV_cache_kernelLauncher_v2

  virtual ~DecodingBeamsearch() {
    // delete[] K_cache_;
    // delete[] V_cache_;
    // delete[] K_mem_cache_;
    // delete[] V_mem_cache_;
    // delete[] h_finished_buf_;
    // delete[] h_trg_length_;
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
  // TODO: this is for debugging, remove in the formal release
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
  auto input_dims = input->dims();
  const int32_t batch_size = (decoding_strategy == "beam_search" ||
                     decoding_strategy == "beam_search_v2")
                        ? input_dims[0] / beam_width
                        : input_dims[0];
  const int32_t memory_max_seq_len = input_dims[1];
  const int32_t memory_hidden_dim = input_dims[2];
  const int32_t vocab_size = word_embedding->dims()[0];
  // VLOG(2) << "VOCAB SIZE IS " << vocab_size;
  // VLOG(2) << "SEQUENCE LEN IS " << sequence_length->dims();
  // VLOG(2) << "EMBEDDING DIM " << word_embedding->dims();
  DecodingInitParam<float> decoding_params;
  decoding_params.output_ids = output_ids->mutable_data<int32_t>(
                                  TARGET(kHost), output_ids->memory_size());
  decoding_params.parent_ids = parent_ids->mutable_data<int32_t>(
                                  TARGET(kHost), parent_ids->memory_size());
  decoding_params.sequence_length = sequence_length->mutable_data<int32_t>(
                                  TARGET(kHost), sequence_length->memory_size());
  decoding_params.memory_tensor = input->data<float>();
  decoding_params.memory_sequence_length = mem_seq_len->data<int32_t>();

  vector<DecoderInitParam<float>> params(num_layer);
  auto q_weight_dims = self_q_weight[0]->dims();
  auto k_weight_dims = self_k_weight[0]->dims();
  bool fuse_qkv = (q_weight_dims[1] == k_weight_dims[1]) ? false : true;
  VLOG(2) << "q_weight_dim: " << q_weight_dims;
  VLOG(2) << "k_weight_dim: " << k_weight_dims;

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
    params[i].ffn.intermediate_weight.kernel = ffn_inter_weight[i]->data<float>(); // [1024, 4096]
    params[i].ffn.intermediate_weight.bias = ffn_inter_bias[i]->data<float>(); 
    // out proj
    params[i].ffn.output_weight.kernel = ffn_out_weight[i]->data<float>(); // [4096, 1024]
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

  std::unique_ptr<DecodingBeamsearch<float>> decoding_beam_search;
  if(decoding_strategy == "beam_search_v2") {
    decoding_beam_search = std::unique_ptr<DecodingBeamsearch<float>>(
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
    CHECK(false) << "\"decoding_strategy\" only support \"beam_search_v2\", assigned value is "
                  << decoding_strategy;
  }
  
  decoding_beam_search->forward(params, decoding_params);


  return;
}

void FusionDecodingCompute::PrepareForRun() {
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
  param.output_ids_->mutable_data(TARGET(kHost), 
                      param.output_ids_->dims().production()*sizeof(int32_t));
  param.parent_ids_->Resize(parent_ids_dims);
  param.parent_ids_->mutable_data(TARGET(kHost), 
                      param.parent_ids_->dims().production()*sizeof(int32_t));
  param.sequence_length_->Resize(sequence_length_dims);
  param.sequence_length_->mutable_data(TARGET(kHost), 
                      param.sequence_length_->dims().production()*sizeof(int32_t));

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
    max_out_len,
    param.beam_search_diversity_rate_,
    param.alpha_);

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
    .BindOutput("OutputIds", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("ParentIds", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("SequenceLength", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
