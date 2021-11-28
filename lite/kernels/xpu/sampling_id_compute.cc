// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/sampling_id_compute.h"
#include "lite/backends/xpu/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

static void Xpu2Host(const Tensor& src, Tensor* dst) {
  dst->Resize(src.dims());
  dst->set_precision(src.precision());
  auto mem_size = src.memory_size();
  auto* data = dst->mutable_data(TARGET(kHost), mem_size);
  TargetWrapperXPU::MemcpySync(
      data, src.raw_data(), mem_size, IoDirection::DtoH);
}

static void Host2Xpu(const Tensor& src, Tensor* dst) {
  dst->Resize(src.dims());
  dst->set_precision(src.precision());
  auto mem_size = src.memory_size();
  auto* data = dst->mutable_data(TARGET(kXPU), mem_size);
  TargetWrapperXPU::MemcpySync(
      data, src.raw_data(), mem_size, IoDirection::HtoD);
}

static std::shared_ptr<std::mt19937_64> GetRandomEngine(uint64_t seed) {
  auto engine = std::make_shared<std::mt19937_64>();
  if (seed == 0) {
    std::random_device rd;
    seed = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
  }
  engine->seed(seed);
  return engine;
}

template <class T>
void SamplingIdCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  const lite::Tensor* x = param.x;
  lite::Tensor* out = param.out;

  Tensor x_t, out_t;
  Xpu2Host(*x, &x_t);
  out_t.Resize(out->dims());

  int64_t batch_size = x_t.dims()[0];
  int64_t width = x_t.dims()[1];
  auto x_data = x_t.data<T>();
  auto out_data = out_t.mutable_data<int64_t>();
  std::uniform_real_distribution<T> dist(static_cast<T>(param.min),
                                         static_cast<T>(param.max));
  if (engine == nullptr) {
    engine = GetRandomEngine(param.seed);
  }

  for (int64_t i = 0; i < batch_size; ++i) {
    T r = dist(*engine);
    int64_t idx = width - 1;
    for (int64_t j = 0; j < width; ++j) {
      if ((r -= x_data[i * width + j]) < 0) {
        idx = j;
        break;
      }
    }
    out_data[i] = idx;
  }

  Host2Xpu(out_t, out);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using sampling_id_float = paddle::lite::kernels::xpu::SamplingIdCompute<float>;
REGISTER_LITE_KERNEL(sampling_id, kXPU, kAny, kAny, sampling_id_float, float32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
