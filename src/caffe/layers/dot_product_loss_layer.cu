#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DotProductLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  Dtype loss;
  caffe_gpu_dot(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), &loss);
  (*top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DotProductLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      int j = (i == 0) ? 1 : 0;
      caffe_gpu_axpby(
          (*bottom)[i]->count(),
          top[0]->cpu_diff()[0],
          (*bottom)[j]->gpu_data(),
          Dtype(0),
          (*bottom)[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_CLASS(DotProductLossLayer);

}  // namespace caffe
