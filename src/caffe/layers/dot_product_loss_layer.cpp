#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DotProductLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
}

template <typename Dtype>
void DotProductLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  Dtype loss = caffe_cpu_dot(count, bottom[0]->cpu_data(), bottom[1]->cpu_data());
  (*top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DotProductLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      int j = (i == 0) ? 1 : 0;
      caffe_cpu_axpby(
          (*bottom)[i]->count(),
          top[0]->cpu_diff()[0],
          (*bottom)[j]->cpu_data(),
          Dtype(0),
          (*bottom)[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DotProductLossLayer);
#endif

INSTANTIATE_CLASS(DotProductLossLayer);

}  // namespace caffe
