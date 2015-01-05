#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxRankingLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void SoftmaxRankingLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data;
  Dtype* prob_data;
  Dtype* multiplier_data;
  int num_data = bottom[0]->num();
  int num_cand = bottom.size(); // number of candidates

  LossLayer<Dtype>::Reshape(bottom, top);

  sum_multiplier_.Reshape(1, num_cand, 1, 1);
  multiplier_data = sum_multiplier_.mutable_cpu_data();
  for (int i = 0; i < sum_multiplier_.count(); ++i) {
    multiplier_data[i] = 1.;
  }

  prob_.Reshape(num_data, num_cand, 1, 1);
  prob_data = prob_.mutable_cpu_data();
  for (int i = 0; i < num_data; ++i) {
    for (int j = 0; j < num_cand; ++j) {
      bottom_data = bottom[j]->cpu_data();
      prob_data[i * num_cand + j] = bottom_data[i];
    }
  }

  scale_.Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void SoftmaxRankingLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data;
  Dtype* prob_data = prob_.mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int num_data = bottom[0]->num();
  int num_cand = bottom.size(); // number of candidates
  int dim = bottom[0]->count() / num_data;

  Dtype loss = 0;
  for (int i = 0; i < num_data; ++i) {
    bottom_data = bottom[0]->cpu_data();
    scale_data[0] = bottom_data[i];
    for (int j = 0; j < num_cand; ++j) {
      bottom_data = bottom[j]->cpu_data();
      scale_data[0] = std::max(scale_data[0], bottom_data[i]);
    }
    // subtraction: prob_data[i] -= scale_data[0];
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_cand, 1,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., prob_data + i * dim);
    // exponentiation
    caffe_exp<Dtype>(dim, prob_data + i * dim, prob_data + i * dim);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, num_cand, 1, 1.,
        prob_data + i * dim, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < num_cand; ++j) {
      prob_data[i * num_cand + j] /= scale_data[0];
    }
    loss -= log(std::max( prob_data[i * num_cand]/scale_data[0], Dtype(FLT_MIN) ));
  }
  (*top)[0]->mutable_cpu_data()[0] = loss / num_data;
}

template <typename Dtype>
void SoftmaxRankingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  int num_data = prob_.num();
  int num_cand = prob_.channels(); // number of candidates

  if (propagate_down[0]) {
    Dtype* bottom_diff;
    const Dtype* prob_data = prob_.cpu_data();

    for (int i = 0; i < num_data; ++i) {
      for (int j = 0; j < num_cand; ++j) {
        bottom_diff = (*bottom)[j]->mutable_cpu_diff();
        bottom_diff[i] = prob_data[i * num_cand + j];
      }
    }
    
    bottom_diff = (*bottom)[0]->mutable_cpu_diff(); // clicked doc
    for (int i = 0; i < num_data; ++i) {
      bottom_diff[i] -= 1;
    }

    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    for (int j = 0; j < num_cand; ++j) {
      bottom_diff = (*bottom)[j]->mutable_cpu_diff();
      caffe_scal(num_cand, loss_weight / num_data, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxRankingLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxRankingLossLayer);


}  // namespace caffe
