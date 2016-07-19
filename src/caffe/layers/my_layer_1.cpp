// My own neuron activation function layer.
// Adapted from tanh layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/my_layer_1.hpp"

namespace caffe {

	template <typename Dtype>
	void My1Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int count = bottom[0]->count();
		for (int i = 0; i < count; ++i) {
			top_data[i] = 0.5*tanh(bottom_data[i])+0.5;
		}
	}

	template <typename Dtype>
	void My1Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {
			const Dtype* top_data = top[0]->cpu_data();
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const int count = bottom[0]->count();
			Dtype tanhx;
			for (int i = 0; i < count; ++i) {
				tanhx = top_data[i];
				bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx) * 0.5;
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(My1Layer);
#endif

	INSTANTIATE_CLASS(My1Layer);
	REGISTER_LAYER_CLASS(My1);

}  // namespace caffe
