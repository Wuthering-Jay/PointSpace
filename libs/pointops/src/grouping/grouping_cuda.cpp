#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "grouping_cuda_kernel.h"


void grouping_forward_cuda(int m, int nsample, int c, at::Tensor input_tensor, at::Tensor idx_tensor, at::Tensor output_tensor)
{
    TORCH_CHECK(input_tensor.is_cuda() && idx_tensor.is_cuda() && output_tensor.is_cuda(), "grouping tensors must be CUDA tensors");
    TORCH_CHECK(input_tensor.get_device() == idx_tensor.get_device() && input_tensor.get_device() == output_tensor.get_device(), "grouping tensors must be on the same device");
    TORCH_CHECK(input_tensor.is_contiguous() && idx_tensor.is_contiguous() && output_tensor.is_contiguous(), "grouping tensors must be contiguous");
    TORCH_CHECK(idx_tensor.scalar_type() == at::kInt, "grouping idx must be int32");
    TORCH_CHECK(input_tensor.scalar_type() == output_tensor.scalar_type(), "grouping input/output dtype mismatch");
    TORCH_CHECK(input_tensor.is_floating_point(), "grouping input must be floating point");
    grouping_forward_cuda_launcher(
        static_cast<int>(input_tensor.size(0)), m, nsample, c,
        input_tensor.data_ptr(), idx_tensor.data_ptr<int>(),
        output_tensor.data_ptr(), input_tensor.scalar_type());
}

void grouping_backward_cuda(int m, int nsample, int c, at::Tensor grad_output_tensor, at::Tensor idx_tensor, at::Tensor grad_input_tensor)
{
    TORCH_CHECK(grad_output_tensor.is_cuda() && idx_tensor.is_cuda() && grad_input_tensor.is_cuda(), "grouping tensors must be CUDA tensors");
    TORCH_CHECK(grad_output_tensor.get_device() == idx_tensor.get_device() && grad_output_tensor.get_device() == grad_input_tensor.get_device(), "grouping tensors must be on the same device");
    TORCH_CHECK(grad_output_tensor.is_contiguous() && idx_tensor.is_contiguous() && grad_input_tensor.is_contiguous(), "grouping tensors must be contiguous");
    TORCH_CHECK(idx_tensor.scalar_type() == at::kInt, "grouping idx must be int32");
    TORCH_CHECK(grad_output_tensor.scalar_type() == grad_input_tensor.scalar_type(), "grouping gradient dtype mismatch");
    grouping_backward_cuda_launcher(
        static_cast<int>(grad_input_tensor.size(0)), m, nsample, c,
        grad_output_tensor.data_ptr(), idx_tensor.data_ptr<int>(),
        grad_input_tensor.data_ptr(), grad_output_tensor.scalar_type());
}
