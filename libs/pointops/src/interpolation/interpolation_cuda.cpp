#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "interpolation_cuda_kernel.h"


void interpolation_forward_cuda(int n, int c, int k, at::Tensor input_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor output_tensor)
{
    TORCH_CHECK(input_tensor.is_cuda() && idx_tensor.is_cuda() && weight_tensor.is_cuda() && output_tensor.is_cuda(), "interpolation tensors must be CUDA tensors");
    TORCH_CHECK(input_tensor.get_device() == idx_tensor.get_device() && input_tensor.get_device() == weight_tensor.get_device() && input_tensor.get_device() == output_tensor.get_device(), "interpolation tensors must be on the same device");
    TORCH_CHECK(input_tensor.is_contiguous() && idx_tensor.is_contiguous() && weight_tensor.is_contiguous() && output_tensor.is_contiguous(), "interpolation tensors must be contiguous");
    TORCH_CHECK(idx_tensor.scalar_type() == at::kInt, "interpolation idx must be int32");
    TORCH_CHECK(weight_tensor.scalar_type() == at::kFloat, "interpolation weight must be float32");
    TORCH_CHECK(input_tensor.scalar_type() == output_tensor.scalar_type(), "interpolation input/output dtype mismatch");
    TORCH_CHECK(input_tensor.is_floating_point(), "interpolation input must be floating point");
    interpolation_forward_cuda_launcher(
        static_cast<int>(input_tensor.size(0)), n, c, k,
        input_tensor.data_ptr(), idx_tensor.data_ptr<int>(),
        weight_tensor.data_ptr<float>(), output_tensor.data_ptr(),
        input_tensor.scalar_type());
}

void interpolation_backward_cuda(int n, int c, int k, at::Tensor grad_output_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor grad_input_tensor)
{
    TORCH_CHECK(grad_output_tensor.is_cuda() && idx_tensor.is_cuda() && weight_tensor.is_cuda() && grad_input_tensor.is_cuda(), "interpolation tensors must be CUDA tensors");
    TORCH_CHECK(grad_output_tensor.get_device() == idx_tensor.get_device() && grad_output_tensor.get_device() == weight_tensor.get_device() && grad_output_tensor.get_device() == grad_input_tensor.get_device(), "interpolation tensors must be on the same device");
    TORCH_CHECK(grad_output_tensor.is_contiguous() && idx_tensor.is_contiguous() && weight_tensor.is_contiguous() && grad_input_tensor.is_contiguous(), "interpolation tensors must be contiguous");
    TORCH_CHECK(idx_tensor.scalar_type() == at::kInt && weight_tensor.scalar_type() == at::kFloat, "interpolation idx/weight dtype mismatch");
    TORCH_CHECK(grad_output_tensor.scalar_type() == grad_input_tensor.scalar_type(), "interpolation gradient dtype mismatch");
    interpolation_backward_cuda_launcher(
        static_cast<int>(grad_input_tensor.size(0)), n, c, k,
        grad_output_tensor.data_ptr(), idx_tensor.data_ptr<int>(),
        weight_tensor.data_ptr<float>(), grad_input_tensor.data_ptr(),
        grad_output_tensor.scalar_type());
}
