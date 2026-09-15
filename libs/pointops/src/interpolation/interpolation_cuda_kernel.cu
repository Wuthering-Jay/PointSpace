#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "../cuda_utils.h"
#include "interpolation_cuda_kernel.h"

template <typename scalar_t>
__global__ void interpolation_forward_cuda_kernel(
    int m, int n, int c, int k, const scalar_t *input, const int *idx,
    const float *weight, scalar_t *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * c) return;
    int c_idx = index % c;
    int n_idx = index / c;
    float value = 0.0f;
    for (int i = 0; i < k; ++i) {
        int edge = n_idx * k + i;
        int support = idx[edge];
        if (support >= 0 && support < m) {
            value += static_cast<float>(input[support * c + c_idx]) * weight[edge];
        }
    }
    output[index] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void interpolation_backward_cuda_kernel(
    int m, int n, int c, int k, const scalar_t *grad_output, const int *idx,
    const float *weight, scalar_t *grad_input) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * c) return;
    int c_idx = index % c;
    int n_idx = index / c;
    float grad = static_cast<float>(grad_output[index]);
    for (int i = 0; i < k; ++i) {
        int edge = n_idx * k + i;
        int support = idx[edge];
        if (support >= 0 && support < m) {
            gpuAtomicAdd(
                grad_input + support * c + c_idx,
                static_cast<scalar_t>(grad * weight[edge]));
        }
    }
}

void interpolation_forward_cuda_launcher(
    int m, int n, int c, int k, const void *input, const int *idx,
    const float *weight, void *output, at::ScalarType scalar_type) {
    dim3 blocks(DIVUP(n * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type,
        "interpolation_forward_cuda", [&] {
            interpolation_forward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                m, n, c, k, static_cast<const scalar_t *>(input), idx, weight,
                static_cast<scalar_t *>(output));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void interpolation_backward_cuda_launcher(
    int m, int n, int c, int k, const void *grad_output, const int *idx,
    const float *weight, void *grad_input, at::ScalarType scalar_type) {
    dim3 blocks(DIVUP(n * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type,
        "interpolation_backward_cuda", [&] {
            interpolation_backward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                m, n, c, k, static_cast<const scalar_t *>(grad_output), idx,
                weight, static_cast<scalar_t *>(grad_input));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
