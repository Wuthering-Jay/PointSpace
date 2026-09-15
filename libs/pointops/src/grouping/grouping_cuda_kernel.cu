#include "../cuda_utils.h"
#include "grouping_cuda_kernel.h"

#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAException.h>

template <typename scalar_t>
__global__ void grouping_forward_cuda_kernel(
    int n, int total, int c, const scalar_t *__restrict__ input,
    const int *__restrict__ idx, scalar_t *__restrict__ output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    int c_idx = index % c;
    int edge = index / c;
    int input_idx = idx[edge];
    output[index] = input_idx >= 0 && input_idx < n
        ? input[input_idx * c + c_idx]
        : static_cast<scalar_t>(0);
}

template <typename scalar_t>
__global__ void grouping_backward_cuda_kernel(
    int n, int total, int c, const scalar_t *__restrict__ grad_output,
    const int *__restrict__ idx, scalar_t *__restrict__ grad_input) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    int c_idx = index % c;
    int edge = index / c;
    int input_idx = idx[edge];
    if (input_idx >= 0 && input_idx < n) {
        gpuAtomicAdd(grad_input + input_idx * c + c_idx, grad_output[index]);
    }
}

void grouping_forward_cuda_launcher(
    int n, int m, int nsample, int c, const void *input, const int *idx,
    void *output, at::ScalarType dtype) {
    int total = m * nsample * c;
    dim3 blocks(DIVUP(total, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, dtype, "grouping_forward_cuda", ([&] {
            grouping_forward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                n, total, c, static_cast<const scalar_t *>(input), idx,
                static_cast<scalar_t *>(output));
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void grouping_backward_cuda_launcher(
    int n, int m, int nsample, int c, const void *grad_output, const int *idx,
    void *grad_input, at::ScalarType dtype) {
    int total = m * nsample * c;
    dim3 blocks(DIVUP(total, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, dtype, "grouping_backward_cuda", ([&] {
            grouping_backward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                n, total, c, static_cast<const scalar_t *>(grad_output), idx,
                static_cast<scalar_t *>(grad_input));
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
