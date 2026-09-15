#ifndef _KNN_QUERY_CUDA_KERNEL
#define _KNN_QUERY_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void knn_query_cuda(int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor);
void knn_query_idx_cuda(int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor idx_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void knn_query_cuda_launcher(
    int n, int m, int batch_size, int nsample, int table_size,
    const float *xyz, const float *new_xyz,
    const int *offset, const int *new_offset,
    int *idx, float *dist2,
    float *batch_min, float *batch_max, float *cell_size,
    int *slot_state, int *slot_x, int *slot_y, int *slot_z,
    int *slot_head, int *point_next, int *fallback_count);

#ifdef __cplusplus
}
#endif
#endif
