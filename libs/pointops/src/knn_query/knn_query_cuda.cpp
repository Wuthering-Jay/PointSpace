#include <vector>
#include <cmath>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "knn_query_cuda_kernel.h"


static void knn_query_cuda_impl(
    int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
    at::Tensor offset_tensor, at::Tensor new_offset_tensor,
    at::Tensor idx_tensor, at::Tensor dist2_tensor, bool write_distance)
{
    TORCH_CHECK(xyz_tensor.is_cuda() && new_xyz_tensor.is_cuda(), "xyz and new_xyz must be CUDA tensors");
    TORCH_CHECK(offset_tensor.is_cuda() && new_offset_tensor.is_cuda(), "offset and new_offset must be CUDA tensors");
    TORCH_CHECK(idx_tensor.is_cuda(), "idx must be a CUDA tensor");
    TORCH_CHECK(!write_distance || dist2_tensor.is_cuda(), "dist2 must be a CUDA tensor");
    TORCH_CHECK(xyz_tensor.get_device() == new_xyz_tensor.get_device() &&
                xyz_tensor.get_device() == offset_tensor.get_device() &&
                xyz_tensor.get_device() == new_offset_tensor.get_device() &&
                xyz_tensor.get_device() == idx_tensor.get_device() &&
                (!write_distance || xyz_tensor.get_device() == dist2_tensor.get_device()),
                "all knn_query tensors must be on the same CUDA device");
    TORCH_CHECK(xyz_tensor.scalar_type() == at::kFloat && new_xyz_tensor.scalar_type() == at::kFloat,
                "xyz and new_xyz must be float32 tensors");
    TORCH_CHECK(offset_tensor.scalar_type() == at::kInt && new_offset_tensor.scalar_type() == at::kInt,
                "offset and new_offset must be int32 tensors");
    TORCH_CHECK(idx_tensor.scalar_type() == at::kInt, "idx must be int32");
    TORCH_CHECK(!write_distance || dist2_tensor.scalar_type() == at::kFloat,
                "dist2 must be float32");
    TORCH_CHECK(xyz_tensor.is_contiguous() && new_xyz_tensor.is_contiguous() &&
                offset_tensor.is_contiguous() && new_offset_tensor.is_contiguous() &&
                idx_tensor.is_contiguous() &&
                (!write_distance || dist2_tensor.is_contiguous()),
                "all knn_query tensors must be contiguous");
    TORCH_CHECK(xyz_tensor.dim() == 2 && xyz_tensor.size(1) == 3, "xyz must have shape [N, 3]");
    TORCH_CHECK(new_xyz_tensor.dim() == 2 && new_xyz_tensor.size(1) == 3, "new_xyz must have shape [M, 3]");
    TORCH_CHECK(new_xyz_tensor.size(0) == m, "m must equal new_xyz.size(0)");
    TORCH_CHECK(offset_tensor.numel() == new_offset_tensor.numel(), "offset and new_offset must have equal batch counts");
    TORCH_CHECK(nsample > 0 && nsample <= 128, "knn_query supports 1 <= nsample <= 128");

    const int n = static_cast<int>(xyz_tensor.size(0));
    const int batch_size = static_cast<int>(offset_tensor.numel());
    TORCH_CHECK(n > 0 && m > 0 && batch_size > 0, "knn_query inputs must be non-empty");

    int table_size = 1;
    while (table_size < n * 2) table_size <<= 1;
    auto float_options = xyz_tensor.options().dtype(at::kFloat);
    auto int_options = xyz_tensor.options().dtype(at::kInt);
    auto batch_min_tensor = at::full({batch_size, 3}, INFINITY, float_options);
    auto batch_max_tensor = at::full({batch_size, 3}, -INFINITY, float_options);
    auto cell_size_tensor = at::empty({batch_size}, float_options);
    auto slot_state_tensor = at::zeros({table_size}, int_options);
    auto slot_x_tensor = at::empty({table_size}, int_options);
    auto slot_y_tensor = at::empty({table_size}, int_options);
    auto slot_z_tensor = at::empty({table_size}, int_options);
    auto slot_head_tensor = at::full({table_size}, -1, int_options);
    auto point_next_tensor = at::empty({n}, int_options);
    auto fallback_count_tensor = at::zeros({1}, int_options);

    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const int *offset = offset_tensor.data_ptr<int>();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    float *dist2 = write_distance ? dist2_tensor.data_ptr<float>() : nullptr;
    knn_query_cuda_launcher(
        n, m, batch_size, nsample, table_size,
        xyz, new_xyz, offset, new_offset, idx, dist2,
        batch_min_tensor.data_ptr<float>(), batch_max_tensor.data_ptr<float>(),
        cell_size_tensor.data_ptr<float>(), slot_state_tensor.data_ptr<int>(),
        slot_x_tensor.data_ptr<int>(), slot_y_tensor.data_ptr<int>(),
        slot_z_tensor.data_ptr<int>(), slot_head_tensor.data_ptr<int>(),
        point_next_tensor.data_ptr<int>(), fallback_count_tensor.data_ptr<int>());
}


void knn_query_cuda(
    int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
    at::Tensor offset_tensor, at::Tensor new_offset_tensor,
    at::Tensor idx_tensor, at::Tensor dist2_tensor)
{
    knn_query_cuda_impl(
        m, nsample, xyz_tensor, new_xyz_tensor, offset_tensor,
        new_offset_tensor, idx_tensor, dist2_tensor, true);
}


void knn_query_idx_cuda(
    int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
    at::Tensor offset_tensor, at::Tensor new_offset_tensor,
    at::Tensor idx_tensor)
{
    knn_query_cuda_impl(
        m, nsample, xyz_tensor, new_xyz_tensor, offset_tensor,
        new_offset_tensor, idx_tensor, at::Tensor(), false);
}
