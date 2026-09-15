#include "../cuda_utils.h"
#include "knn_query_cuda_kernel.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <math.h>
#include <stdint.h>

namespace knn_query_utils {

template <typename DType>
__device__ __forceinline__ void swap(DType *x, DType *y) {
    DType tmp = *x;
    *x = *y;
    *y = tmp;
}

__device__ __forceinline__ void reheap(float *dist, int *idx, int k) {
    int root = 0;
    int child = 1;
    while (child < k) {
        if (child + 1 < k &&
            (dist[child + 1] > dist[child] ||
             (dist[child + 1] == dist[child] && idx[child + 1] > idx[child]))) {
            child++;
        }
        if (dist[root] > dist[child] ||
            (dist[root] == dist[child] && idx[root] >= idx[child])) return;
        swap<float>(&dist[root], &dist[child]);
        swap<int>(&idx[root], &idx[child]);
        root = child;
        child = root * 2 + 1;
    }
}

__device__ __forceinline__ void heap_sort(float *dist, int *idx, int k) {
    for (int i = k - 1; i > 0; --i) {
        swap<float>(&dist[0], &dist[i]);
        swap<int>(&idx[0], &idx[i]);
        reheap(dist, idx, i);
    }
}

__device__ __forceinline__ int get_bt_idx(int idx, const int *offset, int batch_size) {
    int lo = 0;
    int hi = batch_size;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (idx < offset[mid]) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

__device__ __forceinline__ float atomic_min_float(float *address, float value) {
    int *address_as_i = reinterpret_cast<int *>(address);
    int old = *address_as_i;
    while (value < __int_as_float(old)) {
        int assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
    return __int_as_float(old);
}

__device__ __forceinline__ float atomic_max_float(float *address, float value) {
    int *address_as_i = reinterpret_cast<int *>(address);
    int old = *address_as_i;
    while (value > __int_as_float(old)) {
        int assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
    return __int_as_float(old);
}

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ uint32_t cell_hash(int batch, int x, int y, int z) {
    uint32_t h = mix32(static_cast<uint32_t>(x));
    h ^= mix32(static_cast<uint32_t>(y) + 0x9e3779b9U);
    h ^= mix32(static_cast<uint32_t>(z) + 0x85ebca6bU);
    h ^= mix32(static_cast<uint32_t>(batch) + 0xc2b2ae35U);
    return mix32(h);
}

__device__ __forceinline__ void consider_point(
    int point_idx, float qx, float qy, float qz, const float *xyz,
    int nsample, float *best_dist, int *best_idx, int *found) {
    float dx = qx - xyz[point_idx * 3];
    float dy = qy - xyz[point_idx * 3 + 1];
    float dz = qz - xyz[point_idx * 3 + 2];
    float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < best_dist[0] ||
        (d2 == best_dist[0] && point_idx < best_idx[0])) {
        if (*found < nsample) ++(*found);
        best_dist[0] = d2;
        best_idx[0] = point_idx;
        reheap(best_dist, best_idx, nsample);
    }
}

}  // namespace knn_query_utils

__global__ void bbox_kernel(
    int n, int batch_size, const float *__restrict__ xyz,
    const int *__restrict__ offset, float *__restrict__ batch_min,
    float *__restrict__ batch_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = knn_query_utils::get_bt_idx(i, offset, batch_size);
    if (b >= batch_size) return;
    #pragma unroll
    for (int d = 0; d < 3; ++d) {
        float value = xyz[i * 3 + d];
        knn_query_utils::atomic_min_float(&batch_min[b * 3 + d], value);
        knn_query_utils::atomic_max_float(&batch_max[b * 3 + d], value);
    }
}

__global__ void cell_size_kernel(
    int batch_size, const int *__restrict__ offset,
    const float *__restrict__ batch_min, const float *__restrict__ batch_max,
    float *__restrict__ cell_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    int start = b == 0 ? 0 : offset[b - 1];
    int count = max(offset[b] - start, 1);
    float e0 = fmaxf(batch_max[b * 3] - batch_min[b * 3], 0.0f);
    float e1 = fmaxf(batch_max[b * 3 + 1] - batch_min[b * 3 + 1], 0.0f);
    float e2 = fmaxf(batch_max[b * 3 + 2] - batch_min[b * 3 + 2], 0.0f);
    if (e0 < e1) knn_query_utils::swap<float>(&e0, &e1);
    if (e1 < e2) knn_query_utils::swap<float>(&e1, &e2);
    if (e0 < e1) knn_query_utils::swap<float>(&e0, &e1);

    float inv_n = 1.0f / static_cast<float>(count);
    float s1 = e0 * (8.0f * inv_n);
    float s2 = e1 > 0.0f ? sqrtf(e0 * e1 * (8.0f * inv_n)) : 0.0f;
    float s3 = e2 > 0.0f ? cbrtf(e0 * e1 * e2 * (8.0f * inv_n)) : 0.0f;
    float scale = fmaxf(s1, fmaxf(s2, s3));
    float extent_scale = fmaxf(e0, 1.0f) * 1e-7f;
    cell_size[b] = fmaxf(scale, extent_scale);
}

__global__ void build_hash_keys_kernel(
    int n, int batch_size, int table_mask, const float *__restrict__ xyz,
    const int *__restrict__ offset, const float *__restrict__ batch_min,
    const float *__restrict__ cell_size, int *__restrict__ slot_state,
    int *__restrict__ slot_x, int *__restrict__ slot_y, int *__restrict__ slot_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = knn_query_utils::get_bt_idx(i, offset, batch_size);
    if (b >= batch_size) return;
    float inv_cell = 1.0f / cell_size[b];
    int cx = __float2int_rd((xyz[i * 3] - batch_min[b * 3]) * inv_cell);
    int cy = __float2int_rd((xyz[i * 3 + 1] - batch_min[b * 3 + 1]) * inv_cell);
    int cz = __float2int_rd((xyz[i * 3 + 2] - batch_min[b * 3 + 2]) * inv_cell);
    int slot = static_cast<int>(knn_query_utils::cell_hash(b, cx, cy, cz) & table_mask);

    while (true) {
        int state = atomicCAS(&slot_state[slot], 0, -1);
        if (state == 0) {
            slot_x[slot] = cx;
            slot_y[slot] = cy;
            slot_z[slot] = cz;
            __threadfence();
            atomicExch(&slot_state[slot], b + 1);
            state = b + 1;
        } else if (state == -1) {
            do { state = atomicAdd(&slot_state[slot], 0); } while (state == -1);
        }
        if (state == b + 1 && slot_x[slot] == cx && slot_y[slot] == cy && slot_z[slot] == cz) {
            return;
        }
        slot = (slot + 1) & table_mask;
    }
}

__device__ __forceinline__ int find_cell(
    int batch, int x, int y, int z, int table_mask,
    const int *slot_state, const int *slot_x, const int *slot_y,
    const int *slot_z) {
    int slot = static_cast<int>(knn_query_utils::cell_hash(batch, x, y, z) & table_mask);
    while (true) {
        int state = slot_state[slot];
        if (state == 0) return -1;
        if (state == batch + 1 && slot_x[slot] == x && slot_y[slot] == y && slot_z[slot] == z) return slot;
        slot = (slot + 1) & table_mask;
    }
}

__global__ void link_points_kernel(
    int n, int batch_size, int table_mask, const float *__restrict__ xyz,
    const int *__restrict__ offset, const float *__restrict__ batch_min,
    const float *__restrict__ cell_size, const int *__restrict__ slot_state,
    const int *__restrict__ slot_x, const int *__restrict__ slot_y,
    const int *__restrict__ slot_z, int *__restrict__ slot_head,
    int *__restrict__ point_next) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = knn_query_utils::get_bt_idx(i, offset, batch_size);
    if (b >= batch_size) return;
    float inv_cell = 1.0f / cell_size[b];
    int cx = __float2int_rd((xyz[i * 3] - batch_min[b * 3]) * inv_cell);
    int cy = __float2int_rd((xyz[i * 3 + 1] - batch_min[b * 3 + 1]) * inv_cell);
    int cz = __float2int_rd((xyz[i * 3 + 2] - batch_min[b * 3 + 2]) * inv_cell);
    int slot = find_cell(b, cx, cy, cz, table_mask, slot_state, slot_x, slot_y, slot_z);
    if (slot >= 0) {
        int old_head;
        do {
            old_head = slot_head[slot];
            point_next[i] = old_head;
        } while (atomicCAS(&slot_head[slot], old_head, i) != old_head);
    }
}

__global__ void grid_knn_kernel(
    int m, int batch_size, int nsample, int table_mask,
    const float *__restrict__ xyz, const float *__restrict__ new_xyz,
    const int *__restrict__ offset, const int *__restrict__ new_offset,
    const float *__restrict__ batch_min, const float *__restrict__ cell_size,
    const int *__restrict__ slot_state, const int *__restrict__ slot_x,
    const int *__restrict__ slot_y, const int *__restrict__ slot_z,
    const int *__restrict__ slot_head, const int *__restrict__ point_next,
    int *__restrict__ idx, float *__restrict__ dist2,
    int *__restrict__ fallback_count) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= m) return;
    int b = knn_query_utils::get_bt_idx(q, new_offset, batch_size);
    if (b >= batch_size) return;
    int start = b == 0 ? 0 : offset[b - 1];
    int end = offset[b];
    float qx = new_xyz[q * 3];
    float qy = new_xyz[q * 3 + 1];
    float qz = new_xyz[q * 3 + 2];
    float cell = cell_size[b];
    float inv_cell = 1.0f / cell;
    int cx = __float2int_rd((qx - batch_min[b * 3]) * inv_cell);
    int cy = __float2int_rd((qy - batch_min[b * 3 + 1]) * inv_cell);
    int cz = __float2int_rd((qz - batch_min[b * 3 + 2]) * inv_cell);

    float best_dist[128];
    int best_idx[128];
    for (int i = 0; i < nsample; ++i) {
        best_dist[i] = 1e30f;
        best_idx[i] = -1;
    }
    int found = 0;
    bool exact = false;
    constexpr int max_grid_radius = 8;

    for (int radius = 0; radius <= max_grid_radius && !exact; ++radius) {
        for (int dx = -radius; dx <= radius; ++dx) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dz = -radius; dz <= radius; ++dz) {
                    if (radius > 0 && max(abs(dx), max(abs(dy), abs(dz))) != radius) continue;
                    int slot = find_cell(b, cx + dx, cy + dy, cz + dz, table_mask,
                                         slot_state, slot_x, slot_y, slot_z);
                    if (slot < 0) continue;
                    for (int p = slot_head[slot]; p >= 0; p = point_next[p]) {
                        knn_query_utils::consider_point(
                            p, qx, qy, qz, xyz, nsample, best_dist, best_idx, &found);
                    }
                }
            }
        }
        if (found >= nsample) {
            float low_x = batch_min[b * 3] + static_cast<float>(cx - radius) * cell;
            float low_y = batch_min[b * 3 + 1] + static_cast<float>(cy - radius) * cell;
            float low_z = batch_min[b * 3 + 2] + static_cast<float>(cz - radius) * cell;
            float high_x = batch_min[b * 3] + static_cast<float>(cx + radius + 1) * cell;
            float high_y = batch_min[b * 3 + 1] + static_cast<float>(cy + radius + 1) * cell;
            float high_z = batch_min[b * 3 + 2] + static_cast<float>(cz + radius + 1) * cell;
            float boundary = fminf(fminf(qx - low_x, high_x - qx),
                                   fminf(fminf(qy - low_y, high_y - qy),
                                         fminf(qz - low_z, high_z - qz)));
            boundary = fmaxf(boundary, 0.0f);
            exact = best_dist[0] <= boundary * boundary;
        }
    }

    if (!exact) {
        atomicAdd(fallback_count, 1);
        for (int i = 0; i < nsample; ++i) {
            best_dist[i] = 1e30f;
            best_idx[i] = -1;
        }
        found = 0;
        for (int p = start; p < end; ++p) {
            knn_query_utils::consider_point(
                p, qx, qy, qz, xyz, nsample, best_dist, best_idx, &found);
        }
    }

    knn_query_utils::heap_sort(best_dist, best_idx, nsample);
    for (int i = 0; i < nsample; ++i) {
        idx[q * nsample + i] = best_idx[i];
        if (dist2 != nullptr) dist2[q * nsample + i] = best_dist[i];
    }
}

void knn_query_cuda_launcher(
    int n, int m, int batch_size, int nsample, int table_size,
    const float *xyz, const float *new_xyz,
    const int *offset, const int *new_offset,
    int *idx, float *dist2,
    float *batch_min, float *batch_max, float *cell_size,
    int *slot_state, int *slot_x, int *slot_y, int *slot_z,
    int *slot_head, int *point_next, int *fallback_count) {
    dim3 point_blocks(DIVUP(n, THREADS_PER_BLOCK));
    dim3 query_blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 batch_blocks(DIVUP(batch_size, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bbox_kernel<<<point_blocks, threads, 0, stream>>>(n, batch_size, xyz, offset, batch_min, batch_max);
    cell_size_kernel<<<batch_blocks, threads, 0, stream>>>(batch_size, offset, batch_min, batch_max, cell_size);
    build_hash_keys_kernel<<<point_blocks, threads, 0, stream>>>(
        n, batch_size, table_size - 1, xyz, offset, batch_min, cell_size,
        slot_state, slot_x, slot_y, slot_z);
    link_points_kernel<<<point_blocks, threads, 0, stream>>>(
        n, batch_size, table_size - 1, xyz, offset, batch_min, cell_size,
        slot_state, slot_x, slot_y, slot_z, slot_head, point_next);
    grid_knn_kernel<<<query_blocks, threads, 0, stream>>>(
        m, batch_size, nsample, table_size - 1, xyz, new_xyz, offset, new_offset,
        batch_min, cell_size, slot_state, slot_x, slot_y, slot_z, slot_head,
        point_next, idx, dist2, fallback_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
