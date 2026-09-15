/*
CUDA implementation of PointROPE

Author: Yuanwen Yue (yuayue@ethz.ch)
Please cite our work if the code is helpful to you.
*/

#include <torch/extension.h>

// forward declaration
void pointrope_cuda( torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd );
torch::Tensor pointrope_bhnd_cuda(const torch::Tensor tokens, const torch::Tensor pos, const float base, const float fwd);

void pointrope_cpu( torch::Tensor tokens, const torch::Tensor positions, const float base, const float fwd )
{
    const int B = tokens.size(0);
    const int N = tokens.size(1);
    const int H = tokens.size(2); // number head, eg. 2
    const int D = tokens.size(3) / 6; // if dimension per head is 18, then D = 3

    auto tok = tokens.accessor<float, 4>();
    auto pos = positions.accessor<int64_t, 3>();

    for (int b = 0; b < B; b++) {
      for (int x = 0; x < 3; x++) { // x and then y then z (3d)
        for (int n = 0; n < N; n++) {
        
            // grab the token position
            const int p = pos[b][n][x];

            for (int h = 0; h < H; h++) {
                for (int d = 0; d < D; d++) {
                    // grab the two values
                    float u = tok[b][n][h][d+0+x*2*D];
                    float v = tok[b][n][h][d+D+x*2*D];

                    // grab the cos,sin
                    const float inv_freq = fwd * p / powf(base, d/float(D));
                    float c = cosf(inv_freq);
                    float s = sinf(inv_freq);

                    // write the result
                    tok[b][n][h][d+0+x*2*D] = u*c - v*s;
                    tok[b][n][h][d+D+x*2*D] = v*c + u*s;
                }
            }
        }
      }
    }
}

void pointrope( torch::Tensor tokens,   
        const torch::Tensor positions,
        const float base, 
        const float fwd )
{
    TORCH_CHECK(tokens.dim() == 4, "tokens must have 4 dimensions");
    TORCH_CHECK(positions.dim() == 3, "positions must have 3 dimensions");
    TORCH_CHECK(tokens.size(0) == positions.size(0), "batch size differs between tokens & positions");
    TORCH_CHECK(tokens.size(1) == positions.size(1), "seq_length differs between tokens & positions");
    TORCH_CHECK(positions.size(2) == 3, "positions.shape[2] must be equal to 3");
    TORCH_CHECK(tokens.is_cuda() == positions.is_cuda(), "tokens and positions are not on the same device" );

    if (tokens.is_cuda())
        pointrope_cuda( tokens, positions, base, fwd );
    else
        pointrope_cpu( tokens, positions, base, fwd );
}

torch::Tensor pointrope_bhnd(
        const torch::Tensor tokens,
        const torch::Tensor positions,
        const float base,
        const float fwd)
{
    TORCH_CHECK(tokens.is_cuda() && positions.is_cuda(), "pointrope_bhnd requires CUDA tensors");
    TORCH_CHECK(tokens.dim() == 4, "tokens must have shape [B, H, N, D]");
    TORCH_CHECK(positions.dim() == 3, "positions must have shape [B, N, 3]");
    TORCH_CHECK(tokens.size(0) == positions.size(0), "batch size differs between tokens and positions");
    TORCH_CHECK(tokens.size(2) == positions.size(1), "token count differs between tokens and positions");
    TORCH_CHECK(positions.size(2) == 3, "positions.shape[2] must equal 3");
    TORCH_CHECK(tokens.size(3) % 6 == 0, "head dimension must be a multiple of 6");
    TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(tokens.get_device() == positions.get_device(), "tokens and positions must use the same CUDA device");
    TORCH_CHECK(tokens.is_floating_point(), "tokens must have a floating-point dtype");
    return pointrope_bhnd_cuda(tokens, positions, base, fwd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pointrope", &pointrope, "PointROPE forward/backward");
  m.def("pointrope_bhnd", &pointrope_bhnd, "Out-of-place PointROPE for [B,H,N,D]");
}
