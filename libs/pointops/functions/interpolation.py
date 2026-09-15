import torch
from torch.autograd import Function

from pointops._C import interpolation_forward_cuda, interpolation_backward_cuda
from .query import knn_query


class Interpolation(Function):
    @staticmethod
    def forward(ctx, xyz, new_xyz, input, offset, new_offset, k=3):
        """
        input: coords: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        assert xyz.is_contiguous() and new_xyz.is_contiguous() and input.is_contiguous()
        idx, dist = knn_query(k, xyz, offset, new_xyz, new_offset)  # (n, k), (n, k)
        dist_recip = 1.0 / (dist + 1e-8)  # (n, k)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = (dist_recip / norm).float().contiguous()  # (n, k)
        idx = idx.int().contiguous()

        n, c, m = new_xyz.shape[0], input.shape[1], input.shape[0]
        output = input.new_empty((n, c))
        interpolation_forward_cuda(n, c, k, input, idx, weight, output)
        ctx.m, ctx.k = m, k
        ctx.save_for_backward(idx, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: coords: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        m, k = ctx.m, ctx.k
        idx, weight = ctx.saved_tensors
        n, c = grad_output.shape
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_zeros((m, c))
        interpolation_backward_cuda(n, c, k, grad_output, idx, weight, grad_input)
        return None, None, grad_input, None, None, None


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """Inverse-distance weighted KNN interpolation preserving feature dtype."""
    return Interpolation.apply(xyz, new_xyz, feat, offset, new_offset, k)


interpolation2 = Interpolation.apply
