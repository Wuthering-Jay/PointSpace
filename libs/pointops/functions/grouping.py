import torch
from torch.autograd import Function

from pointops._C import grouping_forward_cuda, grouping_backward_cuda


class Grouping(Function):
    @staticmethod
    def forward(ctx, input, idx):
        """
        input: input: (n, c), idx : (m, nsample)
        output: (m, nsample, c)
        """
        assert input.is_contiguous() and idx.is_contiguous()
        m, nsample, n, c = idx.shape[0], idx.shape[1], input.shape[0], input.shape[1]
        output = input.new_empty((m, nsample, c))
        grouping_forward_cuda(m, nsample, c, input, idx, output)
        ctx.n = n
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (m, c, nsample)
        output: (n, c), None
        """
        n = ctx.n
        (idx,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        m, nsample, c = grad_output.shape
        grad_input = grad_output.new_zeros((n, c))
        grouping_backward_cuda(m, nsample, c, grad_output, idx, grad_input)
        return grad_input, None


def grouping(idx, feat, xyz, new_xyz=None, with_xyz=False):
    if new_xyz is None:
        new_xyz = xyz
    assert xyz.is_contiguous() and feat.is_contiguous()
    m, nsample, c = idx.shape[0], idx.shape[1], feat.shape[1]
    if feat.is_cuda:
        idx_cuda = idx.to(device=feat.device, dtype=torch.int32).contiguous()
        grouped_feat = grouping2(feat, idx_cuda)
    else:
        valid = (idx >= 0) & (idx < feat.shape[0])
        safe_idx = idx.clamp(0, max(feat.shape[0] - 1, 0)).long()
        grouped_feat = feat[safe_idx] * valid.unsqueeze(-1)

    if with_xyz:
        assert new_xyz.is_contiguous()
        valid = (idx >= 0) & (idx < xyz.shape[0])
        if xyz.is_cuda:
            idx_cuda = idx.to(device=xyz.device, dtype=torch.int32).contiguous()
            grouped_xyz = grouping2(xyz, idx_cuda) - new_xyz.unsqueeze(1)
        else:
            safe_idx = idx.clamp(0, max(xyz.shape[0] - 1, 0)).long()
            grouped_xyz = xyz[safe_idx] - new_xyz.unsqueeze(1)
        grouped_xyz = grouped_xyz * valid.to(grouped_xyz.device).unsqueeze(-1)
        return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        return grouped_feat


grouping2 = Grouping.apply
