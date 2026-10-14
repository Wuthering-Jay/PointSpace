import pytest
import torch

from pointspace.models import build_model


def _config(frozen=True):
    return dict(
        type="LitePT-v1m4", in_channels=4,
        order=("z", "z-trans"), stride=(2, 2),
        enc_depths=(1, 1, 1), enc_channels=(12, 24, 48),
        enc_num_head=(2, 4, 8), enc_patch_size=(16, 16, 16),
        enc_conv=(True, True, False), enc_attn=(False, False, True),
        enc_rope_freq=(100., 100., 100.),
        dec_depths=(1, 1), dec_channels=(12, 24),
        dec_num_head=(2, 4), dec_patch_size=(16, 16),
        dec_conv=(True, True), dec_attn=(False, False),
        dec_rope_freq=(100., 100.), mlp_ratio=2, drop_path=0.,
        shuffle_orders=False, enable_flash=False, traceable=True,
        enc_mode=False, freeze_encoder=frozen,
    )


def test_cache_isolation_preserves_checkpoint_keys():
    ordinary = build_model(_config(False))
    frozen = build_model(_config(True))
    frozen.load_state_dict(ordinary.state_dict(), strict=True)
    assert frozen.state_dict().keys() == ordinary.state_dict().keys()
    encoder_keys = {m.indice_key for m in frozen.enc.modules()
                    if hasattr(m, "indice_key") and m.indice_key}
    decoder_keys = {m.indice_key for m in frozen.dec.modules()
                    if hasattr(m, "indice_key") and m.indice_key}
    assert encoder_keys.isdisjoint(decoder_keys)
    frozen.train()
    assert not frozen.embedding.training and not frozen.enc.training
    assert frozen.dec.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_frozen_encoder_trainable_decoder_backward(dtype):
    model = build_model(_config()).cuda().train()
    grid = torch.stack(torch.meshgrid(
        torch.arange(8), torch.arange(8), torch.arange(4), indexing="ij"
    ), dim=-1).reshape(-1, 3).cuda()
    data = dict(
        coord=grid.float(), grid_coord=grid.long(),
        feat=torch.randn(len(grid), 4, device="cuda"),
        offset=torch.tensor([len(grid)], dtype=torch.int32, device="cuda"),
    )
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            output = model(data)
            loss = output.feat.float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in model.enc.parameters())
        assert all(p.grad is None for p in model.embedding.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.dec.parameters())
