"""Single-tile full-resolution feature export for CVRL framework assets."""

_base_ = ["../configs/cvrl/audit-cvrl-v1m1-litept-v1m4-swiss-100.py"]

tile_stem = "swisssurface3d_2020_2760-1188_2056_5728_0086"
tile_path = (
    rf"G:\data\swiss\joint_tiles_100\Swiss-Audit\pointcloud\{tile_stem}.las"
)

feature_source = "full"
feature_output_dir = "paper_cvrl/results/method/cvrl_framework_assets/full_feature"
feature_index_dir = None
feature_overwrite = False
batch_size_test = 2

data = dict(
    test=dict(
        data_path=None,
        data_list=[tile_path],
    )
)
