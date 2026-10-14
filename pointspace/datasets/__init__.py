# On Windows, load PROJ before other dataset extensions. Some native libraries
# imported by the indoor/outdoor dataset modules expose DLLs with overlapping
# dependency names; loading them first can make pyproj._context fail even when
# the pyproj installation itself is valid.
import pyproj  # noqa: F401

from .defaults import DefaultDataset, DefaultImagePointDataset, ConcatDataset
from .builder import build_dataset
from .utils import point_collate_fn, collate_fn

# indoor scene
from .s3dis import S3DISDataset
from .scannet import ScanNetDataset, ScanNet200Dataset
from .scannetpp import ScanNetPPDataset
from .scannet_pair import ScanNetPairDataset
from .hm3d import HM3DDataset
from .structure3d import Structured3DDataset
from .aeo import AEODataset

# outdoor scene
from .semantic_kitti import SemanticKITTIDataset
from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset
from .las import LasDataset, LasImageDataset, RRCLLasDataset, CVRLLasDataset, MSCLasPairDataset, SonataLasPairDataset, ConcertoLasPairDataset
from .sparse_las import SparseLabelLasDataset

# object
from .modelnet import ModelNetDataset
from .shapenet_part import ShapeNetPartDataset

# dataloader
from .dataloader import MultiDatasetDataloader
