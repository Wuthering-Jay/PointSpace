"""
EZ-SP Base Configuration

Common settings for all EZ-SP experiments.
"""

# SparseCNN default configuration
sparse_cnn = dict(
    type="EZ-SparseCNN",
    in_channels=6,  # RGB + Normal or XYZ + RGB
    channels=[32, 32, 32],
    kernel_size=3,
    dilation=1,
    norm="gn",  # GraphNorm
    activation="relu",
    residual=True,
    global_residual=False,
)

# Partition module default configuration
partition_module = dict(
    type="GreedyContourPriorPartition",
    reg=2e-2,
    min_size=[5, 30, 90],
    k_adjacency=10,
    spatial_weight=None,  # Pure feature partition (EZ-SP default)
    edge_weight_mode="unit",
    d_0=None,
    w_adjacency=0.0,
    max_iterations=-1,
    edge_reduce="add",
)

# Partition criterion default configuration (Stage 1)
# Now using SPT-style probability-based focal loss
partition_criterion = dict(
    type="PartitionCriterion",
    gamma=1.0,              # Focal loss gamma
    alpha=0.5,              # Class balance weight
    temperature=1.0,        # Affinity temperature
    adaptive_sampling=True,
    adaptive_sampling_ratio=0.9,
    sharding=None,          # Set to int for large graphs
)

# Semantic segmentation criteria (Stage 2)
semantic_criteria = [
    dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
]
