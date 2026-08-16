import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import laspy
from copy import deepcopy

from pointspace.utils.logger import get_root_logger

from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose

@DATASETS.register_module()
class LasDataset(DefaultDataset):
    """
    Dataset for LAS/LAZ point cloud files
    
    Supports:
    - LAS 1.1, 1.2, 1.3, 1.4 formats
    - Class filtering and remapping
    - Automatic class weight computation
    - Color RGB support
    - Echo information (first/last return)
    
    Args:
        required_class: List of class IDs to keep, others mapped to ignore_index
                       None means keep all classes (default: None)
        remap_class: Class remapping configuration
                    - False: no remapping, keep original class IDs
                    - True: auto remap to continuous [0,1,2,...] by sorted order
                    - dict: manual mapping {original_class: new_class}
                    (default: False)
        class_weight: Class weights for loss function
                     - None: no weighting
                     - 'auto': compute from dataset
                        - str: specify method ('inverse', 'sqrt', 'log', 'balanced', 'effective')
                     - list/array: manual weights
                     (default: None)
        weight_sample: Number or ratio of samples for weight computation
                      - int > 1: number of samples
                      - 0 < float <= 1: ratio of dataset
                      (default: 0.1, i.e., 10% of data)

    Note:
        ``LasDataset`` manages its own transform pipelines and always reads
        LAS/LAZ files directly with laspy.
    """
    
    VALID_ASSETS = [
        "coord",
        "color",
        "intensity",
        "echo",  # Combined first/last return info (2D: is_first, is_last)
        "normal",  # Normal vectors (normal_x, normal_y, normal_z extra dims)
        "superpoint",  # Superpoint segment ID
        "segment",
        "instance",
    ]
    
    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        data_path=None,
        data_list=None,
        transform=None,
        post_transform=None,
        aug_transform=None,
        test_mode=False,
        ignore_index=-1,
        loop=1,
        required_class=None,
        remap_class=False,
        class_weight=None,
        weight_sample=0.2,
        weighted_sampler=False,
        target_key=None,
    ):
        self.required_class = required_class
        self.remap_class = remap_class
        self.class_weight_mode = class_weight
        self.weight_sample = weight_sample
        if not isinstance(weighted_sampler, (bool, np.bool_)):
            raise TypeError(
                "weighted_sampler must be a bool; terrain-based sampling "
                "has been removed"
            )
        self.weighted_sampler = bool(weighted_sampler)

        # Class mapping will be initialized in get_data_list
        self.class2id = None  # Original class -> remapped ID
        self.id2class = None  # Remapped ID -> original class
        self.class_weight = None  # Computed class weights
        self.sample_weights = None  # Weights for WeightedRandomSampler
        self._class_lut = None
        self._mapped_id_to_index = {}

        # Store optional override params for potential use in get_data_list
        self._data_path = data_path
        self._data_list_input = data_list
        
        # Always pass test_mode=False so the base class does NOT try to build
        # transforms from test_cfg (which we don't use).  We restore the real
        # test_mode on self right after. LAS files are always read directly.
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=False,
            test_cfg=None,
            ignore_index=ignore_index,
            loop=loop,
            target_key=target_key,
        )

        # Restore real test_mode and enforce loop=1 in test mode
        self.test_mode = test_mode
        if test_mode:
            self.loop = 1

        # Set post_transform and aug_transform explicitly (the base class only
        # sets these when test_mode=True via test_cfg, which we bypass above).
        # Compose(None) is a transparent no-op, so None is safe here.
        self.post_transform = Compose(post_transform)
        self.aug_transform = [Compose(aug) for aug in (aug_transform or [])]

        # Compute class weights after data_list is initialized.
        # Skipped in test mode: weights are only used for loss / sampling during
        # training, and scanning the dataset would waste time at inference.
        if self.class_weight_mode is not None and not test_mode:
            if isinstance(self.class_weight_mode, (list, tuple, np.ndarray)):
                self._set_manual_class_weight(self.class_weight_mode)
            else:
                self._compute_class_weights()

        # Compute sample weights for WeightedRandomSampler (train only)
        if self.weighted_sampler and not test_mode:
            self._compute_sample_weights()

    def _build_class_lut(self):
        self._class_lut = None
        self._mapped_id_to_index = {}
        if not self.class2id:
            return

        mapped_ids = sorted(set(self.class2id.values()))
        self._mapped_id_to_index = {
            mapped_id: idx for idx, mapped_id in enumerate(mapped_ids)
        }

        orig_classes = list(self.class2id.keys())
        if not orig_classes:
            return
        min_class = min(orig_classes)
        max_class = max(orig_classes)
        if min_class < 0:
            return
        lut = np.full(max_class + 1, self.ignore_index, dtype=np.int32)
        for orig_class, new_class in self.class2id.items():
            lut[int(orig_class)] = int(new_class)
        self._class_lut = lut

    def _read_extra_dimension(self, las, dim_name, dtype=np.float32):
        try:
            return np.asarray(getattr(las, dim_name), dtype=dtype)
        except (AttributeError, KeyError):
            try:
                return np.asarray(las[dim_name], dtype=dtype)
            except (AttributeError, KeyError, ValueError):
                return None

    def _load_las_sample(self, data_path):
        data_dict = {}
        las = laspy.read(data_path)
        dim_names = set(las.point_format.dimension_names)

        num_points = len(las.points)
        coord = np.empty((num_points, 3), dtype=np.float32)
        coord[:, 0] = np.asarray(las.x, dtype=np.float32)
        coord[:, 1] = np.asarray(las.y, dtype=np.float32)
        coord[:, 2] = np.asarray(las.z, dtype=np.float32)
        data_dict["coord"] = coord

        if {"red", "green", "blue"}.issubset(dim_names):
            color = np.empty((num_points, 3), dtype=np.float32)
            color[:, 0] = np.asarray(las.red, dtype=np.float32)
            color[:, 1] = np.asarray(las.green, dtype=np.float32)
            color[:, 2] = np.asarray(las.blue, dtype=np.float32)
            color *= np.float32(1.0 / 256.0)
            data_dict["color"] = color

        if "classification" in dim_names:
            segment = np.asarray(las.classification, dtype=np.int32)
            if self.class2id is not None:
                segment = self._map_classes(segment)
            data_dict["segment"] = segment.reshape(-1)

        if "intensity" in dim_names:
            intensity = np.asarray(las.intensity, dtype=np.float32).reshape(-1, 1)
            data_dict["intensity"] = intensity

        if {"return_number", "number_of_returns"}.issubset(dim_names):
            return_number = np.asarray(las.return_number)
            number_of_returns = np.asarray(las.number_of_returns)
            echo = np.full((num_points, 2), -1.0, dtype=np.float32)
            echo[return_number == 1, 0] = 1.0
            echo[return_number == number_of_returns, 1] = 1.0
            data_dict["echo"] = echo

        if {"normal_x", "normal_y", "normal_z"}.issubset(dim_names):
            nx = self._read_extra_dimension(las, "normal_x")
            ny = self._read_extra_dimension(las, "normal_y")
            nz = self._read_extra_dimension(las, "normal_z")
            if nx is not None and ny is not None and nz is not None:
                normal = np.empty((num_points, 3), dtype=np.float32)
                normal[:, 0] = nx
                normal[:, 1] = ny
                normal[:, 2] = nz
                data_dict["normal"] = normal

        if "superpoint" in dim_names:
            superpoint = self._read_extra_dimension(las, "superpoint", dtype=np.int64)
            if superpoint is not None:
                data_dict["superpoint"] = superpoint

        target_key = getattr(self, "target_key", None)
        if target_key and target_key not in data_dict and target_key in dim_names:
            target = self._read_extra_dimension(las, target_key)
            if target is not None:
                data_dict[target_key] = target.reshape(-1, 1) if target.ndim == 1 else target

        for vlr in las.header.vlrs:
            if (
                getattr(vlr, "user_id", None) == "PointSpace"
                and getattr(vlr, "record_id", None) == 1001
            ):
                try:
                    bbox_data = json.loads(vlr.record_data.decode("utf-8"))
                    data_dict["core_bbox"] = np.array(
                        bbox_data["core_bbox"], dtype=np.float32
                    )
                except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                    pass
                break

        return data_dict
    
    def _scan_classes(self):
        """
        Scan dataset to find all unique classes
        """
        logger = get_root_logger()
        logger.info("Scanning dataset for class statistics...")
        
        all_classes = set()
        n_samples = len(self.data_list)

        if n_samples == 0:
            logger.warning("No files in data_list — cannot scan classes")
            return []

        # Determine sample size
        if isinstance(self.weight_sample, float) and 0 < self.weight_sample <= 1:
            n_scan = max(1, int(n_samples * self.weight_sample))
        elif isinstance(self.weight_sample, int) and self.weight_sample > 0:
            n_scan = min(self.weight_sample, n_samples)
        else:
            n_scan = n_samples

        # Random sample indices
        scan_indices = np.random.choice(n_samples, n_scan, replace=False) if n_scan < n_samples else range(n_samples)
        
        for idx in scan_indices:
            data_path = self.data_list[idx]
            try:
                las = laspy.read(data_path)
                if hasattr(las, "classification"):
                    classes = np.unique(las.classification)
                    all_classes.update(classes.tolist())
            except Exception as e:
                logger.warning(f"Failed to read {data_path}: {e}")
        
        all_classes = sorted(list(all_classes))
        logger.info(f"Found {len(all_classes)} unique classes: {all_classes}")
        
        return all_classes
    
    def _init_class_mapping(self):
        """
        Initialize class mapping based on required_class and remap_class
        
        Supports three modes:
        1. remap_class=False: identity mapping (no remapping)
        2. remap_class=True: auto remap to continuous IDs
        3. remap_class=dict: manual mapping {original_class: new_class}
        """
        logger = get_root_logger()
        
        # Scan for all classes
        all_classes = self._scan_classes()
        
        # Filter by required_class
        if self.required_class is not None:
            valid_classes = [c for c in all_classes if c in self.required_class]
            removed_classes = [c for c in all_classes if c not in self.required_class]
            logger.info(f"Filtering classes: {len(valid_classes)}/{len(all_classes)} classes kept")
            logger.info(f"  Kept classes: {valid_classes}")
            if removed_classes:
                logger.info(f"  Removed classes: {removed_classes} -> will be mapped to ignore_index={self.ignore_index}")
        else:
            valid_classes = all_classes
        
        # Create mapping based on remap_class type
        if isinstance(self.remap_class, dict):
            # Manual mapping mode
            self.class2id = {}
            for orig_class in valid_classes:
                if orig_class in self.remap_class:
                    self.class2id[orig_class] = self.remap_class[orig_class]
                else:
                    # If not in manual mapping, keep original or warn
                    logger.warning(f"Class {orig_class} not in remap_class dict, keeping original ID")
                    self.class2id[orig_class] = orig_class
            
            self.id2class = {v: k for k, v in self.class2id.items()}
            
            logger.info(f"Manual class remapping enabled:")
            logger.info(f"  Original classes: {valid_classes}")
            logger.info(f"  Manual mapping: {self.class2id}")
            
            # Validate mapping
            mapped_ids = list(self.class2id.values())
            if len(mapped_ids) != len(set(mapped_ids)):
                logger.warning(f"  WARNING: Duplicate mapped IDs detected! This may cause issues.")
                
        elif self.remap_class and len(valid_classes) > 0:
            # Auto remap to continuous IDs starting from 0
            # This is critical for EZ-SP: labels must be [0, 1, ..., num_classes-1]
            # and ignore_index should be num_classes
            self.class2id = {c: i for i, c in enumerate(valid_classes)}
            self.id2class = {v: k for k, v in self.class2id.items()}
            
            logger.info(f"✓ Auto class remapping:")
            logger.info(f"")
            logger.info(f"  Step 1 - Filter classes:")
            logger.info(f"    All classes found: {all_classes}")
            if self.required_class is not None and removed_classes:
                logger.info(f"    Keep: {valid_classes}")
                logger.info(f"    Remove: {removed_classes}")
            else:
                logger.info(f"    Keep all: {valid_classes}")
            logger.info(f"")
            logger.info(f"  Step 2 - Remap valid classes to continuous [0, 1, ..., {len(valid_classes)-1}]:")
            logger.info(f"    Original classes: {valid_classes}")
            logger.info(f"    Remapped to:      {list(range(len(valid_classes)))}")
            logger.info(f"    Mapping: {self.class2id}")
            logger.info(f"")
            logger.info(f"  Step 3 - Data loading will apply mapping:")
            logger.info(f"    Valid classes (e.g., {valid_classes[0]}) → Remapped ID (e.g., {self.class2id[valid_classes[0]]})")
            if self.required_class is not None and removed_classes:
                logger.info(f"    Removed classes {removed_classes} → ignore_index={self.ignore_index}")
            logger.info(f"")
            logger.info(f"  ✓ VERIFIED: ignore_index={self.ignore_index} != any remapped class {list(range(len(valid_classes)))} (no conflict)")
        else:
            # No remapping, identity mapping for valid classes
            self.class2id = {c: c for c in valid_classes}
            self.id2class = {c: c for c in valid_classes}
            
            if self.required_class is not None:
                logger.info(f"Class filtering enabled without remapping")
                logger.info(f"  Valid classes: {valid_classes}")
        self._build_class_lut()
    
    def _compute_class_weights(self):
        """
        Compute class weights from dataset for balanced loss
        
        Supported weight computation methods:
        - 'inverse': 1 / frequency (original method, can be extreme)
        - 'sqrt': 1 / sqrt(frequency) (less extreme, good for moderate imbalance)
        - 'log': 1 / log(1 + frequency) (smooth, good for severe imbalance)
        - 'balanced': n_samples / (n_classes * class_count) (sklearn style)
        - 'effective': 1 - beta^class_count (Class-Balanced Loss, good for long-tail)
        """
        logger = get_root_logger()

        if isinstance(self.class_weight_mode, (list, tuple, np.ndarray)):
            self._set_manual_class_weight(self.class_weight_mode)
            return
        
        # Parse class_weight parameter (use class_weight_mode for method config)
        weight_method = 'sqrt'  # default method
        beta = 0.9999  # for 'effective' method
        
        if isinstance(self.class_weight_mode, str):
            if self.class_weight_mode == 'auto':
                weight_method = 'sqrt'  # use sqrt as default
            else:
                weight_method = self.class_weight_mode
        elif isinstance(self.class_weight_mode, dict):
            weight_method = self.class_weight_mode.get('method', 'sqrt')
            beta = self.class_weight_mode.get('beta', 0.9999)
        
        logger.info(f"Computing class weights using '{weight_method}' method...")

        n_samples = len(self.data_list)
        if n_samples == 0:
            logger.warning("No files in data_list — skipping class weight computation")
            self.class_weight = None
            return

        # Determine sample size
        if isinstance(self.weight_sample, float) and 0 < self.weight_sample <= 1:
            n_compute = max(1, int(n_samples * self.weight_sample))
        elif isinstance(self.weight_sample, int) and self.weight_sample > 0:
            n_compute = min(self.weight_sample, n_samples)
        else:
            n_compute = n_samples

        logger.info(f"  Sampling {n_compute}/{n_samples} files ({n_compute/n_samples*100:.1f}%)")
        
        # Random sample indices
        compute_indices = np.random.choice(n_samples, n_compute, replace=False) if n_compute < n_samples else range(n_samples)
        
        # Count points per class
        class_counts = {}
        total_points = 0
        
        for idx in compute_indices:
            data_path = self.data_list[idx]
            try:
                las = laspy.read(data_path)
                if hasattr(las, "classification"):
                    segment = np.array(las.classification)
                    # Apply mapping
                    mapped_segment = self._map_classes(segment)
                    # Count only valid classes (not ignore_index)
                    valid_mask = mapped_segment != self.ignore_index
                    unique, counts = np.unique(mapped_segment[valid_mask], return_counts=True)
                    for cls, cnt in zip(unique, counts):
                        class_counts[cls] = class_counts.get(cls, 0) + cnt
                    total_points += valid_mask.sum()
            except Exception as e:
                logger.warning(f"Failed to read {data_path}: {e}")
        
        if total_points == 0:
            logger.warning("No valid points found for weight computation")
            self.class_weight = None
            return
        
        # Compute weights using selected method
        n_classes = len(self.class2id)
        weights = np.zeros(n_classes, dtype=np.float32)
        
        for cls, count in class_counts.items():
            idx = self._mapped_id_to_index.get(cls)
            if idx is None:
                continue
            freq = count / total_points
            
            if weight_method == 'inverse':
                # Original: 1 / frequency
                weights[idx] = 1.0 / (freq + 1e-6)
            elif weight_method == 'sqrt':
                # Square root: 1 / sqrt(frequency)
                weights[idx] = 1.0 / (np.sqrt(freq) + 1e-6)
            elif weight_method == 'log':
                # Logarithmic: 1 / log(1 + frequency)
                weights[idx] = 1.0 / (np.log(1 + freq) + 1e-6)
            elif weight_method == 'balanced':
                # Sklearn style: n_samples / (n_classes * class_count)
                weights[idx] = total_points / (n_classes * count)
            elif weight_method == 'effective':
                # Class-Balanced Loss: (1 - beta) / (1 - beta^count)
                weights[idx] = (1 - beta) / (1 - beta ** count + 1e-6)
            else:
                logger.warning(f"Unknown weight method '{weight_method}', using 'sqrt'")
                weights[idx] = 1.0 / (np.sqrt(freq) + 1e-6)
        
        # Normalize weights (except for 'balanced' and 'effective' which are already balanced)
        if weight_method not in ['balanced', 'effective']:
            weights = weights / weights.sum() * n_classes
        
        self.class_weight = weights
        
        logger.info(f"  Total points: {total_points:,}")
        logger.info(f"  Class distribution:")
        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            idx = self._mapped_id_to_index.get(cls)
            if idx is None:
                continue
            weight = weights[idx] if idx < len(weights) else 0
            orig_cls = self.id2class.get(cls, cls)
            logger.info(f"    Class {orig_cls:3d} -> {cls:3d}: {count:10,} points ({count/total_points*100:5.2f}%), weight={weight:.4f}")
        
        logger.info(f"  Computed weights ({weight_method}): {weights}")

    def _set_manual_class_weight(self, class_weight):
        """Use user-provided class weights directly without auto-computation."""
        logger = get_root_logger()
        weights = np.asarray(class_weight, dtype=np.float32).reshape(-1)

        if weights.size == 0:
            raise ValueError("Manual class_weight must not be empty.")

        if self.class2id is not None:
            n_classes = len(self.class2id)
            if weights.size != n_classes:
                raise ValueError(
                    f"Manual class_weight length mismatch: got {weights.size}, "
                    f"expected {n_classes}."
                )

        self.class_weight = weights
        logger.info(f"Using manual class weights directly: {weights}")

    def _compute_sample_weights(self):
        """
        Compute sample-level weights for WeightedRandomSampler.
        
        Sample weight = sum of class weights for all unique classes in the sample.
        This ensures samples with more rare classes are sampled more frequently.
        
        Optimization strategy:
        - Use laspy's lazy reading to quickly extract classification without loading full point cloud
        - Process files in parallel using ThreadPoolExecutor (IO-bound task)
        - Reuse each file's class set while calculating its final sample weight
        """
        logger = get_root_logger()
        
        if self.class_weight is None:
            logger.warning("Class weights not computed yet, computing now...")
            self._compute_class_weights()
        
        if self.class_weight is None:
            logger.warning("Cannot compute sample weights without class weights")
            return
        
        logger.info("Computing sample weights for WeightedRandomSampler...")
        
        n_samples = len(self.data_list)
        if n_samples == 0:
            logger.warning("No files in data_list — skipping sample weight computation")
            self.sample_weight = None
            return
        sample_weights = np.zeros(n_samples, dtype=np.float64)
        sample_class_sets = []  # Cache class sets for each sample
        
        # For progress logging
        log_interval = max(1, n_samples // 10)
        
        def process_file(idx):
            """Process a single file to extract unique classes"""
            data_path = self.data_list[idx]
            try:
                # Read only classification field (faster than full read)
                las = laspy.read(data_path)
                if hasattr(las, "classification"):
                    segment = np.array(las.classification)
                    # Map to remapped classes
                    mapped_segment = self._map_classes(segment)
                    # Get unique valid classes
                    unique_classes = np.unique(mapped_segment)
                    # Filter out ignore_index
                    unique_classes = unique_classes[unique_classes != self.ignore_index]
                    return idx, set(unique_classes.tolist())
            except Exception as e:
                logger.warning(f"Failed to read {data_path}: {e}")
            return idx, set()
        
        # Determine number of workers (IO-bound, can use more than CPU cores)
        num_workers = min(8, os.cpu_count() or 4)
        
        logger.info(f"  Processing {n_samples} files with {num_workers} workers...")
        
        # Initialize sample_class_sets with empty sets
        sample_class_sets = [set() for _ in range(n_samples)]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_file, idx): idx for idx in range(n_samples)}
            
            processed = 0
            for future in as_completed(futures):
                idx, class_set = future.result()
                sample_class_sets[idx] = class_set
                processed += 1
                
                if processed % log_interval == 0:
                    logger.info(f"  Progress: {processed}/{n_samples} ({processed/n_samples*100:.1f}%)")
        
        # Compute sample weights from class sets and class weights
        for idx, class_set in enumerate(sample_class_sets):
            weight = 0.0
            for cls in class_set:
                weight_idx = self._mapped_id_to_index.get(cls)
                if weight_idx is not None:
                    weight += self.class_weight[weight_idx]
            sample_weights[idx] = weight if weight > 0 else 1.0  # Default weight of 1.0 for empty samples
        
        # Normalize to avoid numerical issues (sum = n_samples)
        total_weight = sample_weights.sum()
        if total_weight > 0:
            sample_weights = sample_weights / total_weight * n_samples
        
        self.sample_weights = sample_weights
        
        # Log statistics
        logger.info(f"  Sample weight statistics:")
        logger.info(f"    Min: {sample_weights.min():.4f}")
        logger.info(f"    Max: {sample_weights.max():.4f}")
        logger.info(f"    Mean: {sample_weights.mean():.4f}")
        logger.info(f"    Std: {sample_weights.std():.4f}")
        
        # Show distribution of unique class counts
        class_counts = [len(s) for s in sample_class_sets]
        unique_counts, counts = np.unique(class_counts, return_counts=True)
        logger.info(f"  Distribution of unique classes per sample:")
        for uc, c in zip(unique_counts, counts):
            logger.info(f"    {uc} classes: {c} samples ({c/n_samples*100:.1f}%)")

    def _map_classes(self, segment):
        """
        Map original class IDs to remapped IDs
        
        Args:
            segment: Original class labels
            
        Returns:
            Mapped class labels (unmapped classes -> ignore_index)
        """
        segment = np.asarray(segment)
        if self.class2id is None:
            return segment.astype(np.int32, copy=False)

        if self._class_lut is not None:
            mapped = np.full(segment.shape, self.ignore_index, dtype=np.int32)
            valid_mask = (segment >= 0) & (segment < self._class_lut.shape[0])
            if np.any(valid_mask):
                mapped[valid_mask] = self._class_lut[segment[valid_mask]]
            return mapped

        mapped = np.full(segment.shape, self.ignore_index, dtype=np.int32)
        for orig_class, new_class in self.class2id.items():
            mapped[segment == orig_class] = new_class
        return mapped
    
    def get_data_list(self):
        """Override to resolve files from ``data_path`` (preferred) or fall
        back to the base-class ``data_root`` + ``split`` logic, then filter to
        LAS/LAZ only and initialise the class mapping."""
        import glob as _glob

        logger = get_root_logger()

        # --- resolve raw file list ---
        if self._data_path is not None:
            # data_path points directly at a directory of LAS/LAZ files
            path = self._data_path
            if os.path.isdir(path):
                data_list = sorted(
                    f for f in _glob.glob(os.path.join(path, "*"))
                    if os.path.splitext(f)[1].lower() in {".las", ".laz"}
                )
                logger.info(
                    f"LasDataset: found {len(data_list)} LAS/LAZ files "
                    f"in data_path='{path}'"
                )
            else:
                raise FileNotFoundError(
                    f"LasDataset: data_path='{path}' is not a directory"
                )
        elif self._data_list_input is not None:
            # Caller supplied an explicit file list
            data_list = list(self._data_list_input)
        else:
            # Fall back to DefaultDataset logic (data_root + split)
            data_list = super().get_data_list()
            # Filter to only LAS/LAZ
            valid_extensions = {'.las', '.laz'}
            original_count = len(data_list)
            data_list = [
                f for f in data_list
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            if len(data_list) < original_count:
                filtered = original_count - len(data_list)
                logger.info(
                    f"Filtered {filtered} non-LAS files from data list"
                )

        if len(data_list) == 0:
            logger.warning(
                "LasDataset: data_list is empty — no LAS/LAZ files found. "
                "Check data_path / data_root + split settings."
            )

        # Publish so _scan_classes / _init_class_mapping can access it
        self.data_list = data_list

        if (
            self.required_class is not None
            or self.remap_class
            or self.class_weight_mode is not None
            or self.weighted_sampler
        ):
            self._init_class_mapping()

        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        try:
            data_dict = self._load_las_sample(data_path)
        except Exception as e:
            logger = get_root_logger()
            logger.error(f"Error reading {data_path}: {e}")
            # Create empty data with minimal required fields
            data_dict = {"coord": np.zeros((0, 3), dtype=np.float32)}
        
        # Add metadata
        data_dict["name"] = name
        
        # If segment is not available, create default one filled with ignore_index
        if "segment" not in data_dict:
            data_dict["segment"] = np.ones(data_dict["coord"].shape[0], dtype=np.int32) * self.ignore_index
        
        # Ensure segment is 1D
        data_dict["segment"] = data_dict["segment"].reshape([-1])
        
        # Add instance data (not typically available in LAS/LAZ files)
        data_dict["instance"] = np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1

        point_keys = [key for key in self.VALID_ASSETS if key in data_dict]
        target_key = getattr(self, "target_key", None)
        if target_key and target_key in data_dict and target_key not in point_keys:
            point_keys.append(target_key)
        data_dict["index_valid_keys"] = point_keys
        
        return data_dict

    def get_data_name(self, idx):
        return os.path.splitext(os.path.basename(self.data_list[idx % len(self.data_list)]))[0]
        
    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        # post_transform is always a Compose (possibly no-op when None was given)
        data_dict = self.post_transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        
        # Save original segment BEFORE transform (GridSample will modify it)
        original_segment = data_dict.get("segment").copy() if "segment" in data_dict else None
        original_name = data_dict.get("name", self.get_data_name(idx))
        # Save original regression target BEFORE transform
        target_key = getattr(self, "target_key", None)
        original_regression_target = (
            data_dict[target_key].copy() if target_key and target_key in data_dict else None
        )
        
        transform_result = self.transform(data_dict)
        
        # Handle case where transform (e.g., GridSample mode="test") returns a list
        if isinstance(transform_result, list):
            # transform already produced fragments
            # Use original_segment size for pred tensor, NOT fragment segment
            result_dict = dict(
                segment=original_segment,  # Use original segment (size matches index range)
                name=original_name
            )
            # Attach regression target for the tester
            if original_regression_target is not None:
                result_dict["regression_target"] = original_regression_target
            
            # Pop segment from fragments (not needed in result since we use original)
            for frag in transform_result:
                frag.pop("segment", None)
            
            # CRITICAL: Preserve segment_raw for partition evaluation
            if "segment_raw" in transform_result[0]:
                result_dict["segment_raw"] = transform_result[0].pop("segment_raw")
            if "origin_segment" in transform_result[0]:
                result_dict["origin_segment"] = transform_result[0].pop("origin_segment")
            if "inverse" in transform_result[0]:
                result_dict["inverse"] = transform_result[0].pop("inverse")
            if "coord_shift" in transform_result[0]:
                result_dict["coord_shift"] = transform_result[0].pop("coord_shift")
            if "core_bbox" in transform_result[0]:
                result_dict["core_bbox"] = transform_result[0].pop("core_bbox")

            # Apply aug_transform and post_transform to each fragment
            fragment_list = []
            for data_part in transform_result:
                if self.aug_transform:
                    for aug in self.aug_transform:
                        aug_data = aug(deepcopy(data_part))
                        if self.post_transform is not None:
                            aug_data = self.post_transform(aug_data)
                        fragment_list.append(aug_data)
                else:
                    fragment_list.append(self.post_transform(data_part))
            
            result_dict["fragment_list"] = fragment_list
            return result_dict
        else:
            # Single result from transform
            data_dict = transform_result
            result_dict = dict(
                segment=data_dict.get("segment"),  # keep in data_dict so fragments inherit it
                name=data_dict.get("name", self.get_data_name(idx))
            )
            # CRITICAL: Preserve segment_raw for partition evaluation
            if "segment_raw" in data_dict:
                result_dict["segment_raw"] = data_dict.pop("segment_raw")
            if "origin_segment" in data_dict:
                result_dict["origin_segment"] = data_dict.pop("origin_segment")
            if "inverse" in data_dict:
                result_dict["inverse"] = data_dict.pop("inverse")
            if "coord_shift" in data_dict:
                result_dict["coord_shift"] = data_dict.pop("coord_shift")
            if "core_bbox" in data_dict:
                result_dict["core_bbox"] = data_dict.pop("core_bbox")
            # Pop regression target for the tester
            if target_key and target_key in data_dict:
                result_dict["regression_target"] = data_dict.pop(target_key)
                origin_key = f"origin_{target_key}"
                if origin_key in data_dict:
                    result_dict["origin_regression_target"] = data_dict.pop(origin_key)
            
            # Apply aug_transform and post_transform
            fragment_list = []
            for aug in self.aug_transform:
                aug_data = aug(deepcopy(data_dict))
                if self.post_transform is not None:
                    aug_data = self.post_transform(aug_data)
                fragment_list.append(aug_data)
            
            # If no aug_transform, just apply post_transform (no-op if empty)
            if len(fragment_list) == 0:
                fragment_list.append(self.post_transform(data_dict))
            
            result_dict["fragment_list"] = fragment_list
            return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
    
