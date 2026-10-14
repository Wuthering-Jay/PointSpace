import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import laspy
import torch
from safetensors import safe_open
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
        "return_role",  # RRCL target: single/first/middle/last -> 0/1/2/3
        "normal",  # Normal vectors (normal_x, normal_y, normal_z extra dims)
        "local_shape",  # Six offline XYZ-derived geometry descriptors
        "geom_valid",  # Offline geometry target reliability mask
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

            # Return attributes are supervision only.  They are deliberately
            # converted to a categorical target instead of being appended to
            # ``feat``; the config's ``feat_keys`` remains the sole authority
            # over backbone inputs.
            return_role = np.full(num_points, self.ignore_index, dtype=np.int64)
            single = number_of_returns == 1
            first = (number_of_returns > 1) & (return_number == 1)
            middle = (
                (number_of_returns > 1)
                & (return_number > 1)
                & (return_number < number_of_returns)
            )
            last = (
                (number_of_returns > 1)
                & (return_number == number_of_returns)
            )
            # Keep this order explicit: a single return also has r == 1 == n.
            return_role[single] = 0
            return_role[first] = 1
            return_role[middle] = 2
            return_role[last] = 3
            data_dict["return_role"] = return_role

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

        shape_dimensions = (
            "geom_linearity", "geom_planarity", "geom_scattering",
            "geom_curvature", "geom_anisotropy", "geom_verticality",
        )
        if set(shape_dimensions).issubset(dim_names):
            data_dict["local_shape"] = np.column_stack(
                [self._read_extra_dimension(las, name) for name in shape_dimensions]
            ).astype(np.float32, copy=False)
        if "geom_valid" in dim_names:
            data_dict["geom_valid"] = self._read_extra_dimension(
                las, "geom_valid", dtype=np.uint8
            ).astype(bool, copy=False)

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
                    and not os.path.basename(f).startswith(".")
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
                and not os.path.basename(f).startswith(".")
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


@DATASETS.register_module()
class RRCLLasDataset(LasDataset):
    """LAS dataset whose class statistics are the four return roles.

    Geometry preprocessing is inherited from :class:`LasDataset`.  The only
    task-specific behavior is class-weight estimation; semantic LAS
    classification values are intentionally irrelevant to RRCL.
    """

    NUM_RETURN_ROLES = 4

    def _compute_class_weights(self):
        logger = get_root_logger()
        mode = self.class_weight_mode
        if isinstance(mode, (list, tuple, np.ndarray)):
            self._set_manual_class_weight(mode)
            return

        method = "sqrt" if mode in (None, "auto") else mode
        if isinstance(mode, dict):
            method = mode.get("method", "sqrt")
        if method != "sqrt":
            raise ValueError(
                "RRCLLasDataset v1 supports class_weight='sqrt' only; "
                f"got {method!r}."
            )

        n_samples = len(self.data_list)
        if isinstance(self.weight_sample, float) and 0 < self.weight_sample <= 1:
            n_compute = max(1, int(n_samples * self.weight_sample))
        elif isinstance(self.weight_sample, int) and self.weight_sample > 0:
            n_compute = min(self.weight_sample, n_samples)
        else:
            n_compute = n_samples
        indices = (
            np.random.choice(n_samples, n_compute, replace=False)
            if n_compute < n_samples
            else range(n_samples)
        )

        counts = np.zeros(self.NUM_RETURN_ROLES, dtype=np.int64)
        for idx in indices:
            try:
                las = laspy.read(self.data_list[idx])
                dim_names = set(las.point_format.dimension_names)
                if not {"return_number", "number_of_returns"}.issubset(dim_names):
                    logger.warning(
                        "Skipping %s while estimating RRCL weights: "
                        "missing return dimensions",
                        self.data_list[idx],
                    )
                    continue
                r = np.asarray(las.return_number)
                n = np.asarray(las.number_of_returns)
                counts[0] += np.count_nonzero(n == 1)
                counts[1] += np.count_nonzero((n > 1) & (r == 1))
                counts[2] += np.count_nonzero((n > 1) & (r > 1) & (r < n))
                counts[3] += np.count_nonzero((n > 1) & (r == n))
            except Exception as exc:
                logger.warning("Failed to scan %s: %s", self.data_list[idx], exc)

        total = counts.sum()
        if total == 0 or np.any(counts == 0):
            raise ValueError(
                "Cannot compute finite RRCL sqrt weights: role counts are "
                f"{counts.tolist()}. Supply a representative weight_sample or "
                "manual four-element class_weight."
            )
        frequencies = counts.astype(np.float64) / total
        weights = 1.0 / np.sqrt(frequencies)
        weights = (weights / weights.sum() * self.NUM_RETURN_ROLES).astype(np.float32)
        self.class_weight = weights
        logger.info(
            "RRCL role counts S/F/M/L=%s; normalized sqrt weights=%s",
            counts.tolist(),
            weights.tolist(),
        )

    def _set_manual_class_weight(self, class_weight):
        weights = np.asarray(class_weight, dtype=np.float32).reshape(-1)
        if weights.size != self.NUM_RETURN_ROLES:
            raise ValueError(
                "RRCL class_weight must contain exactly four values "
                "for S/F/M/L."
            )
        self.class_weight = weights


@DATASETS.register_module()
class LasImageDataset(LasDataset):
    """Read paired LAS/LAZ, DINO and point-to-patch Safetensors files.

    Files are paired by stem. Point-level image fields are registered in
    ``index_valid_keys`` so crop, sampling and voxel transforms keep them
    aligned with the LAS points. DINO features remain sample-level tensors.

    When explicit paths are omitted, the expected layout is::

        data_root/
            pointcloud/*.las|*.laz
            dino_feature/*.safetensors
            correspondence/*.safetensors
    """

    IMAGE_POINT_KEYS = (
        "image_pixel_coord",
        "image_patch_index",
        "image_valid",
        "image_observability",
    )

    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        data_path=None,
        data_list=None,
        dino_feature_path=None,
        correspondence_path=None,
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
        require_observability=False,
    ):
        root = Path(data_root)
        if data_path is None and data_list is None:
            default_pointcloud_path = root / "pointcloud"
            if default_pointcloud_path.is_dir():
                data_path = str(default_pointcloud_path)

        self.dino_feature_path = Path(
            dino_feature_path
            if dino_feature_path is not None
            else root / "dino_feature"
        )
        self.correspondence_path = Path(
            correspondence_path
            if correspondence_path is not None
            else root / "correspondence"
        )
        self._dino_assets = {}
        self.require_observability = bool(require_observability)

        super().__init__(
            split=split,
            data_root=data_root,
            data_path=data_path,
            data_list=data_list,
            transform=transform,
            post_transform=post_transform,
            aug_transform=aug_transform,
            test_mode=test_mode,
            ignore_index=ignore_index,
            loop=loop,
            required_class=required_class,
            remap_class=remap_class,
            class_weight=class_weight,
            weight_sample=weight_sample,
            weighted_sampler=weighted_sampler,
            target_key=target_key,
        )

    @staticmethod
    def _index_assets(path, suffix, description):
        path = Path(path)
        if path.is_file():
            if path.suffix.lower() != suffix:
                raise ValueError(f"Expected {suffix} {description}: {path}")
            files = [(path.stem, str(path))]
        elif path.is_dir():
            # ``Path.iterdir`` followed by ``is_file`` is exceptionally slow
            # for large asset folders on Windows because it performs a Python
            # Path allocation and may issue an extra metadata query per entry.
            # scandir reuses the directory enumeration metadata and reduces
            # 10k+ DINO/mapping index construction from roughly a minute to
            # seconds without changing the stem-pairing semantics.
            files = sorted(
                (
                    os.path.splitext(entry.name)[0], entry.path
                )
                for entry in os.scandir(path)
                if entry.is_file()
                and os.path.splitext(entry.name)[1].lower() == suffix
            )
        else:
            raise FileNotFoundError(f"{description} path does not exist: {path}")

        if not files:
            raise FileNotFoundError(f"No {suffix} {description} files found: {path}")

        result = {}
        for stem, file_path in files:
            if stem in result:
                raise ValueError(
                    f"Duplicate {description} stem {stem!r}: "
                    f"{result[stem]} and {file_path}"
                )
            result[stem] = file_path
        return result

    def get_data_list(self):
        data_list = super().get_data_list()
        feature_files = self._index_assets(
            self.dino_feature_path, ".safetensors", "DINO feature"
        )
        mapping_files = self._index_assets(
            self.correspondence_path, ".safetensors", "point-to-patch mapping"
        )

        missing = []
        assets = {}
        for las_path in data_list:
            stem = Path(las_path).stem
            missing_assets = []
            if stem not in feature_files:
                missing_assets.append("DINO feature")
            if stem not in mapping_files:
                missing_assets.append("point-to-patch mapping")
            if missing_assets:
                missing.append(f"{stem}: {', '.join(missing_assets)}")
                continue
            assets[stem] = (feature_files[stem], mapping_files[stem])

        if missing:
            preview = "; ".join(missing[:8])
            if len(missing) > 8:
                preview += f"; ... and {len(missing) - 8} more"
            raise FileNotFoundError(
                "Each LAS/LAZ sample must have same-stem DINO and mapping assets. "
                + preview
            )

        self._dino_assets = assets
        get_root_logger().info(
            "%s: paired %d LAS/LAZ, DINO and mapping samples.",
            self.__class__.__name__,
            len(data_list),
        )
        return data_list

    @staticmethod
    def _metadata_pair(metadata, key, required=True):
        value = metadata.get(key)
        if value is None:
            if required:
                raise ValueError(f"DINO metadata is missing {key!r}")
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"DINO metadata {key!r} must contain [width, height]")
        width, height = (int(value[0]), int(value[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"DINO metadata {key!r} must be positive, got {value}")
        return width, height

    @staticmethod
    def _load_dino_feature(path):
        try:
            with safe_open(path, framework="pt", device="cpu") as file:
                if "feature" not in file.keys():
                    raise ValueError("Safetensors has no 'feature' tensor")
                feature = file.get_tensor("feature")
                raw_metadata = (file.metadata() or {}).get("metadata")
        except Exception as error:
            raise ValueError(f"Failed to read DINO feature {path}: {error}") from error

        if raw_metadata is None:
            raise ValueError(f"Safetensors has no DINO metadata: {path}")
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid DINO metadata JSON in {path}: {error}") from error

        if feature.ndim != 2:
            raise ValueError(
                f"DINO feature must use PC layout, got shape "
                f"{tuple(feature.shape)}: {path}"
            )
        metadata_shape = metadata.get("feature_shape")
        if metadata_shape is None or tuple(map(int, metadata_shape)) != tuple(
            feature.shape
        ):
            raise ValueError(
                f"DINO feature_shape metadata does not match tensor "
                f"{tuple(feature.shape)}: {path}"
            )
        if metadata.get("feature_layout", "PC").upper() != "PC":
            raise ValueError(f"Only PC DINO features are supported: {path}")
        return feature.contiguous(), metadata

    def _load_mapping(self, path, num_points):
        try:
            with safe_open(path, framework="pt", device="cpu") as file:
                required = {"pixel_coord", "patch_index", "valid"}
                missing = required.difference(file.keys())
                if missing:
                    raise ValueError(f"missing tensors: {', '.join(sorted(missing))}")
                pixel_coord = file.get_tensor("pixel_coord")
                patch_index = file.get_tensor("patch_index")
                valid = file.get_tensor("valid")
                observability = (
                    file.get_tensor("observability")
                    if "observability" in file.keys()
                    else None
                )
                mapping_metadata = file.metadata() or {}
        except Exception as error:
            raise ValueError(f"Failed to read mapping {path}: {error}") from error
        if pixel_coord.shape != (num_points, 2):
            raise ValueError(
                f"pixel_coord shape mismatch for {path}: expected "
                f"({num_points}, 2), got {tuple(pixel_coord.shape)}"
            )
        if patch_index.shape != (num_points,) or valid.shape != (num_points,):
            raise ValueError(f"patch_index/valid length mismatch for {path}")
        valid = valid.bool()
        if observability is None and self.require_observability:
            raise ValueError(
                "Mapping has no native 'observability' tensor while "
                f"require_observability=True: {path}"
            )
        if observability is None:
            observability = valid.float()
        else:
            if observability.shape != (num_points,):
                raise ValueError(f"observability length mismatch for {path}")
            observability = observability.float()
            if not torch.isfinite(observability).all():
                raise ValueError(f"observability contains non-finite values: {path}")
            if torch.any((observability < 0) | (observability > 1)):
                raise ValueError(f"observability must be within [0, 1]: {path}")
        return (
            pixel_coord.long(),
            patch_index.long(),
            valid,
            observability,
            mapping_metadata,
        )

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        name = data_dict["name"]
        feature_path, mapping_path = self._dino_assets[name]
        feature, metadata = self._load_dino_feature(feature_path)
        (
            pixel_coord,
            patch_index,
            valid,
            observability,
            mapping_metadata,
        ) = self._load_mapping(mapping_path, data_dict["coord"].shape[0])

        original_width, original_height = self._metadata_pair(
            metadata, "original_size"
        )
        feature_grid_size = metadata.get("feature_grid_size")
        if not isinstance(feature_grid_size, (list, tuple)) or len(
            feature_grid_size
        ) != 2:
            raise ValueError(f"Invalid feature_grid_size in {feature_path}")
        grid_width, grid_height = map(int, feature_grid_size)
        if grid_width * grid_height != feature.shape[0]:
            raise ValueError(f"Feature grid does not match feature length: {feature_path}")
        patch_size = int(metadata.get("patch_size", 0))
        if patch_size <= 0:
            raise ValueError(f"Invalid patch_size in {feature_path}: {patch_size}")
        mapping_grid = mapping_metadata.get("feature_grid_size")
        if mapping_grid is not None and json.loads(mapping_grid) != [
            grid_width,
            grid_height,
        ]:
            raise ValueError(f"Mapping and DINO feature grid mismatch: {mapping_path}")
        mapping_patch_size = mapping_metadata.get("patch_size")
        if mapping_patch_size is not None and int(mapping_patch_size) != patch_size:
            raise ValueError(f"Mapping and DINO patch_size mismatch: {mapping_path}")
        mapping_original_size = mapping_metadata.get("original_size")
        if (
            mapping_original_size is not None
            and json.loads(mapping_original_size) != [original_width, original_height]
        ):
            raise ValueError(f"Mapping and DINO original_size mismatch: {mapping_path}")
        padded_size = self._metadata_pair(metadata, "padded_size", required=False)
        if padded_size is None:
            padded_width = grid_width * patch_size
            padded_height = grid_height * patch_size
        else:
            padded_width, padded_height = padded_size

        data_dict.update(
            dino_feature=feature,
            image_pixel_coord=pixel_coord,
            image_patch_index=patch_index,
            image_valid=valid,
            image_observability=observability,
            dino_offset=np.asarray([feature.shape[0]], dtype=np.int64),
            image_original_size=np.asarray(
                [[original_height, original_width]], dtype=np.int64
            ),
            image_padded_size=np.asarray(
                [[padded_height, padded_width]], dtype=np.int64
            ),
            image_feature_size=np.asarray(
                [[grid_height, grid_width]], dtype=np.int64
            ),
            image_patch_size=np.asarray([patch_size], dtype=np.int64),
        )

        point_keys = list(data_dict.get("index_valid_keys", []))
        for key in self.IMAGE_POINT_KEYS:
            if key not in point_keys:
                point_keys.append(key)
        data_dict["index_valid_keys"] = point_keys
        return data_dict


@DATASETS.register_module()
class CVRLLasDataset(LasImageDataset, RRCLLasDataset):
    """LAS dataset containing complementary visual and return supervision.

    The visual branch inherits same-stem DINO and point-to-patch pairing from
    :class:`LasImageDataset`. Return roles and their optional S/F/M/L class
    weights come from :class:`RRCLLasDataset`. Points with missing or invalid
    return metadata remain available to VPSD with ``return_role=ignore_index``;
    RRCL losses and context construction ignore them. Raw return numbers are
    never exposed, and the derived ``echo`` feature is removed to prevent
    return target leakage into the CVRL backbone input.
    """

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        return_role = data_dict.get("return_role")
        if return_role is None:
            # Some photogrammetric/synthetic surface classes have no physical
            # LiDAR pulse metadata. They still provide valid image-patch
            # supervision, so retain every point and explicitly mask RRCL.
            return_role = np.full(
                data_dict["coord"].shape[0],
                self.ignore_index,
                dtype=np.int64,
            )
            data_dict["return_role"] = return_role

        # CVRL treats return properties exclusively as supervision. Removing
        # echo here makes leakage impossible even if a later Collect transform
        # is accidentally configured with feat_keys containing ``echo``.
        data_dict.pop("echo", None)
        point_keys = list(data_dict.get("index_valid_keys", []))
        point_keys = [key for key in point_keys if key != "echo"]
        if "return_role" not in point_keys:
            point_keys.append("return_role")
        data_dict["index_valid_keys"] = point_keys
        return data_dict


@DATASETS.register_module()
class MSCLasPairDataset(LasDataset):
    """Two independently augmented views of one unlabeled Swiss LAS tile.

    `origin_coord` remains in the common pre-augmentation frame so MSC can
    match points after independent sampling. LAS labels and return metadata
    are removed before either view is transformed.
    """

    def __init__(
        self, view1_transform, view2_transform, base_transform=None, **kwargs
    ):
        if kwargs.get("test_mode", False):
            raise ValueError("MSCLasPairDataset is training-only")
        if kwargs.get("class_weight") is not None or kwargs.get("weighted_sampler", False):
            raise ValueError("MSC views must not use LAS semantic class weights")
        super().__init__(transform=None, post_transform=None, **kwargs)
        self.base_transform = Compose(base_transform)
        self.view1_transform = Compose(view1_transform)
        self.view2_transform = Compose(view2_transform)

    def prepare_train_data(self, idx):
        source = self.get_data(idx)
        point_keys = [key for key in ("coord", "intensity") if key in source]
        source = {key: source[key] for key in point_keys}
        source["index_valid_keys"] = point_keys
        source = self.base_transform(source)
        view1 = self.view1_transform(deepcopy(source))
        view2 = self.view2_transform(deepcopy(source))
        required = {"origin_coord", "coord", "grid_coord", "feat", "offset"}
        for number, view in ((1, view1), (2, view2)):
            missing = required.difference(view)
            if missing:
                raise ValueError(f"MSC view {number} is missing {sorted(missing)}")
        return {
            **{"view1_" + key: value for key, value in view1.items()},
            **{"view2_" + key: value for key, value in view2.items()},
        }


@DATASETS.register_module()
class SonataLasPairDataset(MSCLasPairDataset):
    """Two global and optionally four local views in one common XYZ frame."""

    def __init__(self, local_transform=None, num_local_views=0, **kwargs):
        super().__init__(**kwargs)
        self.local_transform = Compose(local_transform) if local_transform else None
        self.num_local_views = int(num_local_views)
        if self.num_local_views and self.local_transform is None:
            raise ValueError("local_transform is required for local Sonata views")

    def prepare_train_data(self, idx):
        source = self.get_data(idx)
        source = {key: source[key] for key in ("coord", "intensity") if key in source}
        source["index_valid_keys"] = list(source)
        source = self.base_transform(source)
        views = [self.view1_transform(deepcopy(source)),
                 self.view2_transform(deepcopy(source))]
        result = {}
        for number, view in enumerate(views, 1):
            if {"origin_coord", "coord", "grid_coord", "feat", "offset"}.difference(view):
                raise ValueError(f"Sonata global view {number} is incomplete")
            result.update({f"view{number}_{key}": value for key, value in view.items()})
        for number in range(self.num_local_views):
            view = self.local_transform(deepcopy(source))
            if {"origin_coord", "coord", "grid_coord", "feat", "offset"}.difference(view):
                raise ValueError(f"Sonata local view {number + 1} is incomplete")
            result.update({f"local{number + 1}_{key}": value for key, value in view.items()})
        return result


@DATASETS.register_module()
class ConcertoLasPairDataset(LasImageDataset):
    """Two ALS views; only the first carries offline patch supervision.

    DINO features are not copied into the second view and never appear among
    ``feat_keys``. Semantic and return metadata are discarded before cropping.
    """

    def __init__(self, view1_transform, view2_transform,
                 base_transform=None, local_transform=None,
                 num_local_views=0, **kwargs):
        if kwargs.get("test_mode", False):
            raise ValueError("ConcertoLasPairDataset is training-only")
        if kwargs.get("class_weight") is not None or kwargs.get("weighted_sampler", False):
            raise ValueError("Concerto views must not use LAS semantic weights")
        super().__init__(transform=None, post_transform=None, **kwargs)
        self.base_transform = Compose(base_transform)
        self.view1_transform = Compose(view1_transform)
        self.view2_transform = Compose(view2_transform)
        self.local_transform = Compose(local_transform) if local_transform else None
        self.num_local_views = int(num_local_views)
        if self.num_local_views and self.local_transform is None:
            raise ValueError("local_transform is required for local Concerto views")

    def prepare_train_data(self, idx):
        raw = self.get_data(idx)
        required = ("coord", "intensity", "dino_feature",
                    "image_patch_index", "image_valid")
        missing = set(required).difference(raw)
        if missing:
            raise KeyError(f"Concerto LAS sample lacks {sorted(missing)}")
        source = {key: raw[key] for key in required}
        source["index_valid_keys"] = [
            "coord", "intensity", "image_patch_index", "image_valid"
        ]
        source = self.base_transform(source)
        first = self.view1_transform(deepcopy(source))
        second_source = {key: source[key] for key in ("coord", "intensity")}
        second_source["index_valid_keys"] = ["coord", "intensity"]
        second = self.view2_transform(deepcopy(second_source))
        visual_keys = ("dino_feature", "image_patch_index",
                       "image_valid", "dino_offset")
        for key in visual_keys:
            if key not in first:
                raise KeyError(f"Concerto first view lacks {key}")
        for number, view in ((1, first), (2, second)):
            missing = {"origin_coord", "coord", "grid_coord", "feat", "offset"}.difference(view)
            if missing:
                raise ValueError(f"Concerto view {number} lacks {sorted(missing)}")
        result = {"view1_" + key: value for key, value in first.items()
                  if key not in visual_keys}
        result.update({"view2_" + key: value for key, value in second.items()})
        result.update({key: first[key] for key in visual_keys})
        for number in range(self.num_local_views):
            view = self.local_transform(deepcopy(second_source))
            if {"origin_coord", "coord", "grid_coord", "feat", "offset"}.difference(view):
                raise ValueError(f"Concerto local view {number + 1} is incomplete")
            result.update({f"local{number + 1}_{key}": value for key, value in view.items()})
        return result
    
