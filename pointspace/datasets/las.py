import os
import json
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
        ``test_cfg`` and ``cache`` from ``DefaultDataset`` are intentionally
        absent here.  ``LasDataset`` manages its own ``transform``,
        ``post_transform`` and ``aug_transform`` pipelines directly, so
        ``test_cfg`` is not needed.  Shared-memory caching (``cache``) is also
        not used; LAS files are read directly via laspy every time.
    """
    
    VALID_ASSETS = [
        "coord",
        "color",
        "intensity",
        "echo",  # Combined first/last return info (2D: is_first, is_last)
        "hag",  # Height above ground
        "z_base", # Z_base height
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
        self.weighted_sampler = weighted_sampler

        # Class mapping will be initialized in get_data_list
        self.class2id = None  # Original class -> remapped ID
        self.id2class = None  # Remapped ID -> original class
        self.class_weight = None  # Computed class weights
        self.sample_weights = None  # Weights for WeightedRandomSampler

        # Store optional override params for potential use in get_data_list
        self._data_path = data_path
        self._data_list_input = data_list
        
        # Always pass test_mode=False so the base class does NOT try to build
        # transforms from test_cfg (which we don't use).  We restore the real
        # test_mode on self right after.  cache=False is also always enforced
        # because LasDataset reads LAS files directly via laspy.
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=False,
            test_cfg=None,
            cache=False,
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
            self._compute_class_weights()

        # Compute sample weights for WeightedRandomSampler (train only)
        if self.weighted_sampler and not test_mode:
            if self.weighted_sampler == "terrain":
                self._compute_terrain_sample_weights()
            else:
                self._compute_sample_weights()
    
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
            if cls in self.id2class:  # Valid class
                idx = cls if not self.remap_class else list(self.id2class.keys()).index(cls)
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
            idx = cls if not self.remap_class else list(self.id2class.keys()).index(cls)
            weight = weights[idx] if idx < len(weights) else 0
            orig_cls = self.id2class.get(cls, cls)
            logger.info(f"    Class {orig_cls:3d} -> {cls:3d}: {count:10,} points ({count/total_points*100:5.2f}%), weight={weight:.4f}")
        
        logger.info(f"  Computed weights ({weight_method}): {weights}")

    def _compute_sample_weights(self):
        """
        Compute sample-level weights for WeightedRandomSampler.
        
        Sample weight = sum of class weights for all unique classes in the sample.
        This ensures samples with more rare classes are sampled more frequently.
        
        Optimization strategy:
        - Use laspy's lazy reading to quickly extract classification without loading full point cloud
        - Process files in parallel using ThreadPoolExecutor (IO-bound task)
        - Cache class sets for each file to avoid re-reading
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
        
        # Use ThreadPoolExecutor for parallel IO
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
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
                if 0 <= cls < len(self.class_weight):
                    weight += self.class_weight[cls]
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

    def _compute_terrain_sample_weights(self, cell_size=2.0):
        """Compute per-tile sample weights based on terrain roughness.

        For each LAS tile the method reads only XYZ, bins into a
        ``cell_size``-metre grid, computes per-cell Z standard deviation,
        and takes the **90th percentile** of those stds as the tile's
        roughness score.  The 90th percentile captures how *extreme* the
        roughness is without being dominated by a single noisy cell.

        Higher roughness → higher weight → more frequent sampling.

        Final weights are normalised so that ``sum = N`` (same convention as
        :meth:`_compute_sample_weights`).

        Args:
            cell_size (float): Grid cell size in metres for roughness
                estimation.  Default 2.0 m.
        """
        logger = get_root_logger()
        logger.info("Computing terrain sample weights (roughness-based)...")

        n_samples = len(self.data_list)
        if n_samples == 0:
            logger.warning("No files in data_list — skipping terrain weight computation")
            self.sample_weights = None
            return

        roughness = np.zeros(n_samples, dtype=np.float64)
        log_interval = max(1, n_samples // 10)

        def _tile_roughness(idx):
            """Return (idx, roughness_score) for one tile."""
            data_path = self.data_list[idx]
            try:
                las = laspy.read(data_path)
                x = np.asarray(las.x, dtype=np.float64)
                y = np.asarray(las.y, dtype=np.float64)
                z = np.asarray(las.z, dtype=np.float64)

                if len(z) < 4:
                    return idx, 0.0

                # Grid bin
                gx = ((x - x.min()) / cell_size).astype(np.int32)
                gy = ((y - y.min()) / cell_size).astype(np.int32)
                ncols = int(gx.max()) + 1
                cell_id = gy * ncols + gx

                # Per-cell Z std using bincount tricks
                n_cells = int(cell_id.max()) + 1
                cnt = np.bincount(cell_id, minlength=n_cells).astype(np.float64)
                z_sum = np.bincount(cell_id, weights=z, minlength=n_cells)
                z_sq = np.bincount(cell_id, weights=z * z, minlength=n_cells)

                valid = cnt >= 3  # need ≥3 pts for meaningful std
                if valid.sum() == 0:
                    return idx, 0.0

                mean = z_sum[valid] / cnt[valid]
                var = z_sq[valid] / cnt[valid] - mean ** 2
                var = np.clip(var, 0, None)  # numerical safety
                std = np.sqrt(var)

                # 90th percentile of cell stds
                score = float(np.percentile(std, 90))
                return idx, score
            except Exception as e:
                logger.warning(f"Failed to read {data_path}: {e}")
                return idx, 0.0

        # Parallel IO
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os as _os
        num_workers = min(8, _os.cpu_count() or 4)
        logger.info(f"  Processing {n_samples} tiles with {num_workers} workers...")

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futs = {pool.submit(_tile_roughness, i): i for i in range(n_samples)}
            done = 0
            for fut in as_completed(futs):
                idx, score = fut.result()
                roughness[idx] = score
                done += 1
                if done % log_interval == 0:
                    logger.info(f"  Progress: {done}/{n_samples} ({done / n_samples * 100:.1f}%)")

        # Convert roughness → weight: w = 1 + alpha * roughness
        # alpha chosen so that tile with median roughness gets weight ~1,
        # and the roughest tile gets weight ~5.
        median_r = float(np.median(roughness[roughness > 0])) if (roughness > 0).any() else 1.0
        alpha = 4.0 / (float(np.max(roughness)) - median_r + 1e-6) if float(np.max(roughness)) > median_r else 2.0
        sample_weights = 1.0 + alpha * roughness

        # Normalise so sum = n_samples
        total = sample_weights.sum()
        if total > 0:
            sample_weights = sample_weights / total * n_samples

        self.sample_weights = sample_weights

        logger.info(f"  Roughness statistics (cell_size={cell_size}m):")
        logger.info(f"    Min:  {roughness.min():.4f} m")
        logger.info(f"    Max:  {roughness.max():.4f} m")
        logger.info(f"    Mean: {roughness.mean():.4f} m")
        logger.info(f"    P50:  {np.median(roughness):.4f} m")
        logger.info(f"    P90:  {np.percentile(roughness, 90):.4f} m")
        logger.info(f"  Sample weight statistics:")
        logger.info(f"    Min:  {sample_weights.min():.4f}")
        logger.info(f"    Max:  {sample_weights.max():.4f}")
        logger.info(f"    Mean: {sample_weights.mean():.4f}")
        logger.info(f"    Std:  {sample_weights.std():.4f}")

    def _map_classes(self, segment):
        """
        Map original class IDs to remapped IDs
        
        Args:
            segment: Original class labels
            
        Returns:
            Mapped class labels (unmapped classes -> ignore_index)
        """
        mapped = np.full_like(segment, self.ignore_index, dtype=np.int32)
        
        for orig_class, new_class in self.class2id.items():
            mask = segment == orig_class
            mapped[mask] = new_class
        
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

        if self.required_class is not None or self.remap_class:
            self._init_class_mapping()

        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)

        data_dict = {}

        # Read LAS/LAZ file (supports LAS 1.1, 1.2, 1.3, 1.4)
        try:
            las = laspy.read(data_path)
            
            # Always extract coordinates
            data_dict["coord"] = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float32)
            
            # Extract RGB color if available
            if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
                # LAS stores RGB as 16-bit, normalize to [0, 255]
                red = np.array(las.red) / 256.0
                green = np.array(las.green) / 256.0
                blue = np.array(las.blue) / 256.0
                data_dict["color"] = np.vstack((red, green, blue)).transpose().astype(np.float32)
            
            # Extract classification as segment if available
            if hasattr(las, "classification"):
                segment = np.array(las.classification, dtype=np.int32)
                # Apply class mapping
                # CRITICAL: _map_classes will:
                #   1. Remap valid classes (e.g., 1->0, 2->1, ..., 8->7)
                #   2. Set all removed/invalid classes to ignore_index (e.g., 0->8)
                # This ensures ignore_index is applied AFTER remapping, avoiding conflicts
                if self.class2id is not None:
                    segment = self._map_classes(segment)
                data_dict["segment"] = segment
            
            # Extract intensity if available
            if hasattr(las, "intensity"):
                intensity = np.array(las.intensity, dtype=np.float32).reshape(-1, 1)
                data_dict["intensity"] = intensity
            
            # Extract echo information (first/last return) as 2D feature
            if hasattr(las, "return_number") and hasattr(las, "number_of_returns"):
                return_number = np.array(las.return_number, dtype=np.int32)
                number_of_returns = np.array(las.number_of_returns, dtype=np.int32)
                
                # is_first: 1 if first return, -1 otherwise
                is_first = np.where(return_number == 1, 1, -1).astype(np.float32)
                
                # is_last: 1 if last return, -1 otherwise
                is_last = np.where(return_number == number_of_returns, 1, -1).astype(np.float32)
                
                # Combine into 2D echo feature
                data_dict["echo"] = np.vstack((is_first, is_last)).transpose().copy()  # ensure contiguous
            else:
                # If echo info not available, create default echo feature
                # All points treated as both first and last return (single return)
                n_points = data_dict["coord"].shape[0]
                data_dict["echo"] = np.ones((n_points, 2), dtype=np.float32)

            # Extract height above ground (HAG) if available
            # HAG is typically stored as an extra dimension, not a standard LAS field
            if "hag" in las.point_format.dimension_names:
                try:
                    hag = np.array(las.hag, dtype=np.float32).reshape(-1, 1)
                    data_dict["hag"] = hag
                except (AttributeError, KeyError):
                    # Fallback: try dictionary-style access
                    try:
                        hag = np.array(las["hag"], dtype=np.float32).reshape(-1, 1)
                        data_dict["hag"] = hag
                    except (AttributeError, KeyError):
                        pass  # HAG not available

            # Extract Z_base height if available (another common extra dimension)
            if "z_base" in las.point_format.dimension_names:
                try:
                    zbase = np.array(las.z_base, dtype=np.float32).reshape(-1, 1)
                    data_dict["z_base"] = zbase
                    data_dict["z_delta"] = data_dict["coord"][:, 2:3] - zbase
                except (AttributeError, KeyError):
                    # Fallback: try dictionary-style access
                    try:
                        zbase = np.array(las["z_base"], dtype=np.float32).reshape(-1, 1)
                        data_dict["z_base"] = zbase
                        data_dict["z_delta"] = data_dict["coord"][:, 2:3] - zbase
                    except (AttributeError, KeyError):
                        pass  # Z_base not available

            # Extract normal vectors if available (stored as extra dims by tile_las.py)
            dim_names = las.point_format.dimension_names
            if "normal_x" in dim_names and "normal_y" in dim_names and "normal_z" in dim_names:
                try:
                    nx = np.array(las.normal_x, dtype=np.float32)
                    ny = np.array(las.normal_y, dtype=np.float32)
                    nz = np.array(las.normal_z, dtype=np.float32)
                    data_dict["normal"] = np.stack((nx, ny, nz), axis=1)  # (N, 3)
                except (AttributeError, KeyError):
                    pass  # Normal vectors not available

            # Extract superpoint segment ID if available (stored as extra dim by tile_las.py)
            if "superpoint" in las.point_format.dimension_names:
                try:
                    superpoint = np.array(las.superpoint, dtype=np.int64)
                    # Only keep valid superpoints (>= 0), invalid ones (-1) will be handled in loss
                    data_dict["superpoint"] = superpoint
                except (AttributeError, KeyError):
                    pass  # Superpoint not available

            # Extract core_bbox from PointSpace VLR (written by tile_las.py buffered tiling)
            for vlr in las.header.vlrs:
                if (getattr(vlr, 'user_id', None) == "PointSpace"
                        and getattr(vlr, 'record_id', None) == 1001):
                    try:
                        bbox_data = json.loads(vlr.record_data.decode('utf-8'))
                        data_dict["core_bbox"] = np.array(
                            bbox_data["core_bbox"], dtype=np.float32
                        )  # [xmin, ymin, xmax, ymax]
                    except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                        pass
                    break

        except Exception as e:
            logger = get_root_logger()
            logger.error(f"Error reading {data_path}: {e}")
            # Create empty data with minimal required fields
            data_dict["coord"] = np.zeros((0, 3), dtype=np.float32)
        
        # Add metadata
        data_dict["name"] = name
        
        # If segment is not available, create default one filled with ignore_index
        if "segment" not in data_dict:
            data_dict["segment"] = np.ones(data_dict["coord"].shape[0], dtype=np.int32) * self.ignore_index
        
        # Ensure segment is 1D
        data_dict["segment"] = data_dict["segment"].reshape([-1])
        
        # Add instance data (not typically available in LAS/LAZ files)
        data_dict["instance"] = np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
        
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
                for aug in self.aug_transform:
                    aug_data = aug(deepcopy(data_part))
                    if self.post_transform is not None:
                        aug_data = self.post_transform(aug_data)
                    fragment_list.append(aug_data)
            
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
    
