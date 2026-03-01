"""
LAS/LAZ Tile Merger

将 LASTileProcessor 生成的分块点云合并回原始文件，支持：
- 自动识别 tile 分组（按文件名 {original}_{NNNN}.las/laz 格式分组）
- 若 tile 中存在 orig_idx 字段，按 orig_idx 重排点序，恢复原始顺序
- 合并后丢弃 orig_idx 字段，保留其余所有属性
- 复用 tile 头文件（point_format, scales, offsets, VLRs），确保 tile→merge 往返一致
- 重叠 tile 合并：对离散属性（如 classification）使用多数投票，
  对连续属性可选平均，利用重叠提高精度

性能特性（全 O(n) 向量化，无 O(n log n) 排序）：
- 利用 orig_idx 为连续整数 0..n-1 的性质，用 np.bincount 代替 np.unique
- 用 np.bincount 编码键代替 np.add.at 完成投票
- 用直接散射写入代替花式索引
"""

import re
import numpy as np
import laspy
from pathlib import Path
from typing import Union, List, Dict, Optional, Set
from collections import defaultdict
from tqdm import tqdm


class LASMerger:
    """
    LAS 点云合并器，与 LASTileProcessor 互为逆操作。
    
    将 tile 后的小块点云合并回原始文件：
    - 复用 tile 头文件（point_format, version, scales, offsets, VLRs）
    - 若存在 orig_idx 字段，利用其作为直接索引（O(n)）恢复原始点序并去重
    - 重叠检测：通过 np.bincount 统计每个 orig_idx 出现次数
    - 投票融合：对 vote_dims 中的离散属性使用多数投票（如 classification）
    - 均值融合：对 average_dims 中的连续属性取均值（如 hag）
    - 其余属性直接散射写入（坐标、intensity 等不变属性）
    
    核心算法复杂度：O(n)，其中 n = 所有 tile 总点数之和
    """
    
    # tile 文件中需要在合并后丢弃的 extra dimensions
    DISCARD_DIMS = {'orig_idx'}
    # 默认使用多数投票的维度
    DEFAULT_VOTE_DIMS = {'classification'}
    
    def __init__(self, 
                 input_path: Union[str, Path],
                 output_dir: Union[str, Path] = None,
                 output_format: str = 'las',
                 vote_dims: Optional[Set[str]] = None,
                 average_dims: Optional[Set[str]] = None):
        """
        Initialize LAS point cloud merger.
        
        Args:
            input_path: Path to directory containing tiled LAS files
            output_dir: Directory to save merged files (default: parent of input)
            output_format: Output format, 'las' or 'laz'
            vote_dims: Dimensions to aggregate via majority voting when overlap exists.
                       Default: {'classification'}
            average_dims: Dimensions to aggregate via averaging when overlap exists.
                          Default: empty set (no averaging). Useful for float extra dims
                          like 'hag' when computed per-tile.
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.output_format = output_format.lower()
        self.vote_dims = set(vote_dims) if vote_dims is not None else set(self.DEFAULT_VOTE_DIMS)
        self.average_dims = set(average_dims) if average_dims else set()
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
    
    def _find_tile_groups(self) -> Dict[str, List[Path]]:
        """
        Find all tile LAS files and group them by original file name.
        
        Matches the naming convention from LASTileProcessor: {original_name}_{NNNN}.las
        The greedy regex ensures that only the last _digits suffix is treated as the tile index,
        so original filenames containing underscores or digits are handled correctly.
        
        Returns:
            Dictionary mapping original file names to sorted lists of tile files
        """
        tile_groups = defaultdict(list)
        
        # Find all LAS/LAZ files
        las_files = list(self.input_path.glob("**/*.las")) + list(self.input_path.glob("**/*.laz"))
        
        # Pattern: greedy match for original name, then _digits at the end
        # e.g. "my_scan_0003" -> original="my_scan", index=3
        tile_pattern = re.compile(r'^(.+)_(\d+)$')
        
        for file_path in las_files:
            match = tile_pattern.match(file_path.stem)
            if match:
                original_name = match.group(1)
                tile_groups[original_name].append(file_path)
        
        # Sort tiles by their index to ensure consistent order
        for original_name in tile_groups:
            tile_groups[original_name].sort(
                key=lambda p: int(tile_pattern.match(p.stem).group(2))
            )
        
        return tile_groups
    
    def merge_all(self):
        """Merge all tiled point clouds back into original LAS files."""
        tile_groups = self._find_tile_groups()
        
        if not tile_groups:
            print("No tile groups found. Ensure files follow naming: {name}_{NNNN}.las")
            return
        
        print(f"Found {len(tile_groups)} group(s) of tiles to merge")
        for original_name, tiles in tqdm(tile_groups.items(), desc="Merging files", unit="file"):
            print(f"\nMerging {len(tiles)} tiles for '{original_name}'")
            self._merge_tiles(original_name, tiles)
    
    def _merge_tiles(self, original_name: str, tile_files: List[Path]):
        """
        Merge tile files back into a single LAS file.
        
        Handles three scenarios automatically:
        1. No orig_idx: simple concatenation (no reordering or dedup)
        2. orig_idx without overlap: scatter by orig_idx to restore original order
        3. orig_idx with overlap: deduplicate + vote/average via O(n) algorithms
        
        Key optimization: orig_idx values are contiguous integers 0..n_unique-1
        (assigned by LASTileProcessor), so they can be used as direct array indices.
        This eliminates np.unique (O(n log n) sort) in favor of np.bincount (O(n)).
        
        Args:
            original_name: Name of the original file (without extension)
            tile_files: Sorted list of tile LAS files to merge
        """
        if not tile_files:
            print(f"  Warning: No tiles found for {original_name}")
            return
        
        # --- 1. Read first tile for metadata ---
        with laspy.open(tile_files[0]) as fh:
            first_tile = fh.read()
        
        src_header = first_tile.header
        
        # Detect orig_idx and categorize extra dims
        has_orig_idx = False
        extra_dims_to_keep = []
        for ed in src_header.point_format.extra_dimensions:
            if ed.name == 'orig_idx':
                has_orig_idx = True
            elif ed.name not in self.DISCARD_DIMS:
                extra_dims_to_keep.append(ed)
        
        # Build output header - reuse tile header for consistency with original
        header = laspy.LasHeader(
            point_format=src_header.point_format.id,
            version=src_header.version,
        )
        header.scales = src_header.scales
        header.offsets = src_header.offsets
        for vlr in src_header.vlrs:
            header.vlrs.append(vlr)
        for ed in extra_dims_to_keep:
            header.add_extra_dim(laspy.ExtraBytesParams(
                name=ed.name,
                type=ed.dtype,
                description=getattr(ed, 'description', ''),
            ))
        
        # Determine output dimension names (what goes into the merged file)
        output_dim_names = set(header.point_format.dimension_names)
        # Tile dimensions to process (exclude discarded, keep only those in output)
        tile_dim_names = [d for d in first_tile.point_format.dimension_names
                          if d not in self.DISCARD_DIMS and d in output_dim_names]
        
        # Get dtype for each dimension from first tile
        dim_dtypes = {}
        for dim_name in tile_dim_names:
            dim_dtypes[dim_name] = np.array(getattr(first_tile, dim_name)).dtype
        
        # --- 2. Count total concatenated points (including duplicates from overlap) ---
        total_concat = len(first_tile)
        for tile_file in tile_files[1:]:
            with laspy.open(tile_file) as fh:
                total_concat += fh.header.point_count
        
        print(f"  Total concatenated points: {total_concat:,}")
        
        # --- 3. Pre-allocate staging arrays and read all tiles ---
        staging = {name: np.empty(total_concat, dtype=dt)
                   for name, dt in dim_dtypes.items()}
        staging_orig_idx = np.empty(total_concat, dtype=np.uint32) if has_orig_idx else None
        
        # Fill from first tile (already loaded)
        n = len(first_tile)
        if has_orig_idx:
            staging_orig_idx[:n] = np.array(first_tile.orig_idx)
        for dim_name in tile_dim_names:
            staging[dim_name][:n] = np.array(getattr(first_tile, dim_name))
        offset = n
        del first_tile
        
        # Read remaining tiles
        for tile_file in tqdm(tile_files[1:], desc="  Reading tiles", unit="tile", leave=False):
            with laspy.open(tile_file) as fh:
                tile = fh.read()
            n = len(tile)
            if has_orig_idx:
                staging_orig_idx[offset:offset + n] = np.array(tile.orig_idx)
            for dim_name in tile_dim_names:
                staging[dim_name][offset:offset + n] = np.array(getattr(tile, dim_name))
            offset += n
            del tile
        
        # --- 4. Detect overlap via O(1) comparison (no full bincount needed) ---
        if has_orig_idx:
            # orig_idx values are contiguous 0..n_unique-1, use as direct indices
            n_output = int(staging_orig_idx.max()) + 1
            has_overlap = total_concat > n_output
            
            if has_overlap:
                avg_overlap = total_concat / n_output
                print(f"  Overlap detected: {total_concat:,} concat -> {n_output:,} unique "
                      f"({avg_overlap:.1f}x avg overlap)")
            else:
                print(f"  No overlap, {n_output:,} unique points")
        else:
            n_output = total_concat
            has_overlap = False
        
        # --- 5. Build output LAS with aggregated dimensions ---
        merged_las = laspy.LasData(header)
        merged_las.points = laspy.ScaleAwarePointRecord.zeros(n_output, header=header)
        
        # Lazy-compute per-point counts (only materialized when average dims exist)
        _counts = None
        
        for dim_name in tile_dim_names:
            data = staging[dim_name]
            
            if not has_orig_idx:
                # No orig_idx: just use concatenated data as-is
                result = data
            elif has_overlap and dim_name in self.vote_dims:
                # Overlap + vote dim: majority voting O(n)
                print(f"  Majority voting on '{dim_name}'...")
                result = self._majority_vote(data, staging_orig_idx, n_output)
            elif has_overlap and dim_name in self.average_dims:
                # Overlap + average dim: averaging O(n)
                # Lazy-compute counts once, reused across all average dims
                if _counts is None:
                    _counts = np.bincount(staging_orig_idx, minlength=n_output)
                print(f"  Averaging '{dim_name}'...")
                result = self._average(data, staging_orig_idx, n_output, counts=_counts)
            else:
                # No overlap OR identity dim: direct scatter O(n)
                # For identical dims (coords, intensity), any occurrence is fine.
                # For non-overlap case, each orig_idx appears exactly once.
                result = np.empty(n_output, dtype=data.dtype)
                result[staging_orig_idx] = data
            
            setattr(merged_las, dim_name, result)
        
        # Free staging memory
        del staging
        if staging_orig_idx is not None:
            del staging_orig_idx
        
        # --- 6. Save ---
        output_path = self.output_dir / f"{original_name}.{self.output_format}"
        merged_las.update_header()
        merged_las.write(output_path)
        print(f"  Saved merged file ({n_output:,} points) to {output_path}")
    
    @staticmethod
    def _majority_vote(data: np.ndarray, orig_idx: np.ndarray, n_unique: int) -> np.ndarray:
        """
        O(n) majority voting for overlapping tiles using np.bincount on encoded keys.
        
        Encodes each (point_id, class_label) pair as a single integer key:
            key = orig_idx * n_classes + class_label
        Then uses np.bincount (single O(n) pass) to build the full vote matrix.
        
        For large n_unique * n_classes, falls back to a class-iteration approach
        that is still vectorized within each class (np.bincount per class).
        
        Args:
            data: Concatenated label array (total_concat,), integer-like
            orig_idx: orig_idx array mapping each concat point to unique point 0..n_unique-1
            n_unique: Number of unique points
            
        Returns:
            Voted labels array (n_unique,), same dtype as input
        """
        orig_dtype = data.dtype
        data_int = data.astype(np.intp)
        n_classes = int(data_int.max()) + 1
        
        # Estimate memory for dense vote matrix: n_unique * n_classes * 8 bytes (int64 from bincount)
        estimated_bytes = n_unique * n_classes * 8
        
        if estimated_bytes < 256 * 1024 * 1024:  # < 256 MB (fits in L3 cache region)
            # ---- Fast path: encode (point_id, class) → single key, then bincount ----
            # O(n) single pass, best when vote matrix fits in CPU cache
            keys = orig_idx.astype(np.int64) * n_classes + data_int
            vote_flat = np.bincount(keys, minlength=n_unique * n_classes)
            result = vote_flat.reshape(n_unique, n_classes).argmax(axis=1)
        else:
            # ---- Memory-efficient fallback: iterate over classes ----
            # Each iteration is vectorized (np.bincount), outer loop is tiny (n_classes)
            best_count = np.zeros(n_unique, dtype=np.int32)
            best_label = np.zeros(n_unique, dtype=np.intp)
            for cls in range(n_classes):
                cls_mask = data_int == cls
                if not np.any(cls_mask):
                    continue
                cls_count = np.bincount(orig_idx[cls_mask], minlength=n_unique).astype(np.int32)
                better = cls_count > best_count
                best_count[better] = cls_count[better]
                best_label[better] = cls
            result = best_label
        
        return result.astype(orig_dtype)
    
    @staticmethod
    def _average(data: np.ndarray, orig_idx: np.ndarray, n_unique: int,
                 counts: np.ndarray = None) -> np.ndarray:
        """
        O(n) averaging for overlapping tiles using np.bincount with weights.
        
        Args:
            data: Concatenated value array (total_concat,)
            orig_idx: orig_idx array mapping each concat point to unique point 0..n_unique-1
            n_unique: Number of unique points
            counts: Pre-computed per-point occurrence counts (optional, avoids redundant bincount)
            
        Returns:
            Averaged values array (n_unique,), same dtype as input
        """
        orig_dtype = data.dtype
        weights = data if data.dtype == np.float64 else data.astype(np.float64)
        sums = np.bincount(orig_idx, weights=weights, minlength=n_unique)
        if counts is None:
            counts_f = np.bincount(orig_idx, minlength=n_unique).astype(np.float64)
        else:
            counts_f = counts.astype(np.float64)
        np.maximum(counts_f, 1.0, out=counts_f)  # avoid division by zero, in-place
        result = sums / counts_f
        return result.astype(orig_dtype)


def merge_las_tiles(input_path: Union[str, Path], 
                    output_dir: Optional[Union[str, Path]] = None,
                    output_format: str = 'las',
                    vote_dims: Optional[Set[str]] = None,
                    average_dims: Optional[Set[str]] = None):
    """
    Merge tiled LAS point clouds back into original LAS files.
    
    This is the inverse operation of LASTileProcessor. Handles both
    non-overlapping and overlapping tiles:
    
    - Non-overlap (overlap_factor=1): scatter by orig_idx to restore order
    - Overlap (overlap_factor>1): deduplicate via orig_idx, use majority
      voting for classification and averaging for configurable float dims
    
    All core operations are O(n) — no sorting required.
    
    Args:
        input_path: Path to directory containing tiled LAS files
        output_dir: Directory to save merged files (default: parent of input)
        output_format: Output format, 'las' or 'laz'
        vote_dims: Dimensions to aggregate via majority voting.
                   Default: {'classification'}
        average_dims: Dimensions to aggregate via averaging.
                      Default: empty (no averaging)
    """
    merger = LASMerger(
        input_path=input_path,
        output_dir=output_dir,
        output_format=output_format,
        vote_dims=vote_dims,
        average_dims=average_dims,
    )
    
    merger.merge_all()
    
    
if __name__ == "__main__":
    
    input_path = r"E:\data\DALES\dales_las\tile\pred"
    output_dir = r"E:\data\DALES\dales_las\tile\output"
    
    merge_las_tiles(
        input_path=input_path,
        output_dir=output_dir,
        # vote_dims={'classification'},  # 默认对 classification 多数投票
        # average_dims={'hag'},          # 可选：对 hag 取均值
    )
