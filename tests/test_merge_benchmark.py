"""
LASMerger 核心算法基准测试 — 效率 + 精度验证

在内存中模拟 tile→merge 流程，验证两个方面：

【效率测试】
- O(1) 重叠检测（max + 比较，无需 bincount）
- majority_vote 多数投票 O(n)（cls-iter 或 bincount-key，按 256MB 阈值自动选择）
- average 均值融合 O(n)（bincount with weights + 预计算 counts）
- 直接散射写入 O(n)

【精度测试】
- majority_vote: 验证 vote accuracy > single-copy accuracy（投票一定优于单次观测）
- average:       验证均值 MSE ≈ single_MSE / n_repeat（理论方差缩减 1/n）
- scatter:       验证不变属性（坐标/intensity）bit-exact 还原

跳过 LAS 文件 I/O，专注于纯算法测试。
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.merge_las import LASMerger


# ─────────────────────────────────────────────────────────────────────────────
#  Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_overlap_data(n_unique: int, overlap_factor: int = 2, n_classes: int = 9,
                          noise_ratio: float = 0.2, float_noise_std: float = 0.1,
                          seed: int = 42):
    """
    模拟 tile overlap 产生的数据，并保留 ground truth 用于精度验证。
    """
    n_repeat = overlap_factor ** 2
    rng = np.random.default_rng(seed)
    
    # 每个原始点重复 n_repeat 次，然后打乱
    base_idx = np.arange(n_unique, dtype=np.uint32)
    staging_orig_idx = np.tile(base_idx, n_repeat)
    shuffle_order = rng.permutation(len(staging_orig_idx))
    staging_orig_idx = staging_orig_idx[shuffle_order]
    
    # Ground truth labels
    gt_labels = rng.integers(0, n_classes, size=n_unique, dtype=np.uint8)
    
    # 构造含噪声的 classification
    staging_cls = gt_labels[staging_orig_idx].copy()
    noise_mask = rng.random(len(staging_cls)) < noise_ratio
    staging_cls[noise_mask] = rng.integers(0, n_classes, size=noise_mask.sum(), dtype=np.uint8)
    
    # Ground truth float
    gt_float = rng.uniform(0, 50, size=n_unique).astype(np.float32)
    
    # 构造含噪声的连续属性
    staging_float = gt_float[staging_orig_idx].copy()
    staging_float += rng.normal(0, float_noise_std, size=len(staging_float)).astype(np.float32)
    
    # 模拟不变属性 (如坐标/intensity) — 所有副本完全相同
    gt_identity = rng.uniform(-1000, 1000, size=n_unique).astype(np.float64)
    staging_identity = gt_identity[staging_orig_idx].copy()  # 无噪声
    
    return {
        'staging_orig_idx': staging_orig_idx,
        'staging_cls': staging_cls,
        'staging_float': staging_float,
        'staging_identity': staging_identity,
        'gt_labels': gt_labels,
        'gt_float': gt_float,
        'gt_identity': gt_identity,
        'noise_ratio': noise_ratio,
        'float_noise_std': float_noise_std,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Accuracy Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_accuracy(data: dict, result_vote: np.ndarray, result_avg: np.ndarray,
                    result_scatter: np.ndarray, n_unique: int):
    """
    验证合并算法的精度。
    
    Correctness criteria (must ALL pass):
        1. Vote accuracy > single-copy accuracy (majority voting MUST improve)
        2. Vote accuracy > 90% absolute floor
        3. Average MSE < single_MSE (averaging MUST reduce noise)
        4. MSE ratio ≈ 1/n_repeat within 3x tolerance
        5. Identity scatter: bit-exact match with ground truth
    """
    gt_labels = data['gt_labels']
    gt_float = data['gt_float']
    gt_identity = data['gt_identity']
    n_repeat = len(data['staging_orig_idx']) // n_unique
    
    metrics = {}
    all_pass = True
    
    # --- 1. Majority vote accuracy ---
    vote_correct = int(np.sum(result_vote == gt_labels))
    vote_accuracy = vote_correct / n_unique
    metrics['vote_accuracy'] = vote_accuracy
    
    # Single-copy accuracy: pick one random observation per point (last-write-wins)
    single_obs_cls = np.empty(n_unique, dtype=data['staging_cls'].dtype)
    single_obs_cls[data['staging_orig_idx']] = data['staging_cls']
    single_correct = int(np.sum(single_obs_cls == gt_labels))
    single_accuracy = single_correct / n_unique
    metrics['single_accuracy'] = single_accuracy
    
    # Vote MUST beat single-copy (fundamental property of majority voting with > 50% per-copy accuracy)
    vote_pass = (vote_accuracy > single_accuracy) and (vote_accuracy > 0.90)
    all_pass &= vote_pass
    
    # --- 2. Average MSE improvement ---
    single_obs_float = np.empty(n_unique, dtype=np.float32)
    single_obs_float[data['staging_orig_idx']] = data['staging_float']
    single_mse = float(np.mean((single_obs_float - gt_float) ** 2))
    
    avg_mse = float(np.mean((result_avg - gt_float) ** 2))
    mse_ratio = avg_mse / max(single_mse, 1e-15)
    metrics['single_mse'] = single_mse
    metrics['avg_mse'] = avg_mse
    metrics['mse_ratio'] = mse_ratio
    
    # Averaging MUST reduce MSE, and should be approximately 1/n_repeat
    expected_ratio = 1.0 / n_repeat
    avg_pass = (avg_mse < single_mse) and (mse_ratio < expected_ratio * 3)
    all_pass &= avg_pass
    
    # --- 3. Identity scatter: bit-exact ---
    identity_match = np.array_equal(result_scatter, gt_identity)
    metrics['identity_exact'] = identity_match
    all_pass &= identity_match
    
    metrics['all_pass'] = all_pass
    return metrics


def print_accuracy(metrics: dict, n_unique: int, n_repeat: int):
    """Print accuracy verification results."""
    vote_acc = metrics['vote_accuracy']
    single_acc = metrics['single_accuracy']
    vote_ok = '  PASS' if (vote_acc > single_acc and vote_acc > 0.90) else '**FAIL**'
    
    mse_ratio = metrics['mse_ratio']
    avg_ok = '  PASS' if mse_ratio < 1.0 else '**FAIL**'
    
    id_ok = '  PASS' if metrics['identity_exact'] else '**FAIL**'
    
    all_ok = '  ALL PASS' if metrics['all_pass'] else '**SOME FAILED**'
    
    print(f"  ┌─ Accuracy Verification ─────────────────────────────────────────┐")
    print(f"  │ Vote vs single-copy:  {vote_acc:.4%} vs {single_acc:.4%}  "
          f"(+{vote_acc-single_acc:.2%})    {vote_ok} │")
    print(f"  │ MSE ratio (avg/single): x{mse_ratio:.4f}  "
          f"(theory: x{1/n_repeat:.4f} for {n_repeat}x)     {avg_ok} │")
    print(f"  │ Identity scatter:      bit-exact = {str(metrics['identity_exact']):5s}"
          f"                        {id_ok} │")
    print(f"  │ Overall:  {all_ok:>57s} │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")


# ─────────────────────────────────────────────────────────────────────────────
#  Combined Benchmark: Speed + Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(n_unique: int, overlap_factor: int = 2, n_classes: int = 9,
                  noise_ratio: float = 0.2):
    """运行一次完整的 merge 核心管线基准测试: 速度 + 精度"""
    n_repeat = overlap_factor ** 2
    total_concat = n_unique * n_repeat
    
    print(f"\n{'='*70}")
    print(f"  n_unique={n_unique:>12,}  overlap={n_repeat}x  "
          f"total_concat={total_concat:>14,}  classes={n_classes}")
    print(f"{'='*70}")
    
    # --- Generate data ---
    t0 = time.perf_counter()
    data = generate_overlap_data(n_unique, overlap_factor, n_classes, noise_ratio)
    t_gen = time.perf_counter() - t0
    staging_orig_idx = data['staging_orig_idx']
    staging_cls = data['staging_cls']
    staging_float = data['staging_float']
    staging_identity = data['staging_identity']
    mem_mb = (staging_orig_idx.nbytes + staging_cls.nbytes + 
              staging_float.nbytes + staging_identity.nbytes) / 1024**2
    print(f"  Data generation:       {t_gen:8.3f}s  ({mem_mb:.1f} MB)")
    
    # --- O(1) overlap detection (replaces full bincount) ---
    t0 = time.perf_counter()
    n_output = int(staging_orig_idx.max()) + 1
    has_overlap = total_concat > n_output
    t_dedup = time.perf_counter() - t0
    assert n_output == n_unique
    print(f"  overlap detect:        {t_dedup:8.6f}s  (has_overlap={has_overlap})")
    
    # --- Direct scatter (identity dims) ---
    t0 = time.perf_counter()
    result_identity = np.empty(n_output, dtype=staging_identity.dtype)
    result_identity[staging_orig_idx] = staging_identity
    t_scatter = time.perf_counter() - t0
    print(f"  scatter (identity):    {t_scatter:8.3f}s")
    
    # --- majority_vote ---
    vote_mem_mb = n_output * n_classes * 8 / 1024**2
    path = "bincount-key" if vote_mem_mb < 256 else "cls-iter"
    t0 = time.perf_counter()
    result_vote = LASMerger._majority_vote(staging_cls, staging_orig_idx, n_output)
    t_vote = time.perf_counter() - t0
    print(f"  majority_vote ({path:>12s}): {t_vote:8.3f}s  ({vote_mem_mb:.1f} MB)")
    
    # --- _average (with pre-computed counts) ---
    t0 = time.perf_counter()
    counts = np.bincount(staging_orig_idx, minlength=n_output)
    t_counts = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    result_avg = LASMerger._average(staging_float, staging_orig_idx, n_output, counts=counts)
    t_avg_only = time.perf_counter() - t0
    t_avg = t_counts + t_avg_only
    print(f"  average (counts+avg):  {t_avg:8.3f}s  (counts={t_counts:.3f} + avg={t_avg_only:.3f})")
    
    # --- Total ---
    t_total = t_dedup + t_scatter + t_vote + t_avg
    throughput = n_unique / t_total / 1e6
    print(f"  ─────────────────────────────────")
    print(f"  Total compute:         {t_total:8.3f}s  ({throughput:.2f} M pts/s)")
    
    # --- Accuracy verification ---
    metrics = verify_accuracy(data, result_vote, result_avg, result_identity, n_unique)
    print_accuracy(metrics, n_unique, n_repeat)
    
    return {
        'n_unique': n_unique,
        'overlap': n_repeat,
        'total_concat': total_concat,
        'n_classes': n_classes,
        't_dedup': t_dedup,
        't_scatter': t_scatter,
        't_vote': t_vote,
        't_average': t_avg,
        't_total': t_total,
        'throughput_Mpts': throughput,
        'vote_accuracy': metrics['vote_accuracy'],
        'single_accuracy': metrics['single_accuracy'],
        'mse_ratio': metrics['mse_ratio'],
        'identity_exact': metrics['identity_exact'],
        'all_pass': metrics['all_pass'],
    }


def run_benchmarks():
    """对不同数据量级运行速度 + 精度验证"""
    print("=" * 70)
    print("  LASMerger Benchmark — Speed + Accuracy Verification")
    print("  (Pure compute, no LAS I/O, all O(n))")
    print("=" * 70)
    
    test_configs = [
        # (n_unique, overlap_factor, n_classes)
        (      10_000, 2,  9),
        (     100_000, 2,  9),
        (   1_000_000, 2,  9),
        (   5_000_000, 2,  9),
        (  10_000_000, 2,  9),
        (  20_000_000, 2,  9),
        # overlap_factor=3
        (   1_000_000, 3,  9),
        (   5_000_000, 3,  9),
        # 多类别
        (   1_000_000, 2, 50),
        (   5_000_000, 2, 50),
    ]
    
    results = []
    for n_unique, ovlp, ncls in test_configs:
        staging_mb = n_unique * (ovlp**2) * (4 + 1 + 4 + 8) / 1024**2
        vote_mb = n_unique * ncls * 8 / 1024**2
        total_est_mb = staging_mb + vote_mb
        
        if total_est_mb > 16000:
            print(f"\n  SKIP n_unique={n_unique:,} overlap={ovlp}^2 classes={ncls} "
                  f"(estimated {total_est_mb:.0f} MB)")
            continue
        
        try:
            r = benchmark_one(n_unique, ovlp, ncls)
            results.append(r)
        except MemoryError:
            print(f"  MemoryError! Skipping.")
            break
    
    # ─── Summary Table ───
    print(f"\n\n{'='*130}")
    print(f"  SUMMARY — Speed + Accuracy")
    print(f"{'='*130}")
    hdr = (f"  {'n_unique':>12s}  {'ovlp':>4s}  {'concat':>12s}  {'cls':>3s}  "
           f"{'scat':>7s}  {'vote':>7s}  {'avg':>7s}  "
           f"{'TOTAL':>7s}  {'Mpts/s':>7s}  "
           f"{'vote%':>7s}  {'single%':>7s}  {'MSE_r':>7s}  {'id_ok':>5s}  {'status':>6s}")
    print(hdr)
    sep = (f"  {'-'*12}  {'-'*4}  {'-'*12}  {'-'*3}  "
           f"{'-'*7}  {'-'*7}  {'-'*7}  "
           f"{'-'*7}  {'-'*7}  "
           f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*6}")
    print(sep)
    for r in results:
        status = '  PASS' if r['all_pass'] else '**FAIL'
        id_ok = 'exact' if r['identity_exact'] else 'DIFF'
        print(f"  {r['n_unique']:>12,}  {r['overlap']:>4d}x  {r['total_concat']:>12,}  {r['n_classes']:>3d}  "
              f"{r['t_scatter']:>7.3f}  {r['t_vote']:>7.3f}  {r['t_average']:>7.3f}  "
              f"{r['t_total']:>7.3f}  {r['throughput_Mpts']:>7.2f}  "
              f"{r['vote_accuracy']:>6.2%}  {r['single_accuracy']:>6.2%}  "
              f"{r['mse_ratio']:>7.4f}  {id_ok:>5s}  {status:>6s}")
    
    # ─── Overall Verdict ───
    all_pass = all(r['all_pass'] for r in results)
    print(f"\n  {'='*60}")
    if all_pass:
        print(f"  ALL {len(results)} TESTS PASSED")
        print(f"  - Majority vote always beats single-copy accuracy")
        print(f"  - MSE reduction matches theoretical 1/n_repeat")
        print(f"  - Identity attributes are bit-exact")
    else:
        failed = sum(1 for r in results if not r['all_pass'])
        print(f"  WARNING: {failed}/{len(results)} tests FAILED!")
    print(f"  {'='*60}\n")


if __name__ == "__main__":
    run_benchmarks()
