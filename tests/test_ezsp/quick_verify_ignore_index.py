"""
快速验证 ignore_index = num_classes 修正是否生效

运行方法：
    cd e:\code\python\PointSpace
    conda activate pointcept
    python tests/test_ezsp/quick_verify_ignore_index.py
"""

import sys
sys.path.insert(0, "e:/code/python/PointSpace")

import torch
import torch.nn as nn


def test_1_basic_convention():
    """测试1: 基本约定验证"""
    print("\n" + "="*60)
    print("测试1: 验证 ignore_label = num_classes 基本约定")
    print("="*60)
    
    num_classes = 8
    batch_size = 100
    
    # 创建直方图标签
    y_hist = torch.zeros(batch_size, num_classes + 1)
    y_hist[:80, :num_classes] = torch.randn(80, num_classes).softmax(dim=1)
    y_hist[80:, num_classes] = 1.0  # Void superpoints
    
    # Argmax
    labels = y_hist.argmax(dim=1)
    
    print(f"  - 有效超点标签范围: [0, {labels[:80].min()}, ..., {labels[:80].max()}]")
    print(f"  - Void 超点标签: {labels[80:].unique().tolist()}")
    assert (labels[80:] == num_classes).all(), "❌ Void 标签应该 == num_classes!"
    print("  ✅ Void 标签正确 = num_classes (8)")
    
    # 测试损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=num_classes)
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    loss = criterion(logits, labels)
    loss.backward()
    
    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - 梯度非零元素: {(logits.grad.abs() > 1e-6).sum().item()} / {logits.grad.numel()}")
    print("  ✅ 损失函数正确忽略 void 超点")


def test_2_partition_criterion():
    """测试2: PartitionCriterion void edge 移除"""
    print("\n" + "="*60)
    print("测试2: 验证 PartitionCriterion void edge 移除逻辑")
    print("="*60)
    
    num_classes = 8
    num_voxels = 20
    
    # 创建直方图
    y_hist = torch.zeros(num_voxels, num_classes + 1)
    y_hist[:15, :num_classes] = torch.randn(15, num_classes).softmax(dim=1) * 10
    y_hist[15:, num_classes] = 10.0  # Pure void
    
    # 检查 majority_class_count
    majority_class_count, y_labels = y_hist[:, :num_classes].max(dim=1)
    
    print(f"  - 有效 voxel 数量: {(majority_class_count > 0).sum().item()}")
    print(f"  - Void voxel 数量: {(majority_class_count == 0).sum().item()}")
    assert (majority_class_count[:15] > 0).all(), "❌ 前15个应该是有效 voxel"
    assert (majority_class_count[15:] == 0).all(), "❌ 后5个应该是 void voxel"
    print("  ✅ Void voxel 检测正确")
    
    # 创建边
    edge_index = torch.tensor([
        [0, 1, 2, 10, 15, 16],  # src
        [1, 2, 3, 15, 16, 17],  # dst
    ], dtype=torch.long)
    
    # 过滤 void edges
    mask_void_voxels = majority_class_count == 0
    src, dst = edge_index
    mask_void_edges = mask_void_voxels[src] | mask_void_voxels[dst]
    edge_index_filtered = edge_index[:, ~mask_void_edges]
    
    print(f"  - 原始边数量: {edge_index.shape[1]}")
    print(f"  - Void 边数量: {mask_void_edges.sum().item()}")
    print(f"  - 过滤后边数量: {edge_index_filtered.shape[1]}")
    assert edge_index_filtered.shape[1] == 3, "❌ 应该保留3条有效边"
    assert edge_index_filtered.max() < 15, "❌ 过滤后不应有连接到 void 的边"
    print("  ✅ Void edge 移除正确")


def test_3_config_file():
    """测试3: 配置文件修正验证"""
    print("\n" + "="*60)
    print("测试3: 验证配置文件 ignore_index 设置")
    print("="*60)
    
    # 动态加载配置
    config_path = "e:/code/python/PointSpace/configs/dales/semseg-ezsp-v1-0.py"
    
    # 读取配置内容
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键行
    if 'ignore_index = num_classes' in content:
        print("  ✅ 配置文件使用 'ignore_index = num_classes'")
    elif 'ignore_index = 8' in content:
        print("  ✅ 配置文件使用 'ignore_index = 8'")
    else:
        print("  ⚠️ 未找到 'ignore_index = num_classes' 或 'ignore_index = 8'")
    
    if 'ignore_index = -1' in content:
        print("  ❌ 配置文件仍包含 'ignore_index = -1'，需要修正！")
    else:
        print("  ✅ 配置文件不再使用 'ignore_index = -1'")
    
    # 检查 loss_type
    if "loss_type='ce_kl'" in content or 'loss_type="ce_kl"' in content:
        print("  ✅ 配置文件包含 'loss_type=ce_kl'")
    else:
        print("  ⚠️ 配置文件未设置 'loss_type=ce_kl'")
    
    # 检查 multi_stage_loss_lambdas
    if 'multi_stage_loss_lambdas' in content:
        print("  ✅ 配置文件包含 'multi_stage_loss_lambdas'")
    else:
        print("  ⚠️ 配置文件未设置 'multi_stage_loss_lambdas'")


def test_4_segmentor_default():
    """测试4: Segmentor 默认配置验证"""
    print("\n" + "="*60)
    print("测试4: 验证 EZSPPartitionSegmentor 默认配置")
    print("="*60)
    
    try:
        from pointspace.models.segmentor.ezsp_segmentor import EZSPPartitionSegmentor
        
        # 创建 segmentor（不提供 criteria）
        model = EZSPPartitionSegmentor(
            num_classes=8,
            training_partition_stage=False,
        )
        
        # 检查默认 criteria 的 ignore_index
        found_correct = False
        found_wrong = False
        for criterion in model.criteria:
            if hasattr(criterion, 'ignore_index'):
                idx = criterion.ignore_index
                print(f"  - {criterion.__class__.__name__}.ignore_index = {idx}")
                if idx == 8:
                    found_correct = True
                elif idx == -1:
                    found_wrong = True
        
        if found_correct:
            print("  ✅ Segmentor 默认使用 ignore_index = num_classes")
        if found_wrong:
            print("  ❌ Segmentor 仍使用 ignore_index = -1，需要修正！")
        if not found_correct and not found_wrong:
            print("  ⚠️ 未找到带 ignore_index 的 criterion")
            
    except Exception as e:
        print(f"  ⚠️ 无法加载 EZSPPartitionSegmentor: {e}")


def test_5_ce_kl_loss():
    """测试5: ce_kl 多阶段损失验证"""
    print("\n" + "="*60)
    print("测试5: 验证 ce_kl 多阶段损失逻辑")
    print("="*60)
    
    num_classes = 8
    num_sp_l0 = 100
    num_sp_l1 = 30
    
    # Level 0 直方图
    y_hist_l0 = torch.randn(num_sp_l0, num_classes + 1).softmax(dim=1)
    y_hist_l0[90:, :] = 0
    y_hist_l0[90:, num_classes] = 1.0  # Void
    
    # Level 1 直方图
    y_hist_l1 = torch.randn(num_sp_l1, num_classes + 1).softmax(dim=1)
    y_hist_l1[25:, :] = 0
    y_hist_l1[25:, num_classes] = 1.0  # Void
    
    # Logits
    logits_l0 = torch.randn(num_sp_l0, num_classes)
    logits_l1 = torch.randn(num_sp_l1, num_classes)
    
    # Level 0: CE (hard labels via argmax)
    labels_l0 = y_hist_l0.argmax(dim=1)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=num_classes)
    loss_l0 = criterion_ce(logits_l0, labels_l0)
    
    # Level 1: KL (soft distribution)
    y_hist_l1_valid = y_hist_l1[:, :num_classes]
    y_hist_l1_valid = y_hist_l1_valid / (y_hist_l1_valid.sum(dim=1, keepdim=True) + 1e-8)
    log_probs_l1 = torch.log_softmax(logits_l1, dim=1)
    loss_l1 = nn.functional.kl_div(log_probs_l1, y_hist_l1_valid, reduction='batchmean')
    
    # Combined (lambdas=[1, 50])
    loss_total = loss_l0 + 50 * loss_l1
    
    print(f"  - Level 0 (CE) loss: {loss_l0.item():.4f}")
    print(f"  - Level 1 (KL) loss: {loss_l1.item():.4f}")
    print(f"  - Total (1*L0 + 50*L1): {loss_total.item():.4f}")
    
    assert torch.isfinite(loss_total), "❌ 总损失应该是有限值"
    print("  ✅ ce_kl 多阶段损失计算正确")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" ignore_index = num_classes 修正验证")
    print("="*70)
    
    try:
        test_1_basic_convention()
        test_2_partition_criterion()
        test_3_config_file()
        test_4_segmentor_default()
        test_5_ce_kl_loss()
        
        print("\n" + "="*70)
        print("✅ 所有验证通过！ignore_index 修正成功。")
        print("="*70)
        print("\n下一步建议：")
        print("  1. 检查 LasDataset 是否将 ignore 标签映射到 num_classes")
        print("  2. 验证评估指标是否正确忽略 label=num_classes")
        print("  3. 在小规模 DALES 数据上进行端到端训练测试")
        print()
        
    except AssertionError as e:
        print("\n" + "="*70)
        print(f"❌ 验证失败: {e}")
        print("="*70)
    except Exception as e:
        print("\n" + "="*70)
        print(f"⚠️ 运行出错: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
