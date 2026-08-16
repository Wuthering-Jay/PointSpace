"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        if "img_num" in batch[0].keys():
            max_img_num = max([d["img_num"] for d in batch])
        
        result = {}
        for key in batch[0]:
            items = [d[key] for d in batch]

            # Transform bookkeeping is identical for all samples and is not a
            # point tensor.  Keeping one copy also makes raw-dataset debugging
            # work when no final Collect transform is configured.
            if key == "index_valid_keys":
                result[key] = list(items[0])
                continue

            # Point-to-patch indices are local inside each sample.  Convert
            # valid entries to indices of the concatenated [sum(P), C] feature
            # tensor while preserving -1 for points without image coverage.
            if key == "dino_patch_index":
                adjusted = []
                patch_start = 0
                for sample, item in zip(batch, items):
                    item = (
                        item.clone()
                        if isinstance(item, torch.Tensor)
                        else torch.as_tensor(item).clone()
                    )
                    item[item >= 0] += patch_start
                    adjusted.append(item)
                    patch_start += int(sample["dino_feature"].shape[0])
                result[key] = collate_fn(adjusted)
                continue
            
            # Special handling for 'sub' Cluster dict - needs CSR-style merging
            if key == "sub" and isinstance(items[0], dict) and "pointer" in items[0]:
                # Merge multiple Cluster dicts into one
                # Each sub has pointer (cumsum) and value (point indices)
                # Need to offset value indices and recompute pointer
                merged_pointer = [torch.tensor([0], dtype=torch.long)]
                merged_value = []
                value_offset = 0
                
                for sub_dict in items:
                    ptr = sub_dict["pointer"]
                    val = sub_dict["value"]
                    if isinstance(ptr, np.ndarray):
                        ptr = torch.from_numpy(ptr).long()
                    if isinstance(val, np.ndarray):
                        val = torch.from_numpy(val).long()
                    
                    # Append sizes (diff of pointers) to build new pointer
                    sizes = ptr[1:] - ptr[:-1]
                    merged_pointer.append(sizes)
                    
                    # Offset value indices by cumulative point count
                    merged_value.append(val + value_offset)
                    value_offset += len(val)
                
                merged_pointer = torch.cumsum(torch.cat(merged_pointer), dim=0)
                merged_value = torch.cat(merged_value)
                result[key] = {"pointer": merged_pointer, "value": merged_value}
                continue
            
            # Handle offset keys
            if "offset" in key:
                result[key] = torch.cumsum(
                    collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
                    dim=0,
                )
                continue
            
            # Handle grid_size (scalar, use first sample)
            if key == "grid_size":
                result[key] = batch[0][key]
                continue
            
            # Handle correspondence
            if "correspondence" in key:
                result[key] = collate_fn([
                    F.pad(
                        d[key].permute(0, 2, 1),
                        (0, max_img_num - d[key].shape[1]),
                        value=-1,
                    ).permute(0, 2, 1)
                    for d in batch
                ])
                continue
            
            # Default: recursive collate
            result[key] = collate_fn(items)
        
        return result
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        if "instance" in batch.keys():
            offset = batch["offset"]
            start = 0
            num_instance = 0
            for i in range(len(offset)):
                if i % 2 == 0:
                    num_instance = max(batch["instance"][start : offset[i]])
                if i % 2 != 0:
                    mask = batch["instance"][start : offset[i]] != -1
                    batch["instance"][start : offset[i]] += num_instance * mask
                start = offset[i]
        offset_assets = [asset for asset in batch.keys() if "offset" in asset]
        for offset_asset in offset_assets:
            batch[offset_asset] = torch.cat(
                [batch[offset_asset][1:-1:2], batch[offset_asset][-1].unsqueeze(0)],
                dim=0,
            )
        if "img_num" in batch.keys():
            n = batch["img_num"].shape[0]
            num_pairs = n // 2
            len_pairs = num_pairs * 2
            pairs_tensor = batch["img_num"][:len_pairs]

            if num_pairs == 0:
                pass
            else:
                summed_pairs = pairs_tensor.view(-1, 2).sum(dim=1)
                if n % 2 != 0:
                    last_element = batch["img_num"][-1:]
                    result = torch.cat((summed_pairs, last_element))
                else:
                    result = summed_pairs
                batch["img_num"] = result
        correspondence_assets = [
            asset for asset in batch.keys() if "correspondence" in asset
        ]
        for correspondence_asset in correspondence_assets:
            offset = batch["offset"]
            start = 0
            N, v, n = batch[correspondence_asset].shape
            v2 = v * 2
            batch_correspondence_mix = -torch.ones((N, v2, n))
            for i in range(len(offset)):
                if i % 2 == 0:
                    batch_correspondence_mix[start : offset[i], 0:v] = batch[
                        correspondence_asset
                    ][start : offset[i], 0:v]
                if i % 2 != 0:
                    batch_correspondence_mix[start : offset[i], v:] = batch[
                        correspondence_asset
                    ][start : offset[i], 0:v]
                start = offset[i]
            if len(offset) % 2 == 0:
                pass
            else:
                start = 0 if len(offset) == 1 else offset[-2]
                batch_correspondence_mix[start:N, -v:] = batch[correspondence_asset][
                    start:N, -v:
                ]
            batch[correspondence_asset] = batch_correspondence_mix
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
