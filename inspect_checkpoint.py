# -*- coding: utf-8 -*-
"""
检查 checkpoint 文件，推断模型配置
"""

import torch
import os

checkpoint_path = "checkpoints/timesfm_combined_pretrained_2022.pt"

if not os.path.exists(checkpoint_path):
    print(f"文件不存在: {checkpoint_path}")
    exit(1)

print(f"正在加载: {checkpoint_path}")
state_dict = torch.load(checkpoint_path, map_location="cpu")

print("\n" + "="*60)
print("权重文件中的所有键（前20个）:")
print("="*60)
for i, key in enumerate(list(state_dict.keys())[:20]):
    shape = state_dict[key].shape
    print(f"{key:60s} {shape}")

print("\n" + "="*60)
print("查找输入层（input_ff_layer）的维度:")
print("="*60)

# 查找输入层的权重
for key in state_dict.keys():
    if 'input_ff_layer' in key and 'hidden_layer.0.weight' in key:
        shape = state_dict[key].shape
        print(f"找到: {key}")
        print(f"  形状: {shape}")
        print(f"  输出维度 (intermediate_size): {shape[0]}")
        print(f"  输入维度: {shape[1]}")
        print(f"  推断: 如果 patch_len=128, 则可能:")
        print(f"    - input_dim = patch_len * 2 = {128 * 2} (如果是 {shape[1]})")
        print(f"    - input_dim = patch_len / 2 = {128 / 2} (如果是 {shape[1]})")
        break

print("\n" + "="*60)
print("查找输出层（horizon_ff_layer）的维度:")
print("="*60)

for key in state_dict.keys():
    if 'horizon_ff_layer' in key and 'output_layer.weight' in key:
        shape = state_dict[key].shape
        print(f"找到: {key}")
        print(f"  形状: {shape}")
        print(f"  输出维度 (model_dims): {shape[0]}")
        print(f"  输入维度: {shape[1]}")
        break

print("\n" + "="*60)
print("统计:")
print("="*60)
print(f"总参数数量: {len(state_dict)}")
print(f"总权重大小: {sum(p.numel() for p in state_dict.values()):,} 个参数")
