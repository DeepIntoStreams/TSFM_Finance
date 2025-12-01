# -*- coding: utf-8 -*-
"""
简单示例: 从 Hugging Face 加载并使用 TimesFM 模型

这个脚本展示了如何:
1. 从 Hugging Face Hub 加载自定义 TimesFM 模型
2. 准备时间序列输入数据
3. 进行预测
"""

import torch
import numpy as np
from modeling_timesfm import TimesFMForHF


def main():
    # ========== 第 1 步: 加载模型 ==========
    print("正在加载模型...")
    
    model = TimesFMForHF.from_pretrained(
        "FinText/TimesFM_20M_2023_Augmented"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型已加载到 {device}")
    print(f"  配置: {model.config.num_layers} 层, "
          f"{model.config.hidden_size} 隐藏维度")
    
    # ========== 第 2 步: 准备输入数据 ==========
    # 示例: 创建 2 个时间序列，每个长度为 100
    batch_size = 2
    context_len = 100
    
    # 生成模拟数据 (你可以替换成真实数据)
    input_data = torch.randn(batch_size, context_len).to(device)
    
    print(f"\n输入数据形状: {input_data.shape}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 上下文长度: {context_len}")
    
    # ========== 第 3 步: 进行预测 ==========
    print("\n正在预测...")
    
    with torch.no_grad():
        # 方式 1: 直接调用模型
        predictions = model.forecast(input_data)
        
        # 或者方式 2: 使用 forward 方法
        # predictions = model(input_data)
    
    print(f"✓ 预测完成!")
    print(f"  预测结果形状: {predictions.shape}")
    print(f"  预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # ========== 第 4 步: 处理结果 ==========
    # 转换为 numpy 数组便于后续处理
    predictions_np = predictions.cpu().numpy()
    
    print(f"\n第一个序列的前 10 个预测值:")
    print(predictions_np[0, :10])
    
    return predictions_np


if __name__ == "__main__":
    predictions = main()
    print("\n✓ 完成!")
