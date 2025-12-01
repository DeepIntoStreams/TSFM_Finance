# -*- coding: utf-8 -*-
"""
Load TimesFM model from Hugging Face Hub
从 Hugging Face 加载自定义 TimesFM 模型
"""

import torch
import numpy as np
from modeling_timesfm import TimesFMForHF, TimesFMConfig


def load_model(repo_id="FinText/TimesFM_20M_2023_Augmented", device=None, trust_remote_code=False):
    """
    从 Hugging Face 加载 TimesFM 模型
    
    Args:
        repo_id: HF 仓库 ID，例如 "FinText/TimesFM_20M_2023_Augmented"
        device: 设备 ("cpu", "cuda", 或 None 自动选择)
        trust_remote_code: 是否信任远程代码（如果模型仓库包含自定义代码）
    
    Returns:
        加载好的模型实例
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"正在从 {repo_id} 加载模型...")
    
    # 从 Hugging Face Hub 加载模型
    model = TimesFMForHF.from_pretrained(
        repo_id,
        trust_remote_code=trust_remote_code
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型已成功加载到 {device}")
    print(f"  - 层数: {model.config.num_layers}")
    print(f"  - 注意力头数: {model.config.num_heads}")
    print(f"  - 隐藏维度: {model.config.hidden_size}")
    print(f"  - 上下文长度: {model.config.context_len}")
    print(f"  - 预测长度: {model.config.horizon_len}")
    
    return model


def demo_inference(model, device="cpu"):
    """
    演示如何使用加载的模型进行推理
    
    Args:
        model: 加载好的 TimesFM 模型
        device: 设备
    """
    import pandas as pd
    
    print("\n" + "="*60)
    print("演示: 使用模型进行时间序列预测")
    print("="*60)
    
    # 加载数据
    try:
        df = pd.read_csv('data/two_stocks_excess_returns.csv', index_col=0, parse_dates=True)
        print(f"✓ 数据加载成功，形状: {df.shape}")
    except FileNotFoundError:
        print("⚠ 数据文件未找到，使用模拟数据")
        # 生成模拟数据
        df = pd.DataFrame({
            'series1': np.random.randn(100).cumsum(),
            'series2': np.random.randn(100).cumsum()
        })
    
    # 准备输入数据
    context_len = min(model.config.context_len, len(df))
    input_data = torch.tensor(
        df.iloc[:context_len].values.T, 
        dtype=torch.float32
    ).to(device)
    
    print(f"\n输入数据:")
    print(f"  - 形状: {input_data.shape}")
    print(f"  - 批次大小: {input_data.shape[0]}")
    print(f"  - 时间步数: {input_data.shape[1]}")
    
    # 进行预测
    print(f"\n正在进行预测...")
    with torch.no_grad():
        # 使用 forecast 方法
        if hasattr(model, 'forecast'):
            output = model.forecast(input_data)
        else:
            output = model(input_data)
    
    print(f"\n预测结果:")
    print(f"  - 输出形状: {output.shape}")
    print(f"  - 输出类型: {type(output)}")
    
    # 显示部分预测结果
    if isinstance(output, torch.Tensor):
        output_np = output.cpu().numpy()
        print(f"\n前几个预测值 (第一个序列):")
        print(output_np[0, :min(10, output_np.shape[1])])
    
    return output


if __name__ == "__main__":
    print("TimesFM 模型加载示例")
    print("="*60)
    
    # 1. 加载模型
    model = load_model(
        repo_id="FinText/TimesFM_20M_2023_Augmented",
        device="cpu"  # 使用 "cuda" 如果有 GPU
    )
    
    # 2. 运行推理演示
    output = demo_inference(model, device="cpu")
    
    print("\n" + "="*60)
    print("✓ 示例完成！")
