# 从 Hugging Face 加载自定义 TimesFM 模型

## 📁 文件说明

创建了以下文件来支持从 Hugging Face Hub 加载自定义 TimesFM 模型：

1. **`modeling_timesfm.py`** - TimesFM 模型定义和配置类
2. **`load_model_from_hf.py`** - 完整的模型加载和演示脚本
3. **`example_load_from_hf.py`** - 简单使用示例

## 🚀 快速开始

### 方式 1: 使用简单示例

```python
python example_load_from_hf.py
```

### 方式 2: 在你的代码中使用

```python
import torch
from modeling_timesfm import TimesFMForHF

# 加载模型
model = TimesFMForHF.from_pretrained("FinText/TimesFM_20M_2023_Augmented")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# 准备输入 (batch_size=2, context_len=100)
input_data = torch.randn(2, 100).to(model.device)

# 预测
with torch.no_grad():
    predictions = model.forecast(input_data)

print(f"预测形状: {predictions.shape}")
```

## 📋 依赖要求

确保安装了以下依赖：

```bash
pip install torch transformers timesfm
```

## 🔧 模型配置

你的模型 (`FinText/TimesFM_20M_2023_Augmented`) 配置：

- **层数**: 9 层
- **注意力头数**: 6
- **隐藏维度**: 432
- **上下文长度**: 512
- **预测长度**: 128
- **中间层维度**: 1248

## 📖 详细使用说明

### 加载模型

```python
from modeling_timesfm import TimesFMForHF

# 从 HuggingFace Hub 加载
model = TimesFMForHF.from_pretrained("FinText/TimesFM_20M_2023_Augmented")

# 或从本地路径加载
# model = TimesFMForHF.from_pretrained("./checkpoints/out_hf_batch/2023")
```

### 准备输入数据

TimesFM 模型接受以下输入格式：

```python
import torch

# 格式 1: [batch_size, context_len]
input_data = torch.randn(2, 100)

# 格式 2: [batch_size, context_len, 1]
input_data = torch.randn(2, 100, 1)
```

### 进行预测

```python
model.eval()
with torch.no_grad():
    # 使用 forecast 方法 (推荐)
    predictions = model.forecast(input_data)
    
    # 或直接调用 forward
    # predictions = model(input_data)
```

### 使用真实数据

```python
import pandas as pd
import torch

# 加载 CSV 数据
df = pd.read_csv('data/two_stocks_excess_returns.csv', 
                 index_col=0, parse_dates=True)

# 转换为张量 [batch_size, time_steps]
input_tensor = torch.tensor(df.values.T, dtype=torch.float32)

# 预测
with torch.no_grad():
    predictions = model.forecast(input_tensor)
```

## 🔍 常见问题

### Q: 如何修改模型配置？

A: 模型配置存储在 `config.json` 中，加载时会自动读取。如果需要修改，可以在初始化时传入：

```python
from modeling_timesfm import TimesFMConfig, TimesFMForHF

config = TimesFMConfig(
    num_layers=9,
    hidden_size=432,
    context_len=512,
    # ... 其他参数
)
model = TimesFMForHF(config)
```

### Q: 如何在 GPU 上运行？

A: 使用 `.to()` 方法：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
input_data = input_data.to(device)
```

### Q: 输出的形状是什么？

A: 输出形状为 `[batch_size, horizon_len]`，其中 `horizon_len` 是模型配置中的预测长度（默认 128）。

## 📝 注意事项

1. **自定义模型类**: `TimesFMForHF` 是一个自定义的 Transformers 模型类，需要本地的 `modeling_timesfm.py` 文件。

2. **trust_remote_code**: 如果 HuggingFace 仓库包含自定义代码文件，加载时需要设置 `trust_remote_code=True`。

3. **批次大小**: 根据可用内存调整批次大小，避免 OOM 错误。

## 📚 相关文件

- `Convert.py` - 将 PyTorch 检查点转换为 HuggingFace 格式的脚本
- `checkpoints/` - 本地模型检查点目录

## 🔗 模型链接

- HuggingFace 模型仓库: https://huggingface.co/FinText/TimesFM_20M_2023_Augmented
