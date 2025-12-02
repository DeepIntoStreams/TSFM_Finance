# -*- coding: utf-8 -*-
"""
调试脚本：查看 TimesFM 模型配置的所有属性
"""

import torch
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
from timesfm import TimesFmHparams

device = "cpu"

hparams = TimesFmHparams(
    backend=device,
    per_core_batch_size=32,
    horizon_len=128,
    num_layers=9,  # 使用实际的层数
    context_len=512,
)

# 直接创建解码器来查看需要哪些配置属性
print("="*60)
print("检查 TimesFmHparams 的属性:")
print("="*60)

for attr in sorted(dir(hparams)):
    if not attr.startswith('_'):
        try:
            value = getattr(hparams, attr)
            if not callable(value):
                print(f"{attr:30s} = {value}")
        except:
            pass

print("\n" + "="*60)
print("现在尝试创建一个临时的模型配置对象")
print("="*60)

# 不加载权重，只创建配置
from types import SimpleNamespace

test_config = SimpleNamespace(
    num_layers=9,
    num_heads=6,
    num_kv_heads=6,
    head_dim=72,
    hidden_size=432,
    intermediate_size=1248,
    patch_len=128,
    horizon_len=128,
    context_len=512,
    input_patch_len=32,
    output_patch_len=128,
    use_positional_embedding=True,
    quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    per_core_batch_size=32,
    num_experts=0,
    num_selected_experts=0,
    use_freq=True,
    freq_default=[0],
    vocab_size=0,
    use_lm_head=False,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
    layer_norm_eps=1e-6,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    hidden_act='gelu',
)

print("\n尝试创建 PatchedTimeSeriesDecoder...")
try:
    decoder = PatchedTimeSeriesDecoder(test_config)
    print("✓ 成功创建解码器!")
    print(f"\n解码器结构:")
    print(decoder)
except Exception as e:
    print(f"✗ 创建失败: {e}")
    print(f"\n错误类型: {type(e).__name__}")
    import traceback
    traceback.print_exc()
