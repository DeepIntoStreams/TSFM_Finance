# -*- coding: utf-8 -*-
"""
TimesFM 模型定义，用于从 Hugging Face Hub 加载自定义模型
"""

import torch
from transformers import PreTrainedModel, PretrainedConfig
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
from timesfm import TimesFmHparams


class TimesFMConfig(PretrainedConfig):
    """TimesFM 模型配置类"""
    
    model_type = "timesfm_custom"

    def __init__(
        self,
        num_layers=9,
        num_heads=6,
        num_kv_heads=6,
        head_dim=72,
        hidden_size=432,
        intermediate_size=1248,
        horizon_len=128,
        context_len=512,
        patch_len=128,
        input_patch_len=32,
        output_patch_len=128,
        use_positional_embedding=True,
        quantiles=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.horizon_len = horizon_len
        self.context_len = context_len
        self.patch_len = patch_len
        self.input_patch_len = input_patch_len
        self.output_patch_len = output_patch_len
        self.use_positional_embedding = use_positional_embedding
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TimesFMForHF(PreTrainedModel):
    """TimesFM 模型包装器，兼容 Hugging Face Hub"""
    
    config_class = TimesFMConfig

    def __init__(self, config: TimesFMConfig):
        super().__init__(config)
        
        # 直接构建模型配置，不依赖外部模型
        # 参考 timesfm 库的配置结构，手动创建所有必需属性
        from types import SimpleNamespace
        
        model_config = SimpleNamespace(
            # 架构参数（从 config 读取）
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            patch_len=config.patch_len,
            
            # 时间序列参数
            horizon_len=config.horizon_len,
            context_len=config.context_len,
            input_patch_len=config.input_patch_len,
            output_patch_len=config.output_patch_len,
            
            # 其他配置
            use_positional_embedding=config.use_positional_embedding,
            quantiles=config.quantiles,
            
            # TimesFM 特定的默认参数
            per_core_batch_size=getattr(config, 'per_core_batch_size', 32),
            num_experts=getattr(config, 'num_experts', 0),
            num_selected_experts=getattr(config, 'num_selected_experts', 0),
            use_freq=getattr(config, 'use_freq', True),
            freq_default=getattr(config, 'freq_default', [0]),
            vocab_size=getattr(config, 'vocab_size', 0),
            use_lm_head=getattr(config, 'use_lm_head', False),
            max_position_embeddings=getattr(config, 'max_position_embeddings', config.context_len),
            
            # Normalization 参数
            rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
            layer_norm_eps=getattr(config, 'layer_norm_eps', 1e-6),
            
            # Attention 参数
            attention_dropout=getattr(config, 'attention_dropout', 0.0),
            hidden_dropout=getattr(config, 'hidden_dropout', 0.0),
            
            # Activation 函数
            hidden_act=getattr(config, 'hidden_act', 'gelu'),
        )
        
        # 初始化解码器
        self.model = PatchedTimeSeriesDecoder(model_config)

    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.model(*args, **kwargs)
    
    def forecast(self, inputs, freq=None):
        """
        时间序列预测接口
        
        Args:
            inputs: 输入张量 [batch_size, context_len] 或 [batch_size, context_len, 1]
            freq: 频率参数（可选）
            
        Returns:
            预测结果张量
        """
        self.eval()
        with torch.no_grad():
            # 确保输入是正确的形状
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(-1)  # [B, T] -> [B, T, 1]
            
            return self.model(inputs, freq=freq)


# 注册模型配置，使 AutoConfig 可以识别
from transformers import AutoConfig
AutoConfig.register("timesfm_custom", TimesFMConfig)
