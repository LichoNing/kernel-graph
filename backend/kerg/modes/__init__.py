"""
KernelGraph 开发模式管理器

支持多种开发模式：
- cuda: 标准CUDA C/C++模式
- cute: NVIDIA CUTLASS cute模式
- triton: OpenAI Triton模式 (未来支持)
- halide: Halide模式 (未来支持)
"""

from typing import Dict, Type, Optional
from .base import BaseMode, BaseParser, BaseGraphBuilder, BaseCodeGenerator
from .cuda_mode import CudaMode
from .cute_mode import CuteMode

# 注册所有开发模式
REGISTERED_MODES: Dict[str, Type[BaseMode]] = {
    'cuda': CudaMode,
    'cute': CuteMode,
}


def get_mode(mode_name: str) -> BaseMode:
    """获取指定名称的开发模式实例"""
    mode_name = mode_name.lower()
    if mode_name not in REGISTERED_MODES:
        raise ValueError(f"Unknown mode: {mode_name}. Available modes: {list(REGISTERED_MODES.keys())}")
    return REGISTERED_MODES[mode_name]()


def list_modes() -> list:
    """列出所有支持的开发模式"""
    return list(REGISTERED_MODES.keys())


__all__ = [
    'BaseMode', 'BaseParser', 'BaseGraphBuilder', 'BaseCodeGenerator',
    'CudaMode', 'CuteMode',
    'get_mode', 'list_modes', 'REGISTERED_MODES'
]
