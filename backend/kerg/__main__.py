"""
KernelGraph - CUDA Kernel 计算图可视化工具

用法:
    python -m kerg parse --input kernel.cu --output graph.json
    python -m kerg visualize --input kernel.cu --output graph.json
    python -m kerg prompt --input kernel.cu --output prompt.txt
"""

from .cli.main import cli

if __name__ == '__main__':
    cli()
