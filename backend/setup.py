"""
KernelGraph安装配置
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="kerg",
    version="0.1.0",
    description="CUDA Kernel 计算图可视化与分析工具",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="KernelGraph Team",
    author_email="team@kernelgraph.dev",
    url="https://github.com/LichoNing/kernel-graph",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "kerg=kerg.cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Compilers",
    ],
    python_requires=">=3.10",
)
