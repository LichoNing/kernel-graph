# KernelGraph

CUDA Kernel 计算图可视化与分析工具（**Tcore算子专用**）

## 功能特性

- **Kernel→Graph**: 将 CUDA kernel 代码转换为结构化的图表示
- **Graph→Kernel**: 将图结构反向转换为可编译的 CUDA kernel 代码
- **Graph-UI**: 交互式可视化界面
- **LLM集成**: 生成用于LLM的结构化prompt

## 安装

### 前置要求
- Python 3.10+
- (可选) Clang/LLVM 16+ - 用于更完整的C++语法支持

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/LichoNing/kernel-graph.git
cd kernel-graph

# 进入后端目录
cd backend

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装KernelGraph包（开发模式）
pip install -e .
```

### 快速开始

### 命令行使用

```bash
# 解析kernel源码并生成图
python -m kerg parse --input your_kernel.cu --output graph.json

# 生成LLM prompt
python -m kerg prompt --input your_kernel.cu --output prompt.txt

# 从图生成kernel代码
python -m kerg generate --input graph.json --output generated_kernel.cu
```

### API服务

```bash
# 启动API服务
python -m kerg serve --host 0.0.0.0 --port 8000
```

## 项目结构

```
kernel-graph/
├── backend/              # Python后端
│   ├── kerg/            # 核心包
│   │   ├── parser/      # 解析器
│   │   ├── graph/       # 图构建引擎
│   │   ├── analysis/    # 分析引擎
│   │   ├── generator/  # 代码生成器
│   │   └── cli/        # 命令行接口
│   └── requirements.txt
├── frontend/            # React前端 (待开发)
└── README.md
```

## 技术栈

- **后端**: Python 3.10+, Clang/LLVM, NetworkX, FastAPI
- **前端**: React 18+, React Flow, Material-UI

## 开发计划

详见 [PRD.md](PRD.md)

## 许可证

MIT License