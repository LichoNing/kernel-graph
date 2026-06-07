# KernelGraph 使用指南

## 基本用法

### 1. 解析CUDA Kernel

**使用命令行：**
```bash
python -m kerg parse --input examples/sample_kernels.cu --output parsed.json
```

**使用Python API：**
```python
from kerg.parser.cuda_parser import CudaParser

parser = CudaParser()
with open('your_kernel.cu', 'r') as f:
    source = f.read()

kernel = parser.parse(source)
print(f"Kernel: {kernel.name}")
print(f"Parameters: {[p.name for p in kernel.parameters]}")
```

### 2. 生成计算图

**使用命令行：**
```bash
python -m kerg visualize --input examples/sample_kernels.cu --output graph.json
```

**使用Python API：**
```python
from kerg.parser.cuda_parser import CudaParser
from kerg.graph.graph_builder import GraphBuilder

# 解析
parser = CudaParser()
kernel = parser.parse(source_code)

# 构建图
builder = GraphBuilder()
graph = builder.build(kernel)

# 导出为JSON
json_output = graph.to_json()

# 保存到文件
with open('graph.json', 'w') as f:
    f.write(json_output)
```

### 3. 生成LLM Prompt

**使用命令行：**
```bash
# 生成简短prompt
python -m kerg prompt --input examples/sample_kernels.cu --output brief_prompt.txt --style brief

# 生成详细prompt
python -m kerg prompt --input examples/sample_kernels.cu --output detailed_prompt.txt --style detailed
```

**输出示例：**
```
分析以下CUDA kernel的计算图：

Kernel: vectorAdd

参数:
  float *a
  float *b
  float *c
  int n

计算图结构:
  节点总数: 5
  边总数: 4
  操作数: 1
  内存访问数: 3
  循环数: 0
  估算FLOPs: 1.0

节点详情:
  [op_0_idx] operation - op_type=assign
  ...

依赖关系:
  param_a → mem_0_a
  ...

请分析此kernel的性能特点并提供优化建议。
```

## 数据结构

### 核心类

- `GraphNode` - 图节点
- `GraphEdge` - 图边
- `KernelGraph` - 内核计算图

### 节点类型

| 类型 | 描述 |
|------|------|
| `operation` | 计算操作 |
| `memory` | 内存访问 |
| `variable` | 变量/寄存器 |
| `control` | 控制流 |
| `function` | 函数调用 |

### 边类型

| 类型 | 描述 |
|------|------|
| `data` | 数据依赖 |
| `control` | 控制依赖 |
| `memory` | 内存依赖 |

## 示例Kernel代码

参考 `examples/sample_kernels.cu`，包含以下示例：

1. **vectorAdd** - 向量加法
2. **matrixMul** - 矩阵乘法
3. **relu** - ReLU激活函数
4. **softmax** - Softmax函数

## 运行测试

```bash
# 进入项目根目录
cd kernel-graph

# 运行基础测试
python test_basic.py
```

## 开发指南

### 项目结构
```
kernel-graph/
├── backend/
│   └── kerg/
│       ├── parser/        # 解析器模块
│       ├── graph/         # 图构建引擎
│       ├── cli/           # 命令行接口
│       └── __main__.py    # 主入口
├── examples/              # 示例代码
└── test_basic.py          # 测试脚本
```

### 添加新的解析器

1. 在 `kerg/parser/` 中创建新的解析器类
2. 继承基础解析器接口
3. 实现 `parse()` 方法

### 添加新的图节点类型

1. 在 `kerg/graph/data_structures.py` 中定义新的节点类
2. 继承 `GraphNode` 基类
3. 在 `GraphBuilder` 中实现节点的构建逻辑
