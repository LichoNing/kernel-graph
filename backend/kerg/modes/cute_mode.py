"""
CUTLASS cute 开发模式

支持NVIDIA CUTLASS cute库的kernel开发：
- cute::Tensor, cute::Layout, cute::TiledMMA等核心概念
- 层次化内存管理 (Global, Shared, Register)
- 协作式kernel设计 (CpAsync, TMA, WGMMA)
- 模板元编程模式
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseMode, BaseParser, BaseGraphBuilder, BaseCodeGenerator, BaseAnalyzer, BaseVisualizer


class CuteTensorType(Enum):
    """cute Tensor类型"""
    GLOBAL = "global"           # Global Memory Tensor
    SHARED = "shared"           # Shared Memory Tensor
    REGISTER = "register"       # Register Tensor
    FRAGMENT = "fragment"       # MMA Fragment


class CuteOperationType(Enum):
    """cute操作类型"""
    COPY = "copy"               # 内存拷贝 (g2s, s2r, r2s, s2g)
    MMA = "mma"                 # 矩阵乘累加
    EPILOGUE = "epilogue"       # 尾处理
    REDUCE = "reduce"           # 归约
    BROADCAST = "broadcast"     # 广播
    TRANSFORM = "transform"     # 变换


class CuteMemoryLevel(Enum):
    """cute内存层次"""
    GLOBAL = "global"
    SHARED = "shared"
    REGISTER = "register"


@dataclass
class CuteTensor:
    """cute Tensor表示"""
    name: str
    tensor_type: CuteTensorType
    layout: str                      # 布局描述，如"(M,N):(1,M)"
    dtype: str = "float"
    memory_level: CuteMemoryLevel = CuteMemoryLevel.GLOBAL
    shape: Tuple[int, ...] = field(default_factory=tuple)
    stride: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class CuteTiledMMA:
    """cute TiledMMA配置"""
    name: str
    mma_op: str                     # MMA操作，如"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
    warp_layout: Tuple[int, int] = (1, 1)
    tile_shape: Tuple[int, int, int] = (16, 8, 16)


@dataclass
class CuteCopyAtom:
    """cute CopyAtom配置"""
    name: str
    copy_op: str                    # 拷贝操作，如"cp.async.cg"
    src_layout: str = ""
    dst_layout: str = ""


@dataclass
class CuteOperation:
    """cute操作"""
    op_type: CuteOperationType
    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    tiled_mma: Optional[CuteTiledMMA] = None
    copy_atom: Optional[CuteCopyAtom] = None
    line: int = 0


@dataclass
class CuteKernel:
    """cute Kernel表示"""
    name: str
    params: List[Dict[str, str]] = field(default_factory=list)
    tensors: List[CuteTensor] = field(default_factory=list)
    tiled_mmas: List[CuteTiledMMA] = field(default_factory=list)
    copy_atoms: List[CuteCopyAtom] = field(default_factory=list)
    operations: List[CuteOperation] = field(default_factory=list)
    pipeline_stages: int = 1
    raw_source: str = ""


class CuteParser(BaseParser):
    """
    cute解析器
    
    解析cute风格的CUDA kernel，识别：
    - Tensor声明和Layout
    - TiledMMA配置
    - CopyAtom配置
    - 层次化操作 (g2s, s2r, mma, epilogue)
    """
    
    def __init__(self):
        self.source_code = ""
        self.current_line = 0
    
    def parse(self, source: str) -> CuteKernel:
        """解析cute kernel源码"""
        self.source_code = source
        lines = source.split('\n')
        
        # 提取kernel名称
        kernel_name = self._extract_kernel_name(source)
        kernel = CuteKernel(name=kernel_name, raw_source=source)
        
        # 解析参数
        kernel.params = self._extract_params(source)
        
        # 解析Tensor声明
        kernel.tensors = self._extract_tensors(lines)
        
        # 解析TiledMMA
        kernel.tiled_mmas = self._extract_tiled_mmas(lines)
        
        # 解析CopyAtom
        kernel.copy_atoms = self._extract_copy_atoms(lines)
        
        # 解析操作
        kernel.operations = self._extract_operations(lines)
        
        # 解析pipeline stages
        kernel.pipeline_stages = self._extract_pipeline_stages(lines)
        
        return kernel
    
    def tokenize(self, source: str) -> List[str]:
        """简单的词法分析"""
        # 移除注释
        source = re.sub(r'//.*$', '', source, flags=re.MULTILINE)
        source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
        
        # 分词
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[{}()<>\[\],;:=+\-*/&|]', source)
        return tokens
    
    def _extract_kernel_name(self, source: str) -> str:
        """提取kernel函数名"""
        match = re.search(r'__global__\s+void\s+(\w+)', source)
        return match.group(1) if match else "unknown_kernel"
    
    def _extract_params(self, source: str) -> List[Dict[str, str]]:
        """提取kernel参数"""
        params = []
        match = re.search(r'__global__\s+void\s+\w+\s*\((.*?)\)', source, re.DOTALL)
        if match:
            param_str = match.group(1)
            for param in param_str.split(','):
                param = param.strip()
                if param:
                    parts = param.split()
                    if len(parts) >= 2:
                        params.append({
                            'type': ' '.join(parts[:-1]),
                            'name': parts[-1].replace('*', '').strip()
                        })
        return params
    
    def _extract_tensors(self, lines: List[str]) -> List[CuteTensor]:
        """提取Tensor声明"""
        tensors = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 匹配Tensor声明: Tensor<ViewEngine, Layout> tensor_name;
            # 或 make_tensor(tensor_name, layout)
            tensor_match = re.search(
                r'Tensor<\s*([^,]+)\s*,\s*([^>]+)\s*>\s+(\w+)',
                line
            )
            if tensor_match:
                engine = tensor_match.group(1).strip()
                layout = tensor_match.group(2).strip()
                name = tensor_match.group(3)
                
                # 判断内存层次
                memory_level = CuteMemoryLevel.GLOBAL
                tensor_type = CuteTensorType.GLOBAL
                
                if 'Smem' in engine or 'Shared' in engine:
                    memory_level = CuteMemoryLevel.SHARED
                    tensor_type = CuteTensorType.SHARED
                elif 'Rmem' in engine or 'Register' in engine:
                    memory_level = CuteMemoryLevel.REGISTER
                    tensor_type = CuteTensorType.REGISTER
                
                tensors.append(CuteTensor(
                    name=name,
                    tensor_type=tensor_type,
                    layout=layout,
                    memory_level=memory_level,
                    line=i
                ))
            
            # 匹配make_tensor
            make_match = re.search(
                r'auto\s+(\w+)\s*=\s*make_tensor\s*<\s*([^>]+)\s*>\s*\(',
                line
            )
            if make_match:
                name = make_match.group(1)
                dtype = make_match.group(2).strip()
                
                tensors.append(CuteTensor(
                    name=name,
                    tensor_type=CuteTensorType.GLOBAL,
                    layout="auto",
                    dtype=dtype,
                    line=i
                ))
        
        return tensors
    
    def _extract_tiled_mmas(self, lines: List[str]) -> List[CuteTiledMMA]:
        """提取TiledMMA配置"""
        mmas = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 匹配TiledMMA声明
            mma_match = re.search(
                r'TiledMMA\s*<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^>]+)\s*>\s+(\w+)',
                line
            )
            if mma_match:
                mma_op = mma_match.group(1).strip()
                warp_layout_str = mma_match.group(2).strip()
                tile_shape_str = mma_match.group(3).strip()
                name = mma_match.group(4)
                
                # 解析warp layout
                warp_layout = self._parse_tuple(warp_layout_str, default=(1, 1))
                tile_shape = self._parse_tuple(tile_shape_str, default=(16, 8, 16))
                
                mmas.append(CuteTiledMMA(
                    name=name,
                    mma_op=mma_op,
                    warp_layout=warp_layout,
                    tile_shape=tile_shape
                ))
        
        return mmas
    
    def _extract_copy_atoms(self, lines: List[str]) -> List[CuteCopyAtom]:
        """提取CopyAtom配置"""
        atoms = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 匹配CopyAtom
            atom_match = re.search(
                r'Copy_Atom\s*<\s*([^,]+)\s*,\s*([^>]+)\s*>\s+(\w+)',
                line
            )
            if atom_match:
                copy_op = atom_match.group(1).strip()
                dtype = atom_match.group(2).strip()
                name = atom_match.group(3)
                
                atoms.append(CuteCopyAtom(
                    name=name,
                    copy_op=copy_op
                ))
        
        return atoms
    
    def _extract_operations(self, lines: List[str]) -> List[CuteOperation]:
        """提取cute操作"""
        operations = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 识别各种cute操作模式
            
            # 1. 拷贝操作: copy(atom, src, dst)
            copy_match = re.search(r'copy\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)', line)
            if copy_match:
                atom_name = copy_match.group(1)
                src = copy_match.group(2)
                dst = copy_match.group(3)
                
                # 判断拷贝方向
                op_type = CuteOperationType.COPY
                
                operations.append(CuteOperation(
                    op_type=op_type,
                    name=f"copy_{len(operations)}",
                    inputs=[src],
                    outputs=[dst],
                    copy_atom=CuteCopyAtom(name=atom_name, copy_op="copy"),
                    line=i
                ))
            
            # 2. MMA操作: gemm(tiled_mma, a, b, c)
            mma_match = re.search(r'gemm\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)', line)
            if mma_match:
                mma_name = mma_match.group(1)
                a = mma_match.group(2)
                b = mma_match.group(3)
                c = mma_match.group(4)
                
                operations.append(CuteOperation(
                    op_type=CuteOperationType.MMA,
                    name=f"mma_{len(operations)}",
                    inputs=[a, b],
                    outputs=[c],
                    line=i
                ))
            
            # 3. 尾处理: epilogue/scale
            if any(op in line for op in ['epilogue', 'scale', 'bias']):
                operations.append(CuteOperation(
                    op_type=CuteOperationType.EPILOGUE,
                    name=f"epilogue_{len(operations)}",
                    line=i
                ))
            
            # 4. 归约
            if 'reduce' in line.lower():
                operations.append(CuteOperation(
                    op_type=CuteOperationType.REDUCE,
                    name=f"reduce_{len(operations)}",
                    line=i
                ))
        
        return operations
    
    def _extract_pipeline_stages(self, lines: List[str]) -> int:
        """提取pipeline stages数量"""
        for line in lines:
            match = re.search(r'Pipeline\s*<\s*(\d+)\s*>', line)
            if match:
                return int(match.group(1))
        return 1
    
    def _parse_tuple(self, s: str, default: Tuple[int, ...] = ()) -> Tuple[int, ...]:
        """解析tuple字符串"""
        match = re.search(r'\((.*?)\)', s)
        if match:
            try:
                return tuple(int(x.strip()) for x in match.group(1).split(','))
            except ValueError:
                pass
        return default


class CuteGraphBuilder(BaseGraphBuilder):
    """
    cute图构建器
    
    将cute kernel转换为层次化的计算图：
    - Global Memory层
    - Shared Memory层  
    - Register/Fragment层
    - MMA Compute层
    """
    
    def __init__(self):
        from ..graph.data_structures import KernelGraph, GraphNode, GraphEdge, NodeType, EdgeType
        self.KernelGraph = KernelGraph
        self.GraphNode = GraphNode
        self.GraphEdge = GraphEdge
        self.NodeType = NodeType
        self.EdgeType = EdgeType
    
    def build(self, parsed_result: CuteKernel):
        """构建cute计算图"""
        from ..graph.data_structures import (
            KernelGraph, OperationNode, MemoryNode, VariableNode,
            NodeType, EdgeType, MemoryAccessType, AddressSpace
        )
        
        graph = KernelGraph(name=parsed_result.name)
        node_map = {}  # 名称 -> 节点ID映射
        
        # 1. 添加参数节点
        for param in parsed_result.params:
            node = VariableNode(
                label=f"param_{param['name']}",
                name=param['name'],
                var_type=param['type'],
                scope="parameter"
            )
            graph.add_node(node)
            node_map[param['name']] = node.id
        
        # 2. 添加Tensor节点（按内存层次分组）
        for tensor in parsed_result.tensors:
            # 根据内存层次选择地址空间
            addr_space_map = {
                CuteMemoryLevel.GLOBAL: AddressSpace.GLOBAL,
                CuteMemoryLevel.SHARED: AddressSpace.SHARED,
                CuteMemoryLevel.REGISTER: AddressSpace.REGISTER
            }
            addr_space = addr_space_map.get(tensor.memory_level, AddressSpace.GLOBAL)
            
            node = MemoryNode(
                label=f"tensor_{tensor.name}",
                access_type=MemoryAccessType.LOAD,
                address_space=addr_space,
                properties={
                    'layout': tensor.layout,
                    'dtype': tensor.dtype,
                    'memory_level': tensor.memory_level.value
                }
            )
            graph.add_node(node)
            node_map[tensor.name] = node.id
        
        # 3. 添加TiledMMA节点
        for mma in parsed_result.tiled_mmas:
            node = OperationNode(
                label=f"mma_{mma.name}",
                op_type="tiled_mma",
                properties={
                    'mma_op': mma.mma_op,
                    'warp_layout': mma.warp_layout,
                    'tile_shape': mma.tile_shape
                }
            )
            graph.add_node(node)
            node_map[mma.name] = node.id
        
        # 4. 添加操作节点和边
        for op in parsed_result.operations:
            if op.op_type == CuteOperationType.COPY:
                # 拷贝操作：创建数据流边
                for src in op.inputs:
                    if src in node_map:
                        for dst in op.outputs:
                            if dst in node_map:
                                edge = GraphEdge(
                                    source=node_map[src],
                                    target=node_map[dst],
                                    type=EdgeType.DATA,
                                    properties={'operation': 'copy'}
                                )
                                try:
                                    graph.add_edge(edge)
                                except ValueError:
                                    pass
            
            elif op.op_type == CuteOperationType.MMA:
                # MMA操作：创建计算节点
                node = OperationNode(
                    label=f"compute_{op.name}",
                    op_type="mma",
                    operands=op.inputs,
                    properties={'tiled_mma': op.tiled_mma.name if op.tiled_mma else None}
                )
                graph.add_node(node)
                
                # 连接输入
                for inp in op.inputs:
                    if inp in node_map:
                        edge = GraphEdge(
                            source=node_map[inp],
                            target=node.id,
                            type=EdgeType.DATA
                        )
                        try:
                            graph.add_edge(edge)
                        except ValueError:
                            pass
                
                # 连接输出
                for out in op.outputs:
                    if out in node_map:
                        edge = GraphEdge(
                            source=node.id,
                            target=node_map[out],
                            type=EdgeType.DATA
                        )
                        try:
                            graph.add_edge(edge)
                        except ValueError:
                            pass
        
        # 更新元数据
        graph.update_metadata()
        
        return graph


class CuteCodeGenerator(BaseCodeGenerator):
    """cute代码生成器"""
    
    def generate(self, graph) -> str:
        # TODO: 实现从cute图生成代码
        return "// cute code generation not yet implemented"


class CuteAnalyzer(BaseAnalyzer):
    """cute分析器"""
    
    def analyze(self, graph) -> Dict[str, Any]:
        """分析cute kernel性能特征"""
        analysis = {
            'mode': 'cute',
            'total_operations': graph.metadata.total_operations,
            'memory_access_count': graph.metadata.memory_access_count,
            'loop_count': graph.metadata.loop_count,
            'estimated_flops': graph.metadata.estimated_flops,
            'memory_hierarchy': self._analyze_memory_hierarchy(graph),
            'mma_utilization': self._analyze_mma_utilization(graph),
            'pipeline_efficiency': self._analyze_pipeline(graph)
        }
        return analysis
    
    def _analyze_memory_hierarchy(self, graph) -> Dict[str, int]:
        """分析内存层次使用情况"""
        hierarchy = {'global': 0, 'shared': 0, 'register': 0}
        for node in graph.nodes:
            if node.type.value == 'memory':
                level = node.properties.get('memory_level', 'global')
                hierarchy[level] = hierarchy.get(level, 0) + 1
        return hierarchy
    
    def _analyze_mma_utilization(self, graph) -> float:
        """分析MMA利用率"""
        mma_count = sum(1 for n in graph.nodes if n.properties.get('op_type') == 'mma')
        total_ops = max(graph.metadata.total_operations, 1)
        return mma_count / total_ops
    
    def _analyze_pipeline(self, graph) -> Dict[str, Any]:
        """分析pipeline效率"""
        return {
            'has_pipeline': any('pipeline' in str(n.properties) for n in graph.nodes),
            'estimated_occupancy': 0.75  # 估算值
        }


class CuteVisualizer(BaseVisualizer):
    """cute可视化器"""
    
    def visualize(self, graph, format: str = 'text') -> str:
        if format == 'text':
            return self._visualize_text(graph)
        elif format == 'json':
            return graph.to_json()
        elif format == 'mermaid':
            return self._visualize_mermaid(graph)
        else:
            return f"Unsupported format: {format}"
    
    def _visualize_text(self, graph) -> str:
        """文本可视化"""
        lines = []
        lines.append(f"cute Kernel: {graph.name}")
        lines.append(f"\n内存层次结构:")
        
        # 按内存层次分组显示
        levels = {'global': [], 'shared': [], 'register': []}
        for node in graph.nodes:
            if node.type.value == 'memory':
                level = node.properties.get('memory_level', 'global')
                levels[level].append(node)
        
        for level, nodes in levels.items():
            if nodes:
                lines.append(f"  [{level.upper()}]")
                for node in nodes:
                    lines.append(f"    {node.label}: {node.properties.get('layout', 'N/A')}")
        
        lines.append(f"\n操作序列:")
        for i, node in enumerate(graph.nodes):
            if node.type.value == 'operation':
                lines.append(f"  [{i}] {node.label}: {node.properties.get('op_type', 'N/A')}")
        
        lines.append(f"\n数据流:")
        for edge in graph.edges:
            src = graph.get_node(edge.source)
            dst = graph.get_node(edge.target)
            if src and dst:
                lines.append(f"  {src.label} --> {dst.label}")
        
        return '\n'.join(lines)
    
    def _visualize_mermaid(self, graph) -> str:
        """生成Mermaid图"""
        lines = ["```mermaid", "flowchart TD"]
        
        # 定义节点
        for node in graph.nodes:
            shape = "([" if node.type.value == 'operation' else "["
            end_shape = "])" if node.type.value == 'operation' else "]"
            lines.append(f"    {node.id}{shape}{node.label}{end_shape}")
        
        # 定义边
        for edge in graph.edges:
            lines.append(f"    {edge.source} --> {edge.target}")
        
        lines.append("```")
        return '\n'.join(lines)


class CuteMode(BaseMode):
    """
    CUTLASS cute 开发模式
    
    支持NVIDIA CUTLASS cute库的kernel开发
    """
    
    name = "cute"
    description = "NVIDIA CUTLASS cute开发模式"
    file_extensions = ['.cu', '.cuh', '.cpp']
    
    def _create_parser(self) -> BaseParser:
        return CuteParser()
    
    def _create_graph_builder(self) -> BaseGraphBuilder:
        return CuteGraphBuilder()
    
    def _create_code_generator(self) -> BaseCodeGenerator:
        return CuteCodeGenerator()
    
    def _create_analyzer(self) -> BaseAnalyzer:
        return CuteAnalyzer()
    
    def _create_visualizer(self) -> BaseVisualizer:
        return CuteVisualizer()
