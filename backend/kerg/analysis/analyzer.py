"""
性能分析引擎

分析CUDA kernel计算图的性能特征：
- 计算复杂度分析
- 内存访问模式分析
- 并行度评估
- 瓶颈识别
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

from ..graph.data_structures import KernelGraph, GraphNode, NodeType


class BottleneckType(Enum):
    """瓶颈类型"""
    COMPUTE = "compute"           # 计算瓶颈
    MEMORY = "memory"             # 内存瓶颈
    BANDWIDTH = "bandwidth"       # 带宽瓶颈
    LATENCY = "latency"           # 延迟瓶颈
    OCCUPANCY = "occupancy"       # 占用率瓶颈


@dataclass
class Bottleneck:
    """性能瓶颈"""
    type: BottleneckType
    description: str
    severity: float  # 0-1，严重程度
    location: str    # 位置描述
    suggestion: str  # 优化建议


@dataclass
class AnalysisResult:
    """分析结果"""
    kernel_name: str
    total_nodes: int
    total_edges: int
    
    # 计算指标
    compute_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 内存指标
    memory_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 并行指标
    parallelism_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 瓶颈列表
    bottlenecks: List[Bottleneck] = field(default_factory=list)
    
    # 优化建议
    optimizations: List[Dict[str, str]] = field(default_factory=list)
    
    # 综合评分
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'kernel_name': self.kernel_name,
            'total_nodes': self.total_nodes,
            'total_edges': self.total_edges,
            'compute_metrics': self.compute_metrics,
            'memory_metrics': self.memory_metrics,
            'parallelism_metrics': self.parallelism_metrics,
            'bottlenecks': [
                {
                    'type': b.type.value,
                    'description': b.description,
                    'severity': b.severity,
                    'location': b.location,
                    'suggestion': b.suggestion
                }
                for b in self.bottlenecks
            ],
            'optimizations': self.optimizations,
            'overall_score': self.overall_score
        }


class PerformanceAnalyzer:
    """
    性能分析器
    
    对KernelGraph进行全面的性能分析
    """
    
    def __init__(self):
        self.analysis_rules = [
            self._analyze_compute_intensity,
            self._analyze_memory_access_pattern,
            self._analyze_parallelism,
            self._analyze_occupancy,
            self._analyze_loop_efficiency,
        ]
    
    def analyze(self, graph: KernelGraph) -> AnalysisResult:
        """
        分析计算图
        
        Args:
            graph: 要分析的图
            
        Returns:
            分析结果
        """
        result = AnalysisResult(
            kernel_name=graph.name,
            total_nodes=len(graph.nodes),
            total_edges=len(graph.edges)
        )
        
        # 执行所有分析规则
        for rule in self.analysis_rules:
            rule(graph, result)
        
        # 生成优化建议
        self._generate_optimizations(result)
        
        # 计算综合评分
        result.overall_score = self._calculate_overall_score(result)
        
        return result
    
    def _analyze_compute_intensity(self, graph: KernelGraph, result: AnalysisResult):
        """分析计算强度"""
        # 计算操作数
        compute_ops = sum(1 for n in graph.nodes if n.type == NodeType.OPERATION)
        
        # 内存访问数
        memory_ops = sum(1 for n in graph.nodes if n.type == NodeType.MEMORY)
        
        # 估算FLOPs
        estimated_flops = graph.metadata.estimated_flops
        
        # 计算强度 = FLOPs / 内存访问字节数
        memory_bytes = memory_ops * 4  # 假设float32
        compute_intensity = estimated_flops / max(memory_bytes, 1)
        
        result.compute_metrics = {
            'total_operations': compute_ops,
            'estimated_flops': estimated_flops,
            'compute_intensity': compute_intensity,
            'arithmetic_intensity': compute_ops / max(len(graph.nodes), 1)
        }
        
        # 检查计算瓶颈
        if compute_intensity < 1.0:
            result.bottlenecks.append(Bottleneck(
                type=BottleneckType.MEMORY,
                description="计算强度低，可能受限于内存带宽",
                severity=0.7,
                location="全局内存访问",
                suggestion="考虑使用共享内存缓存数据，或增加每个线程的计算量"
            ))
    
    def _analyze_memory_access_pattern(self, graph: KernelGraph, result: AnalysisResult):
        """分析内存访问模式"""
        memory_nodes = [n for n in graph.nodes if n.type == NodeType.MEMORY]
        
        # 统计各种地址空间的访问
        address_space_counts = {}
        coalesced_count = 0
        strided_count = 0
        
        for node in memory_nodes:
            space = node.properties.get('address_space', 'global')
            address_space_counts[space] = address_space_counts.get(space, 0) + 1
            
            # 简单的合并访问检测
            layout = node.properties.get('layout', '')
            if 'contiguous' in layout or 'linear' in layout:
                coalesced_count += 1
            elif 'strided' in layout:
                strided_count += 1
        
        result.memory_metrics = {
            'total_memory_ops': len(memory_nodes),
            'address_space_distribution': address_space_counts,
            'coalesced_accesses': coalesced_count,
            'strided_accesses': strided_count,
            'coalescing_ratio': coalesced_count / max(len(memory_nodes), 1)
        }
        
        # 检查内存瓶颈
        global_ratio = address_space_counts.get('global', 0) / max(len(memory_nodes), 1)
        if global_ratio > 0.5:
            result.bottlenecks.append(Bottleneck(
                type=BottleneckType.BANDWIDTH,
                description="全局内存访问比例过高",
                severity=0.6,
                location="全局内存",
                suggestion="增加共享内存使用，实现数据复用"
            ))
        
        if strided_count > coalesced_count:
            result.bottlenecks.append(Bottleneck(
                type=BottleneckType.MEMORY,
                description="存在非合并内存访问",
                severity=0.8,
                location="内存访问模式",
                suggestion="优化内存访问模式，确保线程访问连续地址"
            ))
    
    def _analyze_parallelism(self, graph: KernelGraph, result: AnalysisResult):
        """分析并行度"""
        # 计算独立操作数
        independent_ops = 0
        for node in graph.nodes:
            incoming = graph.get_incoming_edges(node.id)
            if len(incoming) <= 1:
                independent_ops += 1
        
        # 计算ILP (指令级并行)
        total_ops = max(len(graph.nodes), 1)
        ilp = independent_ops / total_ops
        
        result.parallelism_metrics = {
            'instruction_level_parallelism': ilp,
            'independent_operations': independent_ops,
            'total_operations': total_ops,
            'parallelism_ratio': ilp
        }
        
        # 检查并行瓶颈
        if ilp < 0.3:
            result.bottlenecks.append(Bottleneck(
                type=BottleneckType.COMPUTE,
                description="指令级并行度低",
                severity=0.5,
                location="计算依赖链",
                suggestion="尝试展开循环或重排指令以增加ILP"
            ))
    
    def _analyze_occupancy(self, graph: KernelGraph, result: AnalysisResult):
        """分析占用率"""
        # 估算寄存器使用
        variable_nodes = [n for n in graph.nodes if n.type == NodeType.VARIABLE]
        estimated_registers = len(variable_nodes)
        
        # 估算共享内存使用
        shared_nodes = [n for n in graph.nodes 
                       if n.type == NodeType.MEMORY and 
                       n.properties.get('address_space') == 'shared']
        estimated_shared_mem = len(shared_nodes) * 1024  # 假设每个1KB
        
        result.parallelism_metrics['estimated_registers'] = estimated_registers
        result.parallelism_metrics['estimated_shared_memory'] = estimated_shared_mem
        
        # 检查占用率瓶颈
        if estimated_registers > 64:
            result.bottlenecks.append(Bottleneck(
                type=BottleneckType.OCCUPANCY,
                description="寄存器使用可能限制占用率",
                severity=0.4,
                location="寄存器分配",
                suggestion="减少局部变量使用，或拆分kernel"
            ))
    
    def _analyze_loop_efficiency(self, graph: KernelGraph, result: AnalysisResult):
        """分析循环效率"""
        loop_count = graph.metadata.loop_count
        
        result.compute_metrics['loop_count'] = loop_count
        
        # 检查循环展开机会
        if loop_count > 0:
            result.optimizations.append({
                'type': 'loop_unroll',
                'description': '考虑循环展开以减少循环开销',
                'benefit': '减少分支指令，增加ILP',
                'effort': '低'
            })
    
    def _generate_optimizations(self, result: AnalysisResult):
        """生成优化建议"""
        # 基于瓶颈生成优化建议
        for bottleneck in result.bottlenecks:
            if bottleneck.type == BottleneckType.MEMORY:
                result.optimizations.append({
                    'type': 'memory_optimization',
                    'description': '优化内存访问模式',
                    'details': bottleneck.suggestion,
                    'priority': 'high' if bottleneck.severity > 0.7 else 'medium'
                })
            elif bottleneck.type == BottleneckType.COMPUTE:
                result.optimizations.append({
                    'type': 'compute_optimization',
                    'description': '提升计算效率',
                    'details': bottleneck.suggestion,
                    'priority': 'medium'
                })
        
        # 通用优化建议
        if result.memory_metrics.get('coalescing_ratio', 1.0) < 0.8:
            result.optimizations.append({
                'type': 'coalescing',
                'description': '确保内存访问合并',
                'details': '调整线程访问模式，使相邻线程访问相邻地址',
                'priority': 'high'
            })
        
        if 'shared' not in result.memory_metrics.get('address_space_distribution', {}):
            result.optimizations.append({
                'type': 'shared_memory',
                'description': '使用共享内存',
                'details': '将频繁访问的数据缓存到共享内存',
                'priority': 'medium'
            })
    
    def _calculate_overall_score(self, result: AnalysisResult) -> float:
        """计算综合评分"""
        score = 1.0
        
        # 根据瓶颈严重程度扣分
        for bottleneck in result.bottlenecks:
            score -= bottleneck.severity * 0.2
        
        # 根据优化建议数量扣分
        score -= len(result.optimizations) * 0.05
        
        return max(0.0, min(1.0, score))
    
    def generate_report(self, result: AnalysisResult, format: str = 'text') -> str:
        """
        生成分析报告
        
        Args:
            result: 分析结果
            format: 报告格式 (text/markdown/json)
            
        Returns:
            报告字符串
        """
        if format == 'json':
            import json
            return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        
        lines = []
        lines.append("=" * 70)
        lines.append(f"性能分析报告: {result.kernel_name}")
        lines.append("=" * 70)
        
        # 计算指标
        lines.append(f"\n[计算指标]")
        for key, value in result.compute_metrics.items():
            lines.append(f"  {key}: {value}")
        
        # 内存指标
        lines.append(f"\n[内存指标]")
        for key, value in result.memory_metrics.items():
            lines.append(f"  {key}: {value}")
        
        # 并行指标
        lines.append(f"\n[并行指标]")
        for key, value in result.parallelism_metrics.items():
            lines.append(f"  {key}: {value}")
        
        # 瓶颈
        if result.bottlenecks:
            lines.append(f"\n[性能瓶颈]")
            for i, bottleneck in enumerate(result.bottlenecks, 1):
                lines.append(f"  {i}. [{bottleneck.type.value.upper()}] {bottleneck.description}")
                lines.append(f"     严重程度: {bottleneck.severity:.2f}")
                lines.append(f"     位置: {bottleneck.location}")
                lines.append(f"     建议: {bottleneck.suggestion}")
        
        # 优化建议
        if result.optimizations:
            lines.append(f"\n[优化建议]")
            for i, opt in enumerate(result.optimizations, 1):
                lines.append(f"  {i}. [{opt['priority'].upper()}] {opt['description']}")
                lines.append(f"     {opt['details']}")
        
        # 综合评分
        lines.append(f"\n[综合评分] {result.overall_score:.2f}/1.00")
        
        lines.append("\n" + "=" * 70)
        
        return '\n'.join(lines)
