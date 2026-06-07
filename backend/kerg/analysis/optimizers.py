"""
优化器模块

提供具体的优化策略：
- 内存优化
- 计算优化
- 并行优化
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..graph.data_structures import KernelGraph, GraphNode, GraphEdge, NodeType


@dataclass
class Optimization:
    """优化方案"""
    name: str
    description: str
    target_nodes: List[str]  # 目标节点ID列表
    transformation: str      # 变换描述
    expected_speedup: float  # 预期加速比
    risk_level: str         # 风险等级 (low/medium/high)


class Optimizer(ABC):
    """优化器基类"""
    
    @abstractmethod
    def optimize(self, graph: KernelGraph) -> List[Optimization]:
        """执行优化"""
        pass
    
    @abstractmethod
    def can_apply(self, graph: KernelGraph) -> bool:
        """检查是否可以应用此优化"""
        pass


class MemoryOptimizer(Optimizer):
    """
    内存优化器
    
    优化策略：
    - 共享内存缓存
    - 内存访问合并
    - 数据预取
    -  bank conflict消除
    """
    
    def can_apply(self, graph: KernelGraph) -> bool:
        """检查是否有内存优化空间"""
        memory_nodes = [n for n in graph.nodes if n.type == NodeType.MEMORY]
        return len(memory_nodes) > 0
    
    def optimize(self, graph: KernelGraph) -> List[Optimization]:
        """执行内存优化"""
        optimizations = []
        
        # 1. 检查共享内存使用机会
        shared_opt = self._suggest_shared_memory(graph)
        if shared_opt:
            optimizations.append(shared_opt)
        
        # 2. 检查内存合并
        coalesce_opt = self._suggest_coalescing(graph)
        if coalesce_opt:
            optimizations.append(coalesce_opt)
        
        # 3. 检查数据预取
        prefetch_opt = self._suggest_prefetch(graph)
        if prefetch_opt:
            optimizations.append(prefetch_opt)
        
        return optimizations
    
    def _suggest_shared_memory(self, graph: KernelGraph) -> Optional[Optimization]:
        """建议使用共享内存"""
        # 查找频繁访问的全局内存
        global_nodes = [
            n for n in graph.nodes 
            if n.type == NodeType.MEMORY 
            and n.properties.get('address_space') == 'global'
        ]
        
        if len(global_nodes) >= 2:
            return Optimization(
                name="shared_memory_caching",
                description="将频繁访问的全局内存数据缓存到共享内存",
                target_nodes=[n.id for n in global_nodes[:2]],
                transformation="添加共享内存Tensor，使用cp.async进行异步拷贝",
                expected_speedup=2.0,
                risk_level="low"
            )
        return None
    
    def _suggest_coalescing(self, graph: KernelGraph) -> Optional[Optimization]:
        """建议内存访问合并"""
        # 查找可能的非合并访问
        strided_nodes = [
            n for n in graph.nodes
            if n.type == NodeType.MEMORY
            and 'strided' in str(n.properties.get('layout', ''))
        ]
        
        if strided_nodes:
            return Optimization(
                name="memory_coalescing",
                description="优化内存访问模式以实现合并访问",
                target_nodes=[n.id for n in strided_nodes],
                transformation="调整线程索引计算，确保相邻线程访问相邻地址",
                expected_speedup=1.5,
                risk_level="low"
            )
        return None
    
    def _suggest_prefetch(self, graph: KernelGraph) -> Optional[Optimization]:
        """建议数据预取"""
        # 查找循环中的内存访问
        loop_nodes = [
            n for n in graph.nodes
            if n.type == NodeType.CONTROL
            and n.properties.get('control_type') == 'for'
        ]
        
        if loop_nodes:
            memory_in_loop = []
            for loop in loop_nodes:
                # 查找循环内的内存节点
                for edge in graph.get_outgoing_edges(loop.id):
                    node = graph.get_node(edge.target)
                    if node and node.type == NodeType.MEMORY:
                        memory_in_loop.append(node)
            
            if memory_in_loop:
                return Optimization(
                    name="data_prefetch",
                    description="在循环中预取下一次迭代的数据",
                    target_nodes=[n.id for n in memory_in_loop[:2]],
                    transformation="使用cp.async添加预取指令",
                    expected_speedup=1.3,
                    risk_level="medium"
                )
        return None


class ComputeOptimizer(Optimizer):
    """
    计算优化器
    
    优化策略：
    - 循环展开
    - 指令重排
    - 强度削弱
    - 向量化
    """
    
    def can_apply(self, graph: KernelGraph) -> bool:
        """检查是否有计算优化空间"""
        op_nodes = [n for n in graph.nodes if n.type == NodeType.OPERATION]
        return len(op_nodes) > 0
    
    def optimize(self, graph: KernelGraph) -> List[Optimization]:
        """执行计算优化"""
        optimizations = []
        
        # 1. 循环展开
        unroll_opt = self._suggest_loop_unroll(graph)
        if unroll_opt:
            optimizations.append(unroll_opt)
        
        # 2. 强度削弱
        strength_opt = self._suggest_strength_reduction(graph)
        if strength_opt:
            optimizations.append(strength_opt)
        
        # 3. 向量化
        vector_opt = self._suggest_vectorization(graph)
        if vector_opt:
            optimizations.append(vector_opt)
        
        return optimizations
    
    def _suggest_loop_unroll(self, graph: KernelGraph) -> Optional[Optimization]:
        """建议循环展开"""
        loop_nodes = [
            n for n in graph.nodes
            if n.type == NodeType.CONTROL
            and n.properties.get('control_type') == 'for'
        ]
        
        if loop_nodes:
            return Optimization(
                name="loop_unrolling",
                description="展开循环以减少循环开销",
                target_nodes=[n.id for n in loop_nodes],
                transformation="使用#pragma unroll或手动展开循环",
                expected_speedup=1.2,
                risk_level="low"
            )
        return None
    
    def _suggest_strength_reduction(self, graph: KernelGraph) -> Optional[Optimization]:
        """建议强度削弱"""
        # 查找可以用位运算替代的乘除法
        div_nodes = [
            n for n in graph.nodes
            if n.type == NodeType.OPERATION
            and n.properties.get('op_type') in ['/', '%']
        ]
        
        if div_nodes:
            return Optimization(
                name="strength_reduction",
                description="用位运算替代乘除法",
                target_nodes=[n.id for n in div_nodes],
                transformation="将除以2的幂次转换为右移，取模转换为位与",
                expected_speedup=1.1,
                risk_level="low"
            )
        return None
    
    def _suggest_vectorization(self, graph: KernelGraph) -> Optional[Optimization]:
        """建议向量化"""
        # 查找连续的相同操作
        op_nodes = [n for n in graph.nodes if n.type == NodeType.OPERATION]
        
        if len(op_nodes) >= 4:
            return Optimization(
                name="vectorization",
                description="使用向量指令处理多个数据",
                target_nodes=[n.id for n in op_nodes[:4]],
                transformation="使用float4或half2进行向量化加载和计算",
                expected_speedup=1.5,
                risk_level="medium"
            )
        return None


class ParallelOptimizer(Optimizer):
    """
    并行优化器
    
    优化策略：
    - 增加occupancy
    - 减少分支发散
    - 优化warp效率
    """
    
    def can_apply(self, graph: KernelGraph) -> bool:
        return True
    
    def optimize(self, graph: KernelGraph) -> List[Optimization]:
        """执行并行优化"""
        optimizations = []
        
        # 检查分支发散
        branch_opt = self._suggest_branch_optimization(graph)
        if branch_opt:
            optimizations.append(branch_opt)
        
        return optimizations
    
    def _suggest_branch_optimization(self, graph: KernelGraph) -> Optional[Optimization]:
        """建议分支优化"""
        # 查找条件分支
        if_nodes = [
            n for n in graph.nodes
            if n.type == NodeType.CONTROL
            and n.properties.get('control_type') == 'if'
        ]
        
        if if_nodes:
            return Optimization(
                name="branch_optimization",
                description="减少warp内的分支发散",
                target_nodes=[n.id for n in if_nodes],
                transformation="重新组织数据或调整线程分配，确保同一warp内线程执行相同分支",
                expected_speedup=1.2,
                risk_level="medium"
            )
        return None
