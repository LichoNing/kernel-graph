"""
图构建引擎

将解析后的CUDA kernel转换为KernelGraph结构
"""

from typing import Dict, List, Optional, Set
from ..parser.cuda_parser import ParsedKernel, Operation, MemoryAccess, Variable
from .data_structures import (
    KernelGraph, GraphNode, GraphEdge, 
    NodeType, EdgeType, 
    OperationNode, MemoryNode, VariableNode, ControlNode,
    MemoryAccessType, AddressSpace, ControlType
)


class GraphBuilder:
    """
    图构建引擎
    
    将ParsedKernel转换为KernelGraph，构建节点和边的关系
    """
    
    def __init__(self):
        self.kernel_graph: Optional[KernelGraph] = None
        self.node_map: Dict[str, GraphNode] = {}  # 变量名 -> 节点映射
        self.execution_order: List[GraphNode] = []  # 执行顺序
    
    def build(self, parsed_kernel: ParsedKernel) -> KernelGraph:
        """
        构建计算图
        
        Args:
            parsed_kernel: 解析后的kernel对象
            
        Returns:
            KernelGraph对象
        """
        # 创建新的图
        self.kernel_graph = KernelGraph(name=parsed_kernel.name)
        
        # 清空映射
        self.node_map = {}
        self.execution_order = []
        
        # 添加参数节点
        self._add_parameter_nodes(parsed_kernel.parameters)
        
        # 添加局部变量节点
        self._add_variable_nodes(parsed_kernel.local_variables)
        
        # 添加操作节点和内存访问节点
        self._add_operation_nodes(parsed_kernel.operations)
        
        # 添加内存访问节点
        self._add_memory_nodes(parsed_kernel.memory_accesses)
        
        # 构建依赖关系
        self._build_dependencies(parsed_kernel)
        
        # 更新元数据
        self.kernel_graph.update_metadata()
        
        return self.kernel_graph
    
    def _add_parameter_nodes(self, parameters: List[Variable]):
        """添加参数节点"""
        for param in parameters:
            node = VariableNode(
                label=f"param_{param.name}",
                name=param.name,
                var_type=param.var_type,
                scope="parameter",
                is_array=param.is_array,
                qualifiers=param.qualifiers
            )
            self.kernel_graph.add_node(node)
            self.node_map[param.name] = node
    
    def _add_variable_nodes(self, variables: List[Variable]):
        """添加局部变量节点"""
        for var in variables:
            node = VariableNode(
                label=f"var_{var.name}",
                name=var.name,
                var_type=var.type if hasattr(var, 'type') else "int",
                scope="local",
                is_array=var.is_array
            )
            self.kernel_graph.add_node(node)
            self.node_map[var.name] = node
    
    def _add_operation_nodes(self, operations: List[Operation]):
        """添加操作节点"""
        for i, op in enumerate(operations):
            # 创建操作节点
            node = OperationNode(
                label=f"op_{i}_{op.result or 'expr'}",
                op_type=op.operator,
                operands=op.operands,
                result=op.result,
                latency_estimate=self._estimate_latency(op.operator)
            )
            
            self.kernel_graph.add_node(node)
            self.execution_order.append(node)
            
            # 记录结果变量
            if op.result:
                self.node_map[op.result] = node
    
    def _add_memory_nodes(self, memory_accesses: List[MemoryAccess]):
        """添加内存访问节点"""
        for i, mem in enumerate(memory_accesses):
            access_type = MemoryAccessType.LOAD if mem.access_type == "load" else MemoryAccessType.STORE
            address_space = self._get_address_space(mem.address_space)
            
            node = MemoryNode(
                label=f"mem_{i}_{mem.variable}",
                access_type=access_type,
                address_space=address_space,
                size=4  # 默认4字节
            )
            
            self.kernel_graph.add_node(node)
            
            # 如果变量还没有对应的节点，创建虚拟节点
            if mem.variable not in self.node_map:
                var_node = VariableNode(
                    label=f"virt_{mem.variable}",
                    name=mem.variable,
                    var_type="float",
                    scope="virtual"
                )
                self.kernel_graph.add_node(var_node)
                self.node_map[mem.variable] = var_node
    
    def _build_dependencies(self, parsed_kernel: ParsedKernel):
        """构建节点间的依赖关系"""
        # 建立数据流边
        for op in parsed_kernel.operations:
            if not op.result:
                continue
            
            # 操作节点
            op_node = self._find_node_by_label(f"op_.*_{op.result}")
            if not op_node:
                continue
            
            # 为每个操作数创建边
            for operand in op.operands:
                # 提取基本变量名
                var_name = self._extract_base_variable(operand)
                
                if var_name in self.node_map:
                    source_node = self.node_map[var_name]
                    edge = GraphEdge(
                        source=source_node.id,
                        target=op_node.id,
                        type=EdgeType.DATA,
                        properties={"operand": operand}
                    )
                    try:
                        self.kernel_graph.add_edge(edge)
                    except ValueError:
                        pass  # 边已存在或节点不存在
    
    def _find_node_by_label(self, pattern: str) -> Optional[GraphNode]:
        """根据标签模式查找节点"""
        import re
        for node in self.kernel_graph.nodes:
            if re.match(pattern, node.label):
                return node
        return None
    
    def _extract_base_variable(self, expr: str) -> str:
        """从表达式中提取基础变量名"""
        # 移除空格
        expr = expr.strip()
        
        # 移除索引表达式
        if '[' in expr:
            expr = expr[:expr.index('[')]
        
        # 移除算术运算符
        for op in ['+', '-', '*', '/', '%']:
            parts = expr.split(op)
            if len(parts) > 1:
                expr = parts[0].strip()
        
        return expr.strip()
    
    def _estimate_latency(self, operator: str) -> float:
        """估算操作延迟（时钟周期）"""
        latency_map = {
            'add': 4,
            'sub': 4,
            'mul': 16,
            'div': 64,
            'assign': 1,
            '+': 4,
            '-': 4,
            '*': 16,
            '/': 64,
            '%': 64
        }
        return latency_map.get(operator, 10)
    
    def _get_address_space(self, space: str) -> AddressSpace:
        """获取地址空间枚举"""
        space_map = {
            'global': AddressSpace.GLOBAL,
            'shared': AddressSpace.SHARED,
            'register': AddressSpace.REGISTER,
            'constant': AddressSpace.CONSTANT,
            'local': AddressSpace.LOCAL
        }
        return space_map.get(space, AddressSpace.GLOBAL)
    
    def get_execution_order(self) -> List[GraphNode]:
        """获取执行顺序（拓扑排序）"""
        return self.execution_order
    
    def visualize_text(self) -> str:
        """
        生成文本形式的可视化输出
        
        用于调试和生成LLM prompt
        """
        if not self.kernel_graph:
            return "No graph built yet"
        
        lines = []
        lines.append(f"Kernel: {self.kernel_graph.name}")
        lines.append(f"\n节点列表:")
        
        for i, node in enumerate(self.kernel_graph.nodes):
            props = ", ".join(f"{k}={v}" for k, v in node.properties.items())
            lines.append(f"  [{i+1}] {node.label} ({node.type.value}) - {props}")
        
        lines.append(f"\n依赖关系:")
        for edge in self.kernel_graph.edges:
            source_node = self.kernel_graph.get_node(edge.source)
            target_node = self.kernel_graph.get_node(edge.target)
            if source_node and target_node:
                lines.append(f"  {source_node.label} --> {target_node.label} ({edge.type.value})")
        
        return "\n".join(lines)


# 示例用法
if __name__ == "__main__":
    from ..parser.cuda_parser import CudaParser, SAMPLE_KERNEL
    
    # 解析kernel
    parser = CudaParser()
    parsed = parser.parse(SAMPLE_KERNEL)
    
    # 构建图
    builder = GraphBuilder()
    graph = builder.build(parsed)
    
    # 打印图信息
    print(builder.visualize_text())
    
    # 导出JSON
    print("\n\nJSON输出:")
    print(graph.to_json())
