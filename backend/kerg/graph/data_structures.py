"""
核心数据结构定义

根据PRD.md定义图节点、边和内核图的数据结构
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import uuid


class NodeType(Enum):
    """图节点类型"""
    OPERATION = "operation"      # 计算操作
    MEMORY = "memory"           # 内存访问
    VARIABLE = "variable"       # 变量/寄存器
    CONTROL = "control"          # 控制流
    FUNCTION = "function"       # 函数调用


class EdgeType(Enum):
    """边类型"""
    DATA = "data"               # 数据依赖
    CONTROL = "control"         # 控制依赖
    MEMORY = "memory"           # 内存依赖


class MemoryAccessType(Enum):
    """内存访问类型"""
    LOAD = "load"
    STORE = "store"


class AddressSpace(Enum):
    """内存地址空间"""
    REGISTER = "register"
    SHARED = "shared"
    GLOBAL = "global"
    CONSTANT = "constant"
    LOCAL = "local"


class ControlType(Enum):
    """控制流类型"""
    IF = "if"
    FOR = "for"
    WHILE = "while"
    SWITCH = "switch"


@dataclass
class GraphNode:
    """
    图节点
    
    属性:
        id: 节点唯一标识
        type: 节点类型
        label: 节点标签/名称
        properties: 节点属性字典
    """
    type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "label": self.label,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """从字典创建节点"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=NodeType(data.get("type", "operation")),
            label=data.get("label", ""),
            properties=data.get("properties", {})
        )


@dataclass
class GraphEdge:
    """
    图边
    
    属性:
        id: 边唯一标识
        source: 源节点ID
        target: 目标节点ID
        type: 边类型
        properties: 边属性字典
    """
    source: str
    target: str
    type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEdge':
        """从字典创建边"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source=data["source"],
            target=data["target"],
            type=EdgeType(data.get("type", "data")),
            properties=data.get("properties", {})
        )


@dataclass
class OperationNode(GraphNode):
    """
    操作节点 - 表示计算操作
    
    额外属性:
        op_type: 操作类型 (add, multiply, etc.)
        operands: 操作数列表
        result: 结果变量
        latency_estimate: 延迟估算
    """
    def __init__(self, label: str, op_type: str, operands: List[str] = None,
                 result: str = None, latency_estimate: float = 0.0, **kwargs):
        super().__init__(
            type=NodeType.OPERATION,
            label=label,
            properties={
                "op_type": op_type,
                "operands": operands or [],
                "result": result,
                "latency_estimate": latency_estimate,
                **kwargs
            }
        )


@dataclass
class MemoryNode(GraphNode):
    """
    内存节点 - 表示内存访问
    
    额外属性:
        access_type: 访问类型 (load/store)
        address_space: 地址空间 (global/shared/register)
        size: 访问大小
    """
    def __init__(self, label: str, access_type: MemoryAccessType,
                 address_space: AddressSpace, size: int = 4, **kwargs):
        super().__init__(
            type=NodeType.MEMORY,
            label=label,
            properties={
                "access_type": access_type.value,
                "address_space": address_space.value,
                "size": size,
                **kwargs
            }
        )


@dataclass
class VariableNode(GraphNode):
    """
    变量节点 - 表示变量或寄存器
    
    额外属性:
        name: 变量名
        var_type: 变量类型
        scope: 作用域
        is_array: 是否为数组
    """
    def __init__(self, label: str, name: str, var_type: str,
                 scope: str = "local", is_array: bool = False, **kwargs):
        super().__init__(
            type=NodeType.VARIABLE,
            label=label,
            properties={
                "name": name,
                "var_type": var_type,
                "scope": scope,
                "is_array": is_array,
                **kwargs
            }
        )


@dataclass
class ControlNode(GraphNode):
    """
    控制流节点 - 表示控制流结构
    
    额外属性:
        control_type: 控制类型 (if/for/while)
        condition: 条件表达式
    """
    def __init__(self, label: str, control_type: ControlType, condition: str = None, **kwargs):
        super().__init__(
            type=NodeType.CONTROL,
            label=label,
            properties={
                "control_type": control_type.value,
                "condition": condition,
                **kwargs
            }
        )


@dataclass
class FunctionNode(GraphNode):
    """
    函数节点 - 表示函数调用
    
    额外属性:
        func_name: 函数名
        arguments: 参数列表
        is_kernel: 是否为kernel函数
    """
    def __init__(self, label: str, func_name: str, arguments: List[str] = None,
                 is_kernel: bool = False, **kwargs):
        super().__init__(
            type=NodeType.FUNCTION,
            label=label,
            properties={
                "func_name": func_name,
                "arguments": arguments or [],
                "is_kernel": is_kernel,
                **kwargs
            }
        )


@dataclass
class KernelMetadata:
    """
    内核图元数据
    """
    total_operations: int = 0
    memory_access_count: int = 0
    loop_count: int = 0
    estimated_flops: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "totalOperations": self.total_operations,
            "memoryAccessCount": self.memory_access_count,
            "loopCount": self.loop_count,
            "estimatedFLOPs": self.estimated_flops
        }


@dataclass
class KernelGraph:
    """
    内核计算图
    
    属性:
        name: 图名称/内核名
        nodes: 节点列表
        edges: 边列表
        entry_point: 入口节点ID
        metadata: 元数据
    """
    name: str
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    entry_point: Optional[str] = None
    metadata: KernelMetadata = field(default_factory=KernelMetadata)
    
    def add_node(self, node: GraphNode) -> None:
        """添加节点"""
        self.nodes.append(node)
        # 如果没有入口点，设置为第一个节点
        if self.entry_point is None:
            self.entry_point = node.id
    
    def add_edge(self, edge: GraphEdge) -> None:
        """添加边"""
        # 验证节点存在
        node_ids = {n.id for n in self.nodes}
        if edge.source in node_ids and edge.target in node_ids:
            self.edges.append(edge)
        else:
            raise ValueError(f"Invalid edge: source or target node not found")
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """根据ID获取节点"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_outgoing_edges(self, node_id: str) -> List[GraphEdge]:
        """获取节点的出边"""
        return [e for e in self.edges if e.source == node_id]
    
    def get_incoming_edges(self, node_id: str) -> List[GraphEdge]:
        """获取节点的入边"""
        return [e for e in self.edges if e.target == node_id]
    
    def update_metadata(self) -> None:
        """更新元数据"""
        self.metadata.total_operations = sum(
            1 for n in self.nodes if n.type == NodeType.OPERATION
        )
        self.metadata.memory_access_count = sum(
            1 for n in self.nodes if n.type == NodeType.MEMORY
        )
        self.metadata.loop_count = sum(
            1 for n in self.nodes 
            if n.type == NodeType.CONTROL 
            and n.properties.get("control_type") == "for"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "entryPoint": self.entry_point,
            "metadata": self.metadata.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KernelGraph':
        """从字典创建图"""
        nodes = [GraphNode.from_dict(n) for n in data.get("nodes", [])]
        edges = [GraphEdge.from_dict(e) for e in data.get("edges", [])]
        metadata_data = data.get("metadata", {})
        metadata = KernelMetadata(
            total_operations=metadata_data.get("totalOperations", 0),
            memory_access_count=metadata_data.get("memoryAccessCount", 0),
            loop_count=metadata_data.get("loopCount", 0),
            estimated_flops=metadata_data.get("estimatedFLOPs", 0.0)
        )
        
        graph = cls(
            name=data.get("name", ""),
            nodes=nodes,
            edges=edges,
            entry_point=data.get("entryPoint"),
            metadata=metadata
        )
        return graph
    
    def to_json(self, indent: int = 2) -> str:
        """导出为JSON字符串"""
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'KernelGraph':
        """从JSON字符串创建"""
        import json
        data = json.loads(json_str)
        return cls.from_dict(data)
