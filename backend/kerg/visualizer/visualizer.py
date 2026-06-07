"""
图可视化器

支持多种可视化格式：
- Text: 纯文本表格
- JSON: 结构化数据
- Mermaid: Markdown流程图
- DOT: Graphviz格式
- SVG: 矢量图
"""

import json
from enum import Enum
from typing import Dict, List, Optional, Any
from ..graph.data_structures import KernelGraph, GraphNode, GraphEdge, NodeType


class VisualizationFormat(Enum):
    """可视化格式"""
    TEXT = "text"
    JSON = "json"
    MERMAID = "mermaid"
    DOT = "dot"
    SVG = "svg"


class GraphVisualizer:
    """
    图可视化器
    
    将KernelGraph转换为各种可视化格式
    """
    
    def __init__(self):
        self.format_handlers = {
            VisualizationFormat.TEXT: self._to_text,
            VisualizationFormat.JSON: self._to_json,
            VisualizationFormat.MERMAID: self._to_mermaid,
            VisualizationFormat.DOT: self._to_dot,
            VisualizationFormat.SVG: self._to_svg,
        }
    
    def visualize(self, graph: KernelGraph, format: VisualizationFormat = VisualizationFormat.TEXT,
                  options: Optional[Dict[str, Any]] = None) -> str:
        """
        可视化图
        
        Args:
            graph: 要可视化的图
            format: 输出格式
            options: 可视化选项
            
        Returns:
            可视化输出字符串
        """
        options = options or {}
        
        if format not in self.format_handlers:
            raise ValueError(f"Unsupported format: {format}")
        
        return self.format_handlers[format](graph, options)
    
    def _to_text(self, graph: KernelGraph, options: Dict[str, Any]) -> str:
        """转换为文本格式"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"KernelGraph: {graph.name}")
        lines.append("=" * 70)
        
        # 元数据
        lines.append(f"\n[元数据]")
        lines.append(f"  总节点数: {len(graph.nodes)}")
        lines.append(f"  总边数: {len(graph.edges)}")
        lines.append(f"  操作数: {graph.metadata.total_operations}")
        lines.append(f"  内存访问数: {graph.metadata.memory_access_count}")
        lines.append(f"  循环数: {graph.metadata.loop_count}")
        lines.append(f"  估算FLOPs: {graph.metadata.estimated_flops}")
        
        # 节点列表
        lines.append(f"\n[节点列表]")
        for i, node in enumerate(graph.nodes):
            node_type = node.type.value
            props = self._format_properties(node.properties)
            lines.append(f"  [{i:3d}] {node.id[:8]}... | {node_type:12s} | {node.label:20s} | {props}")
        
        # 边列表
        lines.append(f"\n[边列表]")
        for i, edge in enumerate(graph.edges):
            src = graph.get_node(edge.source)
            dst = graph.get_node(edge.target)
            src_label = src.label if src else "?"
            dst_label = dst.label if dst else "?"
            lines.append(f"  [{i:3d}] {src_label:20s} --[{edge.type.value:8s}]--> {dst_label:20s}")
        
        # 邻接表
        lines.append(f"\n[邻接表]")
        for node in graph.nodes:
            outgoing = graph.get_outgoing_edges(node.id)
            if outgoing:
                targets = [graph.get_node(e.target).label for e in outgoing if graph.get_node(e.target)]
                lines.append(f"  {node.label:20s} -> {', '.join(targets)}")
        
        lines.append("\n" + "=" * 70)
        return '\n'.join(lines)
    
    def _to_json(self, graph: KernelGraph, options: Dict[str, Any]) -> str:
        """转换为JSON格式"""
        return graph.to_json(indent=options.get('indent', 2))
    
    def _to_mermaid(self, graph: KernelGraph, options: Dict[str, Any]) -> str:
        """转换为Mermaid流程图"""
        lines = ["flowchart TD"]
        
        # 添加子图分组
        groups = self._group_nodes_by_type(graph)
        
        for group_name, nodes in groups.items():
            if len(nodes) > 0:
                lines.append(f"    subgraph {group_name}[{group_name.upper()}]")
                for node in nodes:
                    shape = self._get_mermaid_shape(node)
                    lines.append(f"        {node.id}({node.label})")
                lines.append(f"    end")
        
        # 添加边
        for edge in graph.edges:
            src = graph.get_node(edge.source)
            dst = graph.get_node(edge.target)
            if src and dst:
                edge_style = self._get_mermaid_edge_style(edge)
                lines.append(f"    {src.id} {edge_style} {dst.id}")
        
        # 添加样式
        lines.append("")
        lines.append("    classDef operation fill:#f9f,stroke:#333,stroke-width:2px")
        lines.append("    classDef memory fill:#bbf,stroke:#333,stroke-width:2px")
        lines.append("    classDef variable fill:#bfb,stroke:#333,stroke-width:2px")
        lines.append("    classDef control fill:#fbb,stroke:#333,stroke-width:2px")
        
        for node in graph.nodes:
            lines.append(f"    class {node.id} {node.type.value}")
        
        return '\n'.join(lines)
    
    def _to_dot(self, graph: KernelGraph, options: Dict[str, Any]) -> str:
        """转换为Graphviz DOT格式"""
        lines = [f'digraph {graph.name} {{']
        lines.append('    rankdir=TB;')
        lines.append('    node [shape=box, style="rounded,filled", fontname="Arial"];')
        lines.append('')
        
        # 颜色映射
        color_map = {
            'operation': '#ff99ff',
            'memory': '#bbbbff',
            'variable': '#bbffbb',
            'control': '#ffbbbb',
            'function': '#ffffbb'
        }
        
        # 节点定义
        for node in graph.nodes:
            color = color_map.get(node.type.value, '#ffffff')
            label = node.label.replace('"', '\\"')
            lines.append(f'    "{node.id}" [label="{label}", fillcolor="{color}"];')
        
        lines.append('')
        
        # 边定义
        for edge in graph.edges:
            style = 'dashed' if edge.type.value == 'control' else 'solid'
            lines.append(f'    "{edge.source}" -> "{edge.target}" [style={style}];')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def _to_svg(self, graph: KernelGraph, options: Dict[str, Any]) -> str:
        """转换为SVG格式（简化版）"""
        # 简化的SVG实现，实际使用时可调用graphviz
        width = 800
        height = max(600, len(graph.nodes) * 50)
        
        lines = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '  <defs>',
            '    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
            '      <polygon points="0 0, 10 3.5, 0 7" fill="#333" />',
            '    </marker>',
            '  </defs>',
            '  <rect width="100%" height="100%" fill="#f5f5f5"/>',
            f'  <text x="10" y="30" font-family="Arial" font-size="20" font-weight="bold">{graph.name}</text>',
        ]
        
        # 简化的节点布局（网格布局）
        cols = 3
        node_positions = {}
        for i, node in enumerate(graph.nodes):
            x = 50 + (i % cols) * 250
            y = 80 + (i // cols) * 80
            node_positions[node.id] = (x, y)
            
            color_map = {
                'operation': '#ff99ff',
                'memory': '#bbbbff',
                'variable': '#bbffbb',
                'control': '#ffbbbb',
                'function': '#ffffbb'
            }
            color = color_map.get(node.type.value, '#ffffff')
            
            lines.append(f'  <rect x="{x}" y="{y}" width="200" height="50" rx="5" fill="{color}" stroke="#333" stroke-width="2"/>')
            lines.append(f'  <text x="{x+100}" y="{y+30}" font-family="Arial" font-size="12" text-anchor="middle">{node.label}</text>')
        
        # 边
        for edge in graph.edges:
            if edge.source in node_positions and edge.target in node_positions:
                x1, y1 = node_positions[edge.source]
                x2, y2 = node_positions[edge.target]
                # 调整起点和终点到矩形边缘
                x1 += 200
                y1 += 25
                y2 += 25
                
                lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" stroke-width="1.5" marker-end="url(#arrowhead)"/>')
        
        lines.append('</svg>')
        return '\n'.join(lines)
    
    def _format_properties(self, props: Dict[str, Any]) -> str:
        """格式化属性字典"""
        if not props:
            return ""
        items = [f"{k}={v}" for k, v in list(props.items())[:3]]
        return ", ".join(items)
    
    def _group_nodes_by_type(self, graph: KernelGraph) -> Dict[str, List[GraphNode]]:
        """按类型分组节点"""
        groups = {
            'operations': [],
            'memory': [],
            'variables': [],
            'control': [],
            'functions': []
        }
        
        for node in graph.nodes:
            if node.type == NodeType.OPERATION:
                groups['operations'].append(node)
            elif node.type == NodeType.MEMORY:
                groups['memory'].append(node)
            elif node.type == NodeType.VARIABLE:
                groups['variables'].append(node)
            elif node.type == NodeType.CONTROL:
                groups['control'].append(node)
            elif node.type == NodeType.FUNCTION:
                groups['functions'].append(node)
        
        return groups
    
    def _get_mermaid_shape(self, node: GraphNode) -> str:
        """获取Mermaid节点形状"""
        shape_map = {
            'operation': '([{label}])',
            'memory': '[{label}]',
            'variable': '({label})',
            'control': '{{{label}}}',
            'function': '[/{label}/]'
        }
        return shape_map.get(node.type.value, '[{label}]')
    
    def _get_mermaid_edge_style(self, edge: GraphEdge) --> str:
        """获取Mermaid边样式"""
        if edge.type.value == 'control':
            return '-.->'
        elif edge.type.value == 'memory':
            return '==>'
        else:
            return '-->'
    
    def render_to_file(self, graph: KernelGraph, format: VisualizationFormat,
                       filepath: str, options: Optional[Dict[str, Any]] = None) -> None:
        """
        渲染图到文件
        
        Args:
            graph: 要渲染的图
            format: 输出格式
            filepath: 输出文件路径
            options: 渲染选项
        """
        output = self.visualize(graph, format, options)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
    
    def generate_llm_prompt(self, graph: KernelGraph, style: str = 'detailed') -> str:
        """
        生成LLM prompt
        
        Args:
            graph: 计算图
            style: prompt风格 (brief/detailed)
            
        Returns:
            LLM prompt字符串
        """
        lines = []
        lines.append("分析以下CUDA kernel的计算图：\n")
        lines.append(f"Kernel: {graph.name}")
        
        # 节点统计
        type_counts = {}
        for node in graph.nodes:
            t = node.type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        lines.append(f"\n节点统计:")
        for t, count in type_counts.items():
            lines.append(f"  {t}: {count}")
        
        if style == 'detailed':
            lines.append(f"\n节点详情:")
            for node in graph.nodes:
                props = self._format_properties(node.properties)
                lines.append(f"  [{node.type.value}] {node.label}: {props}")
            
            lines.append(f"\n依赖关系:")
            for edge in graph.edges:
                src = graph.get_node(edge.source)
                dst = graph.get_node(edge.target)
                if src and dst:
                    lines.append(f"  {src.label} -> {dst.label} ({edge.type.value})")
        
        lines.append(f"\n请分析此kernel的性能特点并提供优化建议。")
        
        return '\n'.join(lines)
