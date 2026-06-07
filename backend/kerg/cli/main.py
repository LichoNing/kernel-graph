"""
KernelGraph CLI

命令行接口，提供以下命令：
- parse: 解析kernel源码
- visualize: 可视化图结构
- prompt: 生成LLM prompt
"""

import click
import json
from pathlib import Path

from ..parser.cuda_parser import CudaParser
from ..graph.graph_builder import GraphBuilder


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """KernelGraph - CUDA Kernel 计算图可视化工具"""
    pass


@cli.command()
@click.option('--input', '-i', required=True, help='输入的CUDA kernel文件路径')
@click.option('--output', '-o', default=None, help='输出文件路径')
@click.option('--format', '-f', type=click.Choice(['json', 'text']), default='json', help='输出格式')
def parse(input, output, format):
    """解析CUDA kernel源码"""
    try:
        # 读取输入文件
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"错误: 文件不存在 - {input}", err=True)
            return
        
        source_code = input_path.read_text(encoding='utf-8')
        
        # 解析
        parser = CudaParser()
        parsed_kernel = parser.parse(source_code)
        
        # 输出结果
        result = parsed_kernel.to_dict()
        
        if format == 'json':
            output_content = json.dumps(result, indent=2, ensure_ascii=False)
        else:
            output_content = _format_as_text(result)
        
        if output:
            Path(output).write_text(output_content, encoding='utf-8')
            click.echo(f"解析完成，输出到: {output}")
        else:
            click.echo(output_content)
    
    except Exception as e:
        click.echo(f"错误: {str(e)}", err=True)


@cli.command()
@click.option('--input', '-i', required=True, help='输入的CUDA kernel文件路径')
@click.option('--output', '-o', default=None, help='输出文件路径')
@click.option('--format', '-f', type=click.Choice(['json', 'text']), default='json', help='输出格式')
def visualize(input, output, format):
    """生成计算图可视化"""
    try:
        # 读取输入文件
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"错误: 文件不存在 - {input}", err=True)
            return
        
        source_code = input_path.read_text(encoding='utf-8')
        
        # 解析
        parser = CudaParser()
        parsed_kernel = parser.parse(source_code)
        
        # 构建图
        builder = GraphBuilder()
        graph = builder.build(parsed_kernel)
        
        # 输出结果
        if format == 'json':
            output_content = graph.to_json()
        else:
            output_content = builder.visualize_text()
        
        if output:
            Path(output).write_text(output_content, encoding='utf-8')
            click.echo(f"可视化完成，输出到: {output}")
        else:
            click.echo(output_content)
    
    except Exception as e:
        click.echo(f"错误: {str(e)}", err=True)


@cli.command()
@click.option('--input', '-i', required=True, help='输入的CUDA kernel文件路径')
@click.option('--output', '-o', default=None, help='输出文件路径')
@click.option('--style', '-s', type=click.Choice(['brief', 'detailed']), default='detailed', help='prompt风格')
def prompt(input, output, style):
    """生成LLM prompt"""
    try:
        # 读取输入文件
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"错误: 文件不存在 - {input}", err=True)
            return
        
        source_code = input_path.read_text(encoding='utf-8')
        
        # 解析
        parser = CudaParser()
        parsed_kernel = parser.parse(source_code)
        
        # 构建图
        builder = GraphBuilder()
        graph = builder.build(parsed_kernel)
        
        # 生成prompt
        prompt_text = _generate_llm_prompt(graph, parsed_kernel, style)
        
        if output:
            Path(output).write_text(prompt_text, encoding='utf-8')
            click.echo(f"Prompt生成完成，输出到: {output}")
        else:
            click.echo(prompt_text)
    
    except Exception as e:
        click.echo(f"错误: {str(e)}", err=True)


def _format_as_text(data: dict) -> str:
    """格式化解析结果为文本"""
    lines = []
    lines.append(f"Kernel: {data['name']}")
    
    lines.append("\n参数:")
    for param in data.get('parameters', []):
        qualifiers = ' '.join(param.get('qualifiers', []))
        ptr = '*' if param.get('is_pointer') else ''
        arr = '[]' if param.get('is_array') else ''
        lines.append(f"  {qualifiers} {param['type']} {ptr}{param['name']}{arr}")
    
    lines.append("\n局部变量:")
    for var in data.get('local_variables', []):
        lines.append(f"  {var['type']} {var['name']}")
    
    lines.append("\n操作:")
    for op in data.get('operations', []):
        if op.get('result'):
            lines.append(f"  {op['result']} = {' '.join(op['operands'])} ({op['operator']})")
        else:
            lines.append(f"  {' '.join(op['operands'])} ({op['operator']})")
    
    lines.append("\n内存访问:")
    for mem in data.get('memory_accesses', []):
        indices = ']['.join(mem.get('indices', []))
        lines.append(f"  {mem['access_type']} {mem['variable']}[{indices}] ({mem['address_space']})")
    
    return '\n'.join(lines)


def _generate_llm_prompt(graph, parsed_kernel, style: str) -> str:
    """生成LLM prompt"""
    lines = []
    
    lines.append("分析以下CUDA kernel的计算图：\n")
    lines.append(f"Kernel: {parsed_kernel.name}")
    
    # 参数信息
    lines.append(f"\n参数:")
    for param in parsed_kernel.parameters:
        qualifiers = ' '.join(param.qualifiers)
        ptr = '*' if param.is_pointer else ''
        lines.append(f"  {qualifiers} {param.var_type} {ptr}{param.name}")
    
    # 计算图结构
    lines.append(f"\n计算图结构:")
    lines.append(f"  节点总数: {len(graph.nodes)}")
    lines.append(f"  边总数: {len(graph.edges)}")
    lines.append(f"  操作数: {graph.metadata.total_operations}")
    lines.append(f"  内存访问数: {graph.metadata.memory_access_count}")
    lines.append(f"  循环数: {graph.metadata.loop_count}")
    lines.append(f"  估算FLOPs: {graph.metadata.estimated_flops}")
    
    if style == 'detailed':
        lines.append(f"\n节点详情:")
        for node in graph.nodes:
            props = ", ".join(f"{k}={v}" for k, v in node.properties.items() if v)
            lines.append(f"  [{node.label}] {node.type.value} - {props}")
        
        lines.append(f"\n依赖关系:")
        for edge in graph.edges:
            source = graph.get_node(edge.source)
            target = graph.get_node(edge.target)
            if source and target:
                lines.append(f"  {source.label} → {target.label} ({edge.type.value})")
    
    lines.append(f"\n请分析此kernel的性能特点并提供优化建议。")
    
    return '\n'.join(lines)


# 主入口
if __name__ == '__main__':
    cli()
