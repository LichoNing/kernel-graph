"""
KernelGraph CLI v2.0

支持多开发模式的命令行接口：
- parse: 解析kernel源码
- visualize: 可视化图结构
- analyze: 性能分析
- generate: 生成代码
- prompt: 生成LLM prompt
- serve: 启动API服务
- ui: 启动Web UI
"""

import click
import json
from pathlib import Path

from ..modes import get_mode, list_modes
from ..visualizer import GraphVisualizer, VisualizationFormat
from ..analysis.analyzer import PerformanceAnalyzer
from ..generator.code_generator import CodeGeneratorFactory


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """KernelGraph - CUDA Kernel 计算图可视化工具 (支持多开发模式)"""
    pass


@cli.command()
def modes():
    """列出支持的开发模式"""
    click.echo("支持的开发模式:")
    for mode in list_modes():
        click.echo(f"  - {mode}")


@cli.command()
@click.option('--input', '-i', required=True, help='输入的kernel文件路径')
@click.option('--output', '-o', default=None, help='输出文件路径')
@click.option('--mode', '-m', default='cuda', help='开发模式 (cuda/cute)')
@click.option('--format', '-f', type=click.Choice(['json', 'text']), default='json', help='输出格式')
def parse(input, output, mode, format):
    """解析kernel源码"""
    try:
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"错误: 文件不存在 - {input}", err=True)
            return
        
        source_code = input_path.read_text(encoding='utf-8')
        
        # 获取模式
        mode_instance = get_mode(mode)
        parsed = mode_instance.parse(source_code)
        
        # 输出结果
        if format == 'json':
            if hasattr(parsed, 'to_dict'):
                output_content = json.dumps(parsed.to_dict(), indent=2, ensure_ascii=False)
            else:
                output_content = json.dumps(parsed.__dict__, indent=2, ensure_ascii=False, default=str)
        else:
            output_content = str(parsed)
        
        if output:
            Path(output).write_text(output_content, encoding='utf-8')
            click.echo(f"解析完成，输出到: {output}")
        else:
            click.echo(output_content)
    
    except Exception as e:
        click.echo(f"错误: {str(e)}", err=True)


@cli.command()
@click.option('--input', '-i', required=True, help='输入的kernel文件路径')
@click.option('--output', '-o', default=None, help='输出文件路径')
@click.option('--mode', '-m', default='cuda', help='开发模式 (cuda/cute)')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'mermaid', 'dot', 'svg']), 
              default='mermaid', help='可视化格式')
def visualize(input, output, mode, format):
    """生成计算图可视化"""
    try:
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"错误: 文件不存在 - {input}", err=True)
            return
        
        source_code = input_path.read_text(encoding='utf-8')
        
        # 获取模式
        mode_instance = get_mode(mode)
        parsed = mode_instance.parse(source_code)
        graph = mode_instance.build_graph(parsed)
        
        # 可视化
        format_map = {
            'text': VisualizationFormat.TEXT,
            'json': VisualizationFormat.JSON,
            'mermaid': VisualizationFormat.MERMAID,
            'dot': VisualizationFormat.DOT,
            'svg': VisualizationFormat.SVG
        }
        viz_format = format_map.get(format, VisualizationFormat.MERMAID)
        
        visualizer = GraphVisualizer()
        output_content = visualizer.visualize(graph, viz_format)
        
        if output:
            Path(output).write_text(output_content, encoding='utf-8')
            click.echo(f"可视化完成，输出到: {output}")
        else:
            click.echo(output_content)
    
    except Exception as e:
        click.echo(f"错误: {str(e)}", err=True)


@cli.command()
@click.option('--input', '-i', required=True, help='输入的kernel文件路径')
@click.option('--output', '-o', default=None, help='输出文件路径')
@click.option('--mode', '-m', default='cuda', help='开发模式 (cuda/cute)')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', help='报告格式')
def analyze(input, output, mode, format):
    """性能分析"""
    try:
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"错误: 文件不存在 - {input}", err=True)
            return
        
        source_code = input_path.read_text(encoding='utf-8')
        
        # 获取模式
        mode_instance = get_mode(mode)
        parsed = mode_instance.parse(source_code)
        graph = mode_instance.build_graph(parsed)
        
        # 分析
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(graph)
        
        # 输出
        if format == 'json':
            output_content = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        else:
            output_content = analyzer.generate_report(result, format='text')
        
        if output:
            Path(output).write_text(output_content, encoding='utf-8')
            click.echo(f"分析完成，输出到: {output}")
        else:
            click.echo(output_content)
    
    except Exception as e:
        click.echo(f"错误: {str(e)}", err=True)


@cli.command()
@click.option('--input', '-i', required=True, help='输入的kernel文件路径或图JSON文件')
@click.option('--output', '-o', default=None, help='输出文件路径')
@click.option('--mode', '-m', default='cuda', help='开发模式 (cuda/cute)')
def generate(input, output, mode):
    """从图生成代码"""
    try:
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"错误: 文件不存在 - {input}", err=True)
            return
        
        content = input_path.read_text(encoding='utf-8')
        
        # 判断输入类型
        if input.endswith('.json'):
            # 从JSON图生成代码
            from ..graph.data_structures import KernelGraph
            graph = KernelGraph.from_json(content)
        else:
            # 从源码生成
            mode_instance = get_mode(mode)
            parsed = mode_instance.parse(content)
            graph = mode_instance.build_graph(parsed)
        
        # 生成代码
        generator = CodeGeneratorFactory.get_generator(mode)
        code = generator.generate(graph)
        
        if output:
            Path(output).write_text(code, encoding='utf-8')
            click.echo(f"代码生成完成，输出到: {output}")
        else:
            click.echo(code)
    
    except Exception as e:
        click.echo(f"错误: {str(e)}", err=True)


@cli.command()
@click.option('--input', '-i', required=True, help='输入的kernel文件路径')
@click.option('--output', '-o', default=None, help='输出文件路径')
@click.option('--mode', '-m', default='cuda', help='开发模式 (cuda/cute)')
@click.option('--style', '-s', type=click.Choice(['brief', 'detailed']), default='detailed', help='prompt风格')
def prompt(input, output, mode, style):
    """生成LLM prompt"""
    try:
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"错误: 文件不存在 - {input}", err=True)
            return
        
        source_code = input_path.read_text(encoding='utf-8')
        
        # 获取模式
        mode_instance = get_mode(mode)
        parsed = mode_instance.parse(source_code)
        graph = mode_instance.build_graph(parsed)
        
        # 生成prompt
        visualizer = GraphVisualizer()
        prompt_text = visualizer.generate_llm_prompt(graph, style=style)
        
        if output:
            Path(output).write_text(prompt_text, encoding='utf-8')
            click.echo(f"Prompt生成完成，输出到: {output}")
        else:
            click.echo(prompt_text)
    
    except Exception as e:
        click.echo(f"错误: {str(e)}", err=True)


@cli.command()
@click.option('--host', default='0.0.0.0', help='服务主机地址')
@click.option('--port', '-p', default=8000, help='服务端口')
def serve(host, port):
    """启动API服务"""
    click.echo(f"启动API服务: http://{host}:{port}")
    
    try:
        from ..api.server import start_server
        start_server(host=host, port=port)
    except ImportError:
        click.echo("错误: API服务依赖未安装，请安装fastapi和uvicorn", err=True)


@cli.command()
@click.option('--port', '-p', default=8080, help='UI服务端口')
def ui(port):
    """启动Web UI"""
    click.echo(f"启动Web UI: http://localhost:{port}")
    
    import http.server
    import socketserver
    import os
    
    # 获取frontend目录
    frontend_dir = Path(__file__).parent.parent.parent.parent.parent / 'frontend'
    
    if not frontend_dir.exists():
        click.echo(f"错误: 前端目录不存在: {frontend_dir}", err=True)
        return
    
    os.chdir(frontend_dir)
    
    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
        click.echo(f"Web UI 运行在 http://localhost:{port}")
        click.echo("按 Ctrl+C 停止服务")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            click.echo("\n服务已停止")


# 主入口
if __name__ == '__main__':
    cli()
