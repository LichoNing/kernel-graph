"""
cute模式端到端测试

验证cute GEMM kernel的解析、建图、分析、可视化完整流程
"""

import sys
from pathlib import Path

# 添加backend到路径
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from kerg.modes import get_mode
from kerg.visualizer import GraphVisualizer, VisualizationFormat
from kerg.analysis.analyzer import PerformanceAnalyzer
from kerg.generator.code_generator import CodeGeneratorFactory


def load_cute_gemm():
    """加载cute GEMM示例代码"""
    example_path = Path(__file__).parent / 'examples' / 'cute_gemm.cu'
    return example_path.read_text(encoding='utf-8')


def test_cute_parse():
    """测试cute解析"""
    print("=" * 70)
    print("测试1: cute模式解析")
    print("=" * 70)
    
    source = load_cute_gemm()
    mode = get_mode('cute')
    
    # 解析
    parsed = mode.parse(source)
    
    print(f"\n✓ 解析成功!")
    print(f"  Kernel名称: {parsed.name}")
    print(f"  参数数量: {len(parsed.params)}")
    print(f"  Tensor数量: {len(parsed.tensors)}")
    print(f"  TiledMMA数量: {len(parsed.tiled_mmas)}")
    print(f"  CopyAtom数量: {len(parsed.copy_atoms)}")
    print(f"  操作数量: {len(parsed.operations)}")
    
    # 显示Tensor信息
    print(f"\n  Tensor详情:")
    for tensor in parsed.tensors:
        print(f"    {tensor.name}: {tensor.tensor_type.value} | {tensor.layout} | {tensor.memory_level.value}")
    
    # 显示TiledMMA信息
    print(f"\n  TiledMMA详情:")
    for mma in parsed.tiled_mmas:
        print(f"    {mma.name}: {mma.mma_op} | warp_layout={mma.warp_layout} | tile={mma.tile_shape}")
    
    return parsed


def test_cute_graph_build(parsed):
    """测试cute图构建"""
    print("\n" + "=" * 70)
    print("测试2: cute图构建")
    print("=" * 70)
    
    mode = get_mode('cute')
    graph = mode.build_graph(parsed)
    
    print(f"\n✓ 图构建成功!")
    print(f"  图名称: {graph.name}")
    print(f"  节点数量: {len(graph.nodes)}")
    print(f"  边数量: {len(graph.edges)}")
    print(f"  元数据:")
    print(f"    - 总操作数: {graph.metadata.total_operations}")
    print(f"    - 内存访问数: {graph.metadata.memory_access_count}")
    print(f"    - 循环数: {graph.metadata.loop_count}")
    
    # 按类型统计节点
    type_counts = {}
    for node in graph.nodes:
        t = node.type.value
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"\n  节点类型分布:")
    for t, count in type_counts.items():
        print(f"    {t}: {count}")
    
    return graph


def test_cute_visualization(graph):
    """测试cute可视化"""
    print("\n" + "=" * 70)
    print("测试3: cute可视化")
    print("=" * 70)
    
    mode = get_mode('cute')
    
    # 文本可视化
    print(f"\n  [文本可视化]")
    text_viz = mode.visualize(graph, format='text')
    print(text_viz[:1000] + "..." if len(text_viz) > 1000 else text_viz)
    
    # Mermaid可视化
    print(f"\n  [Mermaid可视化]")
    mermaid_viz = mode.visualize(graph, format='mermaid')
    print(mermaid_viz[:500] + "..." if len(mermaid_viz) > 500 else mermaid_viz)
    
    # 使用可视化器
    visualizer = GraphVisualizer()
    
    # JSON格式
    json_viz = visualizer.visualize(graph, VisualizationFormat.JSON)
    print(f"\n  [JSON可视化] 长度: {len(json_viz)} 字符")
    
    # DOT格式
    dot_viz = visualizer.visualize(graph, VisualizationFormat.DOT)
    print(f"\n  [DOT可视化] 长度: {len(dot_viz)} 字符")
    
    # 生成LLM prompt
    prompt = visualizer.generate_llm_prompt(graph, style='detailed')
    print(f"\n  [LLM Prompt] 长度: {len(prompt)} 字符")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    return visualizer


def test_cute_analysis(graph):
    """测试cute分析"""
    print("\n" + "=" * 70)
    print("测试4: cute性能分析")
    print("=" * 70)
    
    mode = get_mode('cute')
    analysis = mode.analyze(graph)
    
    print(f"\n✓ 分析完成!")
    print(f"  综合评分: {analysis['overall_score']:.2f}")
    
    print(f"\n  计算指标:")
    for key, value in analysis.get('compute_metrics', {}).items():
        print(f"    {key}: {value}")
    
    print(f"\n  内存指标:")
    for key, value in analysis.get('memory_metrics', {}).items():
        print(f"    {key}: {value}")
    
    print(f"\n  并行指标:")
    for key, value in analysis.get('parallelism_metrics', {}).items():
        print(f"    {key}: {value}")
    
    # 使用分析器生成报告
    analyzer = PerformanceAnalyzer()
    result = analyzer.analyze(graph)
    report = analyzer.generate_report(result, format='text')
    
    print(f"\n  [分析报告]")
    print(report[:1500] + "..." if len(report) > 1500 else report)
    
    return analysis


def test_cute_code_generation(graph):
    """测试cute代码生成"""
    print("\n" + "=" * 70)
    print("测试5: cute代码生成")
    print("=" * 70)
    
    mode = get_mode('cute')
    code = mode.generate_code(graph)
    
    print(f"\n✓ 代码生成完成!")
    print(f"  代码长度: {len(code)} 字符")
    print(f"\n  [生成的代码]")
    print(code[:1000] + "..." if len(code) > 1000 else code)
    
    return code


def test_end_to_end():
    """端到端测试"""
    print("\n" + "=" * 70)
    print("测试6: 端到端流程")
    print("=" * 70)
    
    source = load_cute_gemm()
    mode = get_mode('cute')
    
    # 完整流程
    result = mode.process(source)
    
    print(f"\n✓ 端到端流程完成!")
    print(f"  模式: {result['mode']}")
    print(f"  解析: {result['parsed'].name}")
    print(f"  图节点: {len(result['graph'].nodes)}")
    print(f"  分析评分: {result['analysis'].get('overall_score', 0):.2f}")
    print(f"  可视化长度: {len(result['visualization'])} 字符")
    
    return result


def export_outputs(graph, analysis):
    """导出输出文件"""
    print("\n" + "=" * 70)
    print("导出输出文件")
    print("=" * 70)
    
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # 导出JSON
    json_path = output_dir / 'cute_gemm_graph.json'
    json_path.write_text(graph.to_json(), encoding='utf-8')
    print(f"\n✓ 导出JSON: {json_path}")
    
    # 导出Mermaid
    mode = get_mode('cute')
    mermaid = mode.visualize(graph, format='mermaid')
    mermaid_path = output_dir / 'cute_gemm_graph.mmd'
    mermaid_path.write_text(mermaid, encoding='utf-8')
    print(f"✓ 导出Mermaid: {mermaid_path}")
    
    # 导出DOT
    visualizer = GraphVisualizer()
    dot = visualizer.visualize(graph, VisualizationFormat.DOT)
    dot_path = output_dir / 'cute_gemm_graph.dot'
    dot_path.write_text(dot, encoding='utf-8')
    print(f"✓ 导出DOT: {dot_path}")
    
    # 导出分析报告
    import json
    report_path = output_dir / 'cute_gemm_analysis.json'
    report_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"✓ 导出分析报告: {report_path}")
    
    # 导出LLM prompt
    prompt = visualizer.generate_llm_prompt(graph, style='detailed')
    prompt_path = output_dir / 'cute_gemm_prompt.txt'
    prompt_path.write_text(prompt, encoding='utf-8')
    print(f"✓ 导出LLM Prompt: {prompt_path}")


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("KernelGraph cute模式端到端测试")
    print("=" * 70)
    
    try:
        # 1. 解析
        parsed = test_cute_parse()
        
        # 2. 建图
        graph = test_cute_graph_build(parsed)
        
        # 3. 可视化
        visualizer = test_cute_visualization(graph)
        
        # 4. 分析
        analysis = test_cute_analysis(graph)
        
        # 5. 代码生成
        code = test_cute_code_generation(graph)
        
        # 6. 端到端
        result = test_end_to_end()
        
        # 7. 导出
        export_outputs(graph, analysis)
        
        print("\n" + "=" * 70)
        print("所有测试通过! ✓")
        print("=" * 70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
