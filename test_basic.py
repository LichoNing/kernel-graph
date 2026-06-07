"""
测试脚本 - 验证KernelGraph功能
"""

import sys
from pathlib import Path

# 添加backend到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from kerg.parser.cuda_parser import CudaParser, SAMPLE_KERNEL
from kerg.graph.graph_builder import GraphBuilder


def test_parser():
    """测试解析器"""
    print("=" * 60)
    print("测试解析器")
    print("=" * 60)
    
    parser = CudaParser()
    kernel = parser.parse(SAMPLE_KERNEL)
    
    print(f"\n✓ 解析成功!")
    print(f"  Kernel名称: {kernel.name}")
    print(f"  参数数量: {len(kernel.parameters)}")
    print(f"  局部变量: {len(kernel.local_variables)}")
    print(f"  操作数: {len(kernel.operations)}")
    print(f"  内存访问: {len(kernel.memory_accesses)}")
    
    return kernel


def test_graph_builder(kernel):
    """测试图构建器"""
    print("\n" + "=" * 60)
    print("测试图构建器")
    print("=" * 60)
    
    builder = GraphBuilder()
    graph = builder.build(kernel)
    
    print(f"\n✓ 图构建成功!")
    print(f"  图名称: {graph.name}")
    print(f"  节点数量: {len(graph.nodes)}")
    print(f"  边数量: {len(graph.edges)}")
    print(f"  元数据:")
    print(f"    - 总操作数: {graph.metadata.total_operations}")
    print(f"    - 内存访问数: {graph.metadata.memory_access_count}")
    print(f"    - 循环数: {graph.metadata.loop_count}")
    print(f"    - 估算FLOPs: {graph.metadata.estimated_flops}")
    
    return graph


def test_json_export(graph):
    """测试JSON导出"""
    print("\n" + "=" * 60)
    print("测试JSON导出")
    print("=" * 60)
    
    json_str = graph.to_json()
    print(f"\n✓ JSON导出成功! (长度: {len(json_str)} 字符)")
    
    # 测试导入
    graph2 = graph.from_json(json_str)
    print(f"✓ JSON导入成功!")
    print(f"  验证: 节点数={len(graph2.nodes)}, 边数={len(graph2.edges)}")
    
    return json_str


def test_visualization():
    """测试可视化文本生成"""
    print("\n" + "=" * 60)
    print("测试可视化文本生成")
    print("=" * 60)
    
    parser = CudaParser()
    kernel = parser.parse(SAMPLE_KERNEL)
    builder = GraphBuilder()
    graph = builder.build(kernel)
    
    text = builder.visualize_text()
    print(f"\n✓ 可视化文本生成成功!")
    print("\n" + text)
    
    return text


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("KernelGraph 功能测试")
    print("=" * 60)
    
    try:
        # 测试解析器
        kernel = test_parser()
        
        # 测试图构建器
        graph = test_graph_builder(kernel)
        
        # 测试JSON导出
        test_json_export(graph)
        
        # 测试可视化
        test_visualization()
        
        print("\n" + "=" * 60)
        print("所有测试通过! ✓")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
