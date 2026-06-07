"""
代码生成器

将KernelGraph转换回可编译的CUDA kernel代码
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..graph.data_structures import KernelGraph, GraphNode, GraphEdge, NodeType


class CodeGenerator(ABC):
    """代码生成器基类"""
    
    @abstractmethod
    def generate(self, graph: KernelGraph) -> str:
        """生成代码"""
        pass
    
    @abstractmethod
    def generate_function_signature(self, graph: KernelGraph) -> str:
        """生成函数签名"""
        pass


class CudaCodeGenerator(CodeGenerator):
    """
    标准CUDA代码生成器
    
    将计算图转换为标准CUDA C/C++代码
    """
    
    def __init__(self):
        self.indent = "    "
        self.current_indent = 0
    
    def generate(self, graph: KernelGraph) -> str:
        """生成完整的CUDA kernel代码"""
        lines = []
        
        # 生成头文件包含
        lines.extend(self._generate_headers())
        lines.append("")
        
        # 生成函数签名
        lines.append(self.generate_function_signature(graph))
        lines.append("{")
        
        # 生成变量声明
        var_declarations = self._generate_variable_declarations(graph)
        if var_declarations:
            lines.append(self.indent + "// Variable declarations")
            for decl in var_declarations:
                lines.append(self.indent + decl)
            lines.append("")
        
        # 生成操作代码
        operations = self._generate_operations(graph)
        if operations:
            lines.append(self.indent + "// Operations")
            for op in operations:
                lines.append(self.indent + op)
            lines.append("")
        
        # 生成内存访问
        memory_accesses = self._generate_memory_accesses(graph)
        if memory_accesses:
            lines.append(self.indent + "// Memory accesses")
            for mem in memory_accesses:
                lines.append(self.indent + mem)
            lines.append("")
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def generate_function_signature(self, graph: KernelGraph) -> str:
        """生成函数签名"""
        # 查找参数节点
        param_nodes = [
            n for n in graph.nodes 
            if n.type == NodeType.VARIABLE and n.properties.get('scope') == 'parameter'
        ]
        
        params = []
        for node in param_nodes:
            var_type = node.properties.get('var_type', 'float')
            name = node.properties.get('name', 'param')
            is_pointer = node.properties.get('is_pointer', False)
            
            if is_pointer:
                param_str = f"{var_type} *{name}"
            else:
                param_str = f"{var_type} {name}"
            
            params.append(param_str)
        
        return f"__global__ void {graph.name}({', '.join(params)})"
    
    def _generate_headers(self) -> List[str]:
        """生成头文件包含"""
        return [
            "#include <cuda_runtime.h>",
            "#include <cuda_fp16.h>",
        ]
    
    def _generate_variable_declarations(self, graph: KernelGraph) -> List[str]:
        """生成变量声明"""
        declarations = []
        
        for node in graph.nodes:
            if node.type == NodeType.VARIABLE:
                scope = node.properties.get('scope', 'local')
                if scope == 'local':
                    var_type = node.properties.get('var_type', 'float')
                    name = node.properties.get('name', 'var')
                    declarations.append(f"{var_type} {name};")
        
        return declarations
    
    def _generate_operations(self, graph: KernelGraph) -> List[str]:
        """生成操作代码"""
        operations = []
        
        for node in graph.nodes:
            if node.type == NodeType.OPERATION:
                op_type = node.properties.get('op_type', 'assign')
                result = node.properties.get('result', '')
                operands = node.properties.get('operands', [])
                
                if op_type == 'assign' and len(operands) == 1:
                    operations.append(f"{result} = {operands[0]};")
                elif op_type in ['+', '-', '*', '/', '%'] and len(operands) == 2:
                    operations.append(f"{result} = {operands[0]} {op_type} {operands[1]};")
                elif result:
                    # 通用操作
                    op_str = f"{result} = {op_type}("
                    op_str += ", ".join(operands)
                    op_str += ");"
                    operations.append(op_str)
        
        return operations
    
    def _generate_memory_accesses(self, graph: KernelGraph) -> List[str]:
        """生成内存访问代码"""
        accesses = []
        
        for node in graph.nodes:
            if node.type == NodeType.MEMORY:
                access_type = node.properties.get('access_type', 'load')
                address_space = node.properties.get('address_space', 'global')
                
                # 这里简化处理，实际应根据图的边关系生成
                if access_type == 'load':
                    accesses.append(f"// Load from {address_space} memory")
                elif access_type == 'store':
                    accesses.append(f"// Store to {address_space} memory")
        
        return accesses


class CuteCodeGenerator(CodeGenerator):
    """
    cute风格代码生成器
    
    将计算图转换为CUTLASS cute风格的代码
    """
    
    def __init__(self):
        self.indent = "    "
    
    def generate(self, graph: KernelGraph) -> str:
        """生成cute风格的kernel代码"""
        lines = []
        
        # 生成头文件
        lines.extend(self._generate_headers())
        lines.append("")
        
        # 生成函数签名
        lines.append(self.generate_function_signature(graph))
        lines.append("{")
        
        # 生成Tensor声明
        tensor_decls = self._generate_tensor_declarations(graph)
        if tensor_decls:
            lines.append(self.indent + "// Tensor declarations")
            for decl in tensor_decls:
                lines.append(self.indent + decl)
            lines.append("")
        
        # 生成TiledMMA配置
        mma_configs = self._generate_mma_configs(graph)
        if mma_configs:
            lines.append(self.indent + "// TiledMMA configuration")
            for config in mma_configs:
                lines.append(self.indent + config)
            lines.append("")
        
        # 生成CopyAtom配置
        copy_configs = self._generate_copy_configs(graph)
        if copy_configs:
            lines.append(self.indent + "// CopyAtom configuration")
            for config in copy_configs:
                lines.append(self.indent + config)
            lines.append("")
        
        # 生成主循环
        main_loop = self._generate_main_loop(graph)
        if main_loop:
            lines.append(self.indent + "// Main computation loop")
            for line in main_loop:
                lines.append(self.indent + line)
            lines.append("")
        
        # 生成Epilogue
        epilogue = self._generate_epilogue(graph)
        if epilogue:
            lines.append(self.indent + "// Epilogue")
            for line in epilogue:
                lines.append(self.indent + line)
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def generate_function_signature(self, graph: KernelGraph) -> str:
        """生成cute kernel函数签名"""
        # 查找参数节点
        param_nodes = [
            n for n in graph.nodes 
            if n.type == NodeType.VARIABLE and n.properties.get('scope') == 'parameter'
        ]
        
        params = []
        for node in param_nodes:
            var_type = node.properties.get('var_type', 'float')
            name = node.properties.get('name', 'param')
            params.append(f"{var_type} *{name}")
        
        return f"__global__ void {graph.name}({', '.join(params)})"
    
    def _generate_headers(self) -> List[str]:
        """生成cute头文件"""
        return [
            "#include <cute/tensor.hpp>",
            "#include <cute/atom/mma_atom.hpp>",
            "using namespace cute;",
        ]
    
    def _generate_tensor_declarations(self, graph: KernelGraph) -> List[str]:
        """生成Tensor声明"""
        declarations = []
        
        for node in graph.nodes:
            if node.type == NodeType.MEMORY:
                name = node.label.replace('tensor_', '')
                layout = node.properties.get('layout', 'auto')
                dtype = node.properties.get('dtype', 'float')
                memory_level = node.properties.get('memory_level', 'global')
                
                if memory_level == 'global':
                    declarations.append(
                        f"Tensor<{dtype}> {name} = make_tensor(make_gmem_ptr({name}_ptr), {layout});"
                    )
                elif memory_level == 'shared':
                    declarations.append(
                        f"__shared__ {dtype} {name}_smem[size];"
                    )
                    declarations.append(
                        f"Tensor<{dtype}> {name} = make_tensor(make_smem_ptr({name}_smem), {layout});"
                    )
        
        return declarations
    
    def _generate_mma_configs(self, graph: KernelGraph) -> List[str]:
        """生成TiledMMA配置"""
        configs = []
        
        for node in graph.nodes:
            if node.type == NodeType.OPERATION and node.properties.get('op_type') == 'tiled_mma':
                mma_op = node.properties.get('mma_op', 'MMA_OP')
                warp_layout = node.properties.get('warp_layout', (1, 1))
                tile_shape = node.properties.get('tile_shape', (16, 8, 16))
                
                configs.append(
                    f"using TiledMMA = decltype(make_tiled_mma("
                )
                configs.append(
                    f"    {mma_op}(),"
                )
                configs.append(
                    f"    Layout<Shape<{warp_layout[0]},{warp_layout[1]}>>(),"
                )
                configs.append(
                    f"    Tile<{tile_shape[0]},{tile_shape[1]},{tile_shape[2]}>>()"
                )
                configs.append(f"));")
        
        return configs
    
    def _generate_copy_configs(self, graph: KernelGraph) -> List[str]:
        """生成CopyAtom配置"""
        configs = []
        
        for node in graph.nodes:
            if node.type == NodeType.OPERATION and node.properties.get('op_type') == 'copy':
                copy_op = node.properties.get('copy_op', 'cp.async')
                configs.append(
                    f"using CopyAtom = Copy_Atom<{copy_op}, float>;"
                )
        
        return configs
    
    def _generate_main_loop(self, graph: KernelGraph) -> List[str]:
        """生成主计算循环"""
        lines = []
        
        # 查找MMA操作
        mma_nodes = [
            n for n in graph.nodes
            if n.type == NodeType.OPERATION and n.properties.get('op_type') == 'mma'
        ]
        
        if mma_nodes:
            lines.append("// Pipeline main loop")
            lines.append("for (int i = 0; i < K; i += K_TILE) {")
            lines.append(self.indent + "// Copy A and B to shared memory")
            lines.append(self.indent + "copy(gmem_a, smem_a);")
            lines.append(self.indent + "copy(gmem_b, smem_b);")
            lines.append("")
            lines.append(self.indent + "// Perform MMA")
            lines.append(self.indent + "gemm(tiled_mma, frag_a, frag_b, accum);")
            lines.append("}")
        
        return lines
    
    def _generate_epilogue(self, graph: KernelGraph) -> List[str]:
        """生成Epilogue代码"""
        lines = []
        
        # 查找epilogue操作
        epilogue_nodes = [
            n for n in graph.nodes
            if n.type == NodeType.OPERATION and node.properties.get('op_type') == 'epilogue'
        ]
        
        if epilogue_nodes:
            lines.append("// Store results to global memory")
            lines.append("copy(smem_c, gmem_c);")
        else:
            lines.append("// Store accumulator to output")
            lines.append("copy(accum, gmem_c);")
        
        return lines


class CodeGeneratorFactory:
    """代码生成器工厂"""
    
    _generators = {
        'cuda': CudaCodeGenerator,
        'cute': CuteCodeGenerator,
    }
    
    @classmethod
    def get_generator(cls, mode: str) -> CodeGenerator:
        """获取对应模式的代码生成器"""
        mode = mode.lower()
        if mode not in cls._generators:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(cls._generators.keys())}")
        return cls._generators[mode]()
    
    @classmethod
    def register_generator(cls, mode: str, generator_class: type):
        """注册新的代码生成器"""
        cls._generators[mode] = generator_class
