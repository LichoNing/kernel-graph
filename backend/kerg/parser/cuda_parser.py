"""
CUDA Kernel 解析器

解析CUDA kernel源码，提取函数定义、变量声明、内存访问和计算操作
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class TokenType(Enum):
    """词法标记类型"""
    KEYWORD = "keyword"
    IDENTIFIER = "identifier"
    OPERATOR = "operator"
    NUMBER = "number"
    PUNCTUATION = "punctuation"
    COMMENT = "comment"


@dataclass
class Token:
    """词法标记"""
    type: TokenType
    value: str
    line: int
    column: int


@dataclass
class Variable:
    """变量信息"""
    name: str
    var_type: str
    is_pointer: bool = False
    is_array: bool = False
    qualifiers: List[str] = field(default_factory=list)  # __global__, __shared__等


@dataclass
class Operation:
    """操作信息"""
    operator: str
    operands: List[str]
    result: Optional[str] = None
    line: int


@dataclass
class MemoryAccess:
    """内存访问信息"""
    variable: str
    access_type: str  # "load" or "store"
    address_space: str  # "global", "shared", "register"
    indices: List[str] = field(default_factory=list)
    line: int


@dataclass
class ParsedKernel:
    """
    解析后的kernel信息
    
    属性:
        name: kernel函数名
        parameters: 参数列表
        local_variables: 局部变量列表
        operations: 操作列表
        memory_accesses: 内存访问列表
        raw_source: 原始源码
    """
    name: str
    parameters: List[Variable] = field(default_factory=list)
    local_variables: List[Variable] = field(default_factory=list)
    operations: List[Operation] = field(default_factory=list)
    memory_accesses: List[MemoryAccess] = field(default_factory=list)
    raw_source: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "parameters": [
                {
                    "name": v.name,
                    "type": v.var_type,
                    "is_pointer": v.is_pointer,
                    "is_array": v.is_array,
                    "qualifiers": v.qualifiers
                }
                for v in self.parameters
            ],
            "local_variables": [
                {
                    "name": v.name,
                    "type": v.var_type,
                    "is_pointer": v.is_pointer,
                    "is_array": v.is_array
                }
                for v in self.local_variables
            ],
            "operations": [
                {
                    "operator": op.operator,
                    "operands": op.operands,
                    "result": op.result,
                    "line": op.line
                }
                for op in self.operations
            ],
            "memory_accesses": [
                {
                    "variable": ma.variable,
                    "access_type": ma.access_type,
                    "address_space": ma.address_space,
                    "indices": ma.indices,
                    "line": ma.line
                }
                for ma in self.memory_accesses
            ]
        }


class CudaParser:
    """
    CUDA Kernel 解析器
    
    提供词法分析和语法分析功能，提取kernel的关键信息
    """
    
    # CUDA关键字
    CUDA_KEYWORDS = {
        '__global__', '__device__', '__host__', '__shared__', '__constant__',
        'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
        'return', 'break', 'continue', 'void', 'int', 'float', 'double',
        'char', 'short', 'long', 'unsigned', 'signed', 'const', 'volatile',
        'struct', 'typedef', 'extern', 'static', 'inline'
    }
    
    # CUDA内置变量
    CUDA_BUILTINS = {
        'blockIdx', 'blockDim', 'threadIdx', 'gridDim',
        'warpSize', 'laneId', 'smem', 'gmem', 'cmem'
    }
    
    # 算术运算符
    ARITHMETIC_OPS = {'+', '-', '*', '/', '%'}
    
    # 赋值运算符
    ASSIGNMENT_OPS = {'=', '+=', '-=', '*=', '/=', '%='}
    
    # 比较运算符
    COMPARISON_OPS = {'==', '!=', '<', '>', '<=', '>='}
    
    # 逻辑运算符
    LOGICAL_OPS = {'&&', '||', '!'}
    
    def __init__(self):
        self.source_code = ""
        self.tokens: List[Token] = []
        self.current_pos = 0
        self.current_line = 1
        self.current_column = 1
        
    def parse(self, source: str) -> ParsedKernel:
        """
        解析CUDA kernel源码
        
        Args:
            source: CUDA kernel源码字符串
            
        Returns:
            ParsedKernel对象
        """
        self.source_code = source
        self._reset_state()
        
        # 查找kernel函数
        kernel_name, kernel_params, kernel_body = self._extract_kernel()
        
        if not kernel_name:
            raise ValueError("No kernel function found in source code")
        
        # 创建ParsedKernel对象
        kernel = ParsedKernel(name=kernel_name, raw_source=source)
        
        # 解析参数
        kernel.parameters = self._parse_parameters(kernel_params)
        
        # 解析函数体
        self._parse_body(kernel_body, kernel)
        
        return kernel
    
    def _reset_state(self):
        """重置解析器状态"""
        self.tokens = []
        self.current_pos = 0
        self.current_line = 1
        self.current_column = 1
    
    def _extract_kernel(self) -> Tuple[Optional[str], str, str]:
        """提取kernel函数定义和函数体"""
        # 匹配kernel函数定义
        pattern = r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)\s*\{'
        match = re.search(pattern, self.source_code)
        
        if match:
            kernel_name = match.group(1)
            kernel_params = match.group(2)
            
            # 提取函数体
            start_pos = match.end() - 1  # 从{开始
            brace_count = 1
            end_pos = start_pos + 1
            
            while end_pos < len(self.source_code) and brace_count > 0:
                if self.source_code[end_pos] == '{':
                    brace_count += 1
                elif self.source_code[end_pos] == '}':
                    brace_count -= 1
                end_pos += 1
            
            kernel_body = self.source_code[start_pos:end_pos]
            return kernel_name, kernel_params, kernel_body
        
        return None, "", ""
    
    def _parse_parameters(self, params_str: str) -> List[Variable]:
        """解析函数参数"""
        variables = []
        
        if not params_str.strip():
            return variables
        
        # 按逗号分割参数
        param_list = [p.strip() for p in params_str.split(',')]
        
        for param in param_list:
            if not param:
                continue
            
            parts = param.split()
            if len(parts) < 2:
                continue
            
            # 提取类型修饰符
            qualifiers = []
            var_type = ""
            
            for part in parts[:-1]:
                if part.startswith('__') or part in self.CUDA_KEYWORDS:
                    qualifiers.append(part)
                else:
                    var_type = part
            
            # 提取变量名和处理指针/数组
            var_name = parts[-1].strip()
            is_pointer = '*' in var_name
            is_array = '[' in var_name
            
            # 清理变量名
            var_name = var_name.replace('*', '').replace('[', '_').replace(']', '').replace(' ', '')
            
            variables.append(Variable(
                name=var_name,
                var_type=var_type,
                is_pointer=is_pointer,
                is_array=is_array,
                qualifiers=qualifiers
            ))
        
        return variables
    
    def _parse_body(self, body: str, kernel: ParsedKernel):
        """解析函数体"""
        lines = body.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 跳过空行和注释
            if not line or line.startswith('//') or line.startswith('/*'):
                i += 1
                continue
            
            # 解析变量声明
            var = self._parse_variable_declaration(line)
            if var:
                kernel.local_variables.append(var)
            
            # 解析赋值语句
            assignment = self._parse_assignment(line, i + 1)
            if assignment:
                kernel.operations.append(assignment)
            
            # 解析内存访问
            mem_access = self._parse_memory_access(line, i + 1)
            if mem_access:
                kernel.memory_accesses.append(mem_access)
            
            # 解析复合语句 (if, for, while等)
            if line.startswith('if') or line.startswith('for') or line.startswith('while'):
                # 简单处理，递归解析块内内容
                pass
            
            i += 1
    
    def _parse_variable_declaration(self, line: str) -> Optional[Variable]:
        """解析变量声明"""
        # 匹配简单变量声明: type var_name;
        # 或带初始化的声明: type var_name = value;
        pattern = r'(int|float|double|char|short|long|unsigned)\s+(\w+)\s*;?'
        
        match = re.search(pattern, line)
        if match and '=' not in line:
            var_type = match.group(1)
            var_name = match.group(2)
            
            return Variable(
                name=var_name,
                var_type=var_type,
                is_pointer=False,
                is_array=False
            )
        
        return None
    
    def _parse_assignment(self, line: str, line_num: int) -> Optional[Operation]:
        """解析赋值语句"""
        # 匹配赋值语句: result = op1 op op2;
        # 例如: c = a + b; idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        # 跳过if/for/while条件
        if any(line.startswith(kw) for kw in ['if', 'for', 'while', 'return', '//']):
            return None
        
        # 匹配赋值操作
        pattern = r'(\w+)\s*=\s*(.+?);?'
        match = re.search(pattern, line)
        
        if match:
            result = match.group(1)
            expression = match.group(2).strip().rstrip(';')
            
            # 解析操作数和运算符
            operands, operator = self._parse_expression(expression)
            
            return Operation(
                operator=operator or "assign",
                operands=operands,
                result=result,
                line=line_num
            )
        
        return None
    
    def _parse_expression(self, expr: str) -> Tuple[List[str], Optional[str]]:
        """解析表达式，提取操作数和运算符"""
        operands = []
        operator = None
        
        # 清理表达式
        expr = expr.strip().rstrip(';')
        
        # 尝试识别运算符
        for op in ['+', '-', '*', '/', '%']:
            if op in expr:
                parts = expr.split(op)
                if len(parts) == 2:
                    operands = [p.strip() for p in parts]
                    operator = op
                    break
        
        if not operator:
            # 没有运算符，可能是简单赋值
            operands = [expr.strip()]
        
        return operands, operator
    
    def _parse_memory_access(self, line: str, line_num: int) -> Optional[MemoryAccess]:
        """解析内存访问"""
        # 匹配数组访问: var[idx] 或 var[idx1][idx2]
        # 例如: a[idx], b[i], c[i * N + j]
        
        # 检查是否是load操作 (在表达式中)
        load_pattern = r'(\w+)\s*\[\s*([^\]]+)\s*\]'
        matches = re.finditer(load_pattern, line)
        
        accesses = []
        for match in matches:
            var_name = match.group(1)
            indices_str = match.group(2)
            
            # 跳过CUDA内置变量
            if var_name in self.CUDA_BUILTINS:
                continue
            
            # 检查是否是store操作 (在赋值左侧)
            is_store = re.match(rf'^{var_name}\s*\[', line) and '=' in line and line.index('=') > line.index(var_name)
            
            # 解析索引
            indices = [idx.strip() for idx in indices_str.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').split() if idx.strip()]
            
            # 判断地址空间
            address_space = "global"  # 默认全局内存
            
            accesses.append(MemoryAccess(
                variable=var_name,
                access_type="store" if is_store else "load",
                address_space=address_space,
                indices=indices,
                line=line_num
            ))
        
        return accesses[0] if accesses else None
    
    def tokenize(self, source: str) -> List[Token]:
        """
        词法分析，将源码转换为token序列
        
        Args:
            source: 源码字符串
            
        Returns:
            Token列表
        """
        self.source_code = source
        self._reset_state()
        
        while self.current_pos < len(source):
            char = source[self.current_pos]
            
            # 跳过空白字符
            if char.isspace():
                self._advance()
                continue
            
            # 注释
            if char == '/' and self._peek() == '/':
                self._skip_line_comment()
                continue
            if char == '/' and self._peek() == '*':
                self._skip_block_comment()
                continue
            
            # 标识符或关键字
            if char.isalpha() or char == '_':
                self._read_identifier()
            # 数字
            elif char.isdigit() or (char == '.' and self._peek().isdigit()):
                self._read_number()
            # 运算符
            elif char in '+-*/%=<>!&|':
                self._read_operator()
            # 标点符号
            elif char in '(){}[];,:*':
                self.tokens.append(Token(
                    type=TokenType.PUNCTUATION,
                    value=char,
                    line=self.current_line,
                    column=self.current_column
                ))
                self._advance()
            else:
                self._advance()
        
        return self.tokens
    
    def _advance(self):
        """前进一个字符"""
        if self.current_pos < len(self.source_code):
            if self.source_code[self.current_pos] == '\n':
                self.current_line += 1
                self.current_column = 1
            else:
                self.current_column += 1
            self.current_pos += 1
    
    def _peek(self, offset: int = 1) -> str:
        """查看当前字符后的字符"""
        pos = self.current_pos + offset
        if pos < len(self.source_code):
            return self.source_code[pos]
        return ''
    
    def _read_identifier(self):
        """读取标识符"""
        start = self.current_pos
        while self.current_pos < len(self.source_code):
            char = self.source_code[self.current_pos]
            if char.isalnum() or char == '_':
                self._advance()
            else:
                break
        
        value = self.source_code[start:self.current_pos]
        
        # 判断类型
        if value in self.CUDA_KEYWORDS:
            token_type = TokenType.KEYWORD
        else:
            token_type = TokenType.IDENTIFIER
        
        self.tokens.append(Token(
            type=token_type,
            value=value,
            line=self.current_line,
            column=self.current_column - (self.current_pos - start)
        ))
    
    def _read_number(self):
        """读取数字"""
        start = self.current_pos
        has_dot = False
        
        while self.current_pos < len(self.source_code):
            char = self.source_code[self.current_pos]
            if char.isdigit():
                self._advance()
            elif char == '.' and not has_dot:
                has_dot = True
                self._advance()
            else:
                break
        
        self.tokens.append(Token(
            type=TokenType.NUMBER,
            value=self.source_code[start:self.current_pos],
            line=self.current_line,
            column=self.current_column
        ))
    
    def _read_operator(self):
        """读取运算符"""
        char = self.source_code[self.current_pos]
        
        # 检查双字符运算符
        two_char_ops = ['==', '!=', '<=', '>=', '&&', '||', '+=', '-=', '*=', '/=']
        two_char = char + self._peek()
        
        if two_char in two_char_ops:
            self.tokens.append(Token(
                type=TokenType.OPERATOR,
                value=two_char,
                line=self.current_line,
                column=self.current_column
            ))
            self._advance()
            self._advance()
        else:
            self.tokens.append(Token(
                type=TokenType.OPERATOR,
                value=char,
                line=self.current_line,
                column=self.current_column
            ))
            self._advance()
    
    def _skip_line_comment(self):
        """跳过行注释"""
        while self.current_pos < len(self.source_code) and self.source_code[self.current_pos] != '\n':
            self._advance()
    
    def _skip_block_comment(self):
        """跳过块注释"""
        self._advance()  # 跳过 /
        self._advance()  # 跳过 *
        while self.current_pos < len(self.source_code) - 1:
            if self.source_code[self.current_pos] == '*' and self.source_code[self.current_pos + 1] == '/':
                self._advance()  # 跳过 *
                self._advance()  # 跳过 /
                break
            self._advance()


# 示例kernel代码
SAMPLE_KERNEL = """
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""


if __name__ == "__main__":
    # 测试解析器
    parser = CudaParser()
    kernel = parser.parse(SAMPLE_KERNEL)
    
    print(f"Kernel: {kernel.name}")
    print(f"\nParameters:")
    for param in kernel.parameters:
        print(f"  {param.var_type} {param.name}")
    
    print(f"\nLocal Variables:")
    for var in kernel.local_variables:
        print(f"  {var.var_type} {var.name}")
    
    print(f"\nOperations:")
    for op in kernel.operations:
        print(f"  {op.result} = {op.operands} ({op.operator})")
    
    print(f"\nMemory Accesses:")
    for mem in kernel.memory_accesses:
        print(f"  {mem.access_type} {mem.variable}[{mem.indices}] ({mem.address_space})")
