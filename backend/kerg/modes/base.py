"""
开发模式基类

定义所有开发模式必须实现的接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ParsedResult:
    """解析结果基类"""
    kernel_name: str
    raw_source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    """解析器基类"""
    
    @abstractmethod
    def parse(self, source: str) -> ParsedResult:
        """解析源码"""
        pass
    
    @abstractmethod
    def tokenize(self, source: str) -> List[Any]:
        """词法分析"""
        pass


class BaseGraphBuilder(ABC):
    """图构建器基类"""
    
    @abstractmethod
    def build(self, parsed_result: ParsedResult) -> Any:
        """构建计算图"""
        pass


class BaseCodeGenerator(ABC):
    """代码生成器基类"""
    
    @abstractmethod
    def generate(self, graph: Any) -> str:
        """从图生成代码"""
        pass


class BaseAnalyzer(ABC):
    """分析引擎基类"""
    
    @abstractmethod
    def analyze(self, graph: Any) -> Dict[str, Any]:
        """分析计算图"""
        pass


class BaseVisualizer(ABC):
    """可视化基类"""
    
    @abstractmethod
    def visualize(self, graph: Any, format: str = 'text') -> str:
        """可视化图"""
        pass


class BaseMode(ABC):
    """
    开发模式基类
    
    每个开发模式需要提供：
    - 解析器
    - 图构建器
    - 代码生成器
    - 分析器
    - 可视化器
    """
    
    name: str = ""
    description: str = ""
    file_extensions: List[str] = []
    
    def __init__(self):
        self.parser = self._create_parser()
        self.graph_builder = self._create_graph_builder()
        self.code_generator = self._create_code_generator()
        self.analyzer = self._create_analyzer()
        self.visualizer = self._create_visualizer()
    
    @abstractmethod
    def _create_parser(self) -> BaseParser:
        """创建解析器"""
        pass
    
    @abstractmethod
    def _create_graph_builder(self) -> BaseGraphBuilder:
        """创建图构建器"""
        pass
    
    @abstractmethod
    def _create_code_generator(self) -> BaseCodeGenerator:
        """创建代码生成器"""
        pass
    
    @abstractmethod
    def _create_analyzer(self) -> BaseAnalyzer:
        """创建分析器"""
        pass
    
    @abstractmethod
    def _create_visualizer(self) -> BaseVisualizer:
        """创建可视化器"""
        pass
    
    def parse(self, source: str) -> ParsedResult:
        """解析源码"""
        return self.parser.parse(source)
    
    def build_graph(self, parsed_result: ParsedResult) -> Any:
        """构建图"""
        return self.graph_builder.build(parsed_result)
    
    def generate_code(self, graph: Any) -> str:
        """生成代码"""
        return self.code_generator.generate(graph)
    
    def analyze(self, graph: Any) -> Dict[str, Any]:
        """分析图"""
        return self.analyzer.analyze(graph)
    
    def visualize(self, graph: Any, format: str = 'text') -> str:
        """可视化"""
        return self.visualizer.visualize(graph, format)
    
    def process(self, source: str) -> Dict[str, Any]:
        """
        完整处理流程：解析 -> 建图 -> 分析 -> 可视化
        
        Returns:
            包含所有中间结果的字典
        """
        parsed = self.parse(source)
        graph = self.build_graph(parsed)
        analysis = self.analyze(graph)
        visualization = self.visualize(graph)
        
        return {
            'mode': self.name,
            'parsed': parsed,
            'graph': graph,
            'analysis': analysis,
            'visualization': visualization
        }
