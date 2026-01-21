"""
配置管理模块
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class GLMConfig(BaseModel):
    """GLM Embedding 配置"""
    api_key: str
    model: str = "embedding-3"
    dimension: int = 1024
    batch_size: int = 10


class MilvusConfig(BaseModel):
    """Milvus 配置"""
    uri: str = "milvus_lite.db"
    collection_name: str = "doc_chunks"
    dense_vector_field: str = "dense_vector"
    sparse_vector_field: str = "sparse_vector"


class ESConfig(BaseModel):
    """Elasticsearch 配置"""
    host: str = "localhost"
    port: int = 9200
    username: str = ""
    password: str = ""
    index_name: str = "doc_chunks"
    verify_certs: bool = False


class ChunkingConfig(BaseModel):
    """文档分块配置"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks_per_doc: int = 50
    separator: str = "\n\n"


class DenseSearchConfig(BaseModel):
    """Dense 检索配置"""
    ef: int = 256
    metric_type: str = "IP"


class SparseSearchConfig(BaseModel):
    """Sparse 检索配置"""
    drop_ratio: float = 0.1
    metric_type: str = "IP"


class HybridSearchConfig(BaseModel):
    """混合检索配置"""
    fusion_method: str = "rrf"  # "rrf" or "weighted"
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5


class SearchConfig(BaseModel):
    """检索配置"""
    default_top_k: int = 10
    dense_search: DenseSearchConfig = Field(default_factory=DenseSearchConfig)
    sparse_search: SparseSearchConfig = Field(default_factory=SparseSearchConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)


class EvaluationConfig(BaseModel):
    """评估配置"""
    k_values: list = Field(default_factory=lambda: [1, 3, 5, 10])
    metrics: list = Field(default_factory=lambda: ["recall", "precision", "mrr", "ndcg", "map"])


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    file: str = "outputs/logs/app.log"
    rotation: str = "100 MB"
    retention: str = "7 days"


class Config(BaseModel):
    """全局配置"""
    project: dict
    glm: GLMConfig
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    elasticsearch: ESConfig = Field(default_factory=ESConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    class Config:
        arbitrary_types_allowed = True


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，默认为项目根目录下的 config.yaml

    Returns:
        Config: 配置对象
    """
    # 加载环境变量
    load_dotenv()

    # 默认配置文件路径
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    # 读取 YAML 配置
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # 替换环境变量
    def replace_env(obj):
        if isinstance(obj, str):
            # 替换 ${VAR_NAME} 格式的环境变量
            if obj.startswith("${") and obj.endswith("}"):
                var_name = obj[2:-1]
                return os.getenv(var_name, obj)
            return obj
        elif isinstance(obj, dict):
            return {k: replace_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_env(item) for item in obj]
        return obj

    config_data = replace_env(config_data)

    return Config(**config_data)


# 全局配置实例
_global_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


if __name__ == "__main__":
    # 测试配置加载
    config = get_config()
    print(f"项目: {config.project}")
    print(f"GLM 模型: {config.glm.model}, 维度: {config.glm.dimension}")
    print(f"Milvus Collection: {config.milvus.collection_name}")
    print(f"ES Index: {config.elasticsearch.index_name}")
