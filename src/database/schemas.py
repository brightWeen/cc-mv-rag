"""
Milvus 和 Elasticsearch Schema 定义
"""

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)

# Milvus Collection Schema
MILVUS_SCHEMA = CollectionSchema(
    fields=[
        # 主键
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),
        # 文档信息
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        # Dense 向量 (GLM Embedding)
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        # Sparse 向量 (BM25)
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ],
    description="Document chunks with dense and sparse vectors for hybrid search",
    enable_dynamic_field=True
)

# Milvus Index 配置
# 注意: Milvus Lite 本地模式只支持 AUTOINDEX, FLAT, IVF_FLAT
DENSE_INDEX_CONFIG = {
    "index_type": "AUTOINDEX",  # 本地模式使用 AUTOINDEX
    "metric_type": "IP",   # Inner Product (余弦相似度需要归一化)
    "params": {}
}

SPARSE_INDEX_CONFIG = {
    "index_type": "SPARSE_INVERTED_INDEX",  # 稀疏倒排索引
    "metric_type": "IP",   # Inner Product
    "params": {
        "drop_ratio_build": 0.1  # 构建时丢弃比例
    }
}

# Elasticsearch Index Mapping
ES_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            # 文档字段
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "content": {
                "type": "text",
                "analyzer": "standard"
            },
            "metadata": {"type": "object"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "standard": {
                    "type": "standard"
                }
            }
        }
    }
}


def get_milvus_schema(dense_dim: int = 1024) -> CollectionSchema:
    """
    获取 Milvus Collection Schema

    Args:
        dense_dim: Dense 向量维度

    Returns:
        CollectionSchema: Milvus Schema
    """
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ],
        description="Document chunks with dense and sparse vectors for hybrid search",
        enable_dynamic_field=True
    )


def get_es_mapping(use_ik_analyzer: bool = False) -> dict:
    """
    获取 Elasticsearch Index Mapping

    Args:
        use_ik_analyzer: 是否使用 IK 中文分词器

    Returns:
        dict: ES Mapping
    """
    analyzer = "ik_max_word" if use_ik_analyzer else "standard"
    search_analyzer = "ik_smart" if use_ik_analyzer else "standard"

    return {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": analyzer,
                    "search_analyzer": search_analyzer,
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "content": {
                    "type": "text",
                    "analyzer": analyzer,
                    "search_analyzer": search_analyzer
                },
                "metadata": {"type": "object"},
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "standard": {
                        "type": "standard"
                    }
                }
            }
        }
    }


if __name__ == "__main__":
    # 测试代码
    schema = get_milvus_schema(dense_dim=1024)
    print("Milvus Schema Fields:")
    for field in schema.fields:
        print(f"  {field.name}: {field.dtype}")

    mapping = get_es_mapping(use_ik_analyzer=True)
    print("\nES Mapping:")
    print(mapping)
