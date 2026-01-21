#!/usr/bin/env python3
"""
索引构建脚本

构建 Milvus 和 Elasticsearch 索引
"""

import json
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config.settings import get_config
from src.models.dense_embedding import GLMEmbedding
from src.models.sparse_embedding import BM25Sparse
from src.pipeline.chunker import DocumentChunker
from src.database.milvus_client import MilvusClient
from src.database.es_client import ESClient


def load_documents(data_path: Path) -> list:
    """加载文档数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_documents(data_dir: Path) -> list:
    """加载所有文档数据（技术文档 + 产品文档）"""
    all_docs = []

    # 加载技术文档
    sample_docs = data_dir / "raw" / "sample_docs.json"
    if sample_docs.exists():
        with open(sample_docs, "r", encoding="utf-8") as f:
            all_docs.extend(json.load(f))

    # 加载产品文档
    products = data_dir / "raw" / "products.json"
    if products.exists():
        with open(products, "r", encoding="utf-8") as f:
            all_docs.extend(json.load(f))

    # 如果合并文档存在，直接使用
    all_docs_file = data_dir / "raw" / "all_docs.json"
    if all_docs_file.exists():
        with open(all_docs_file, "r", encoding="utf-8") as f:
            return json.load(f)

    return all_docs


def load_queries(data_path: Path) -> list:
    """加载查询数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_milvus_index(config, documents: list, chunks: list):
    """构建 Milvus 索引"""
    logger.info("=" * 50)
    logger.info("开始构建 Milvus 索引")
    logger.info("=" * 50)

    # 准备文本列表（用于生成向量）
    texts = [chunk.content for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    doc_ids = [chunk.doc_id for chunk in chunks]
    titles = [chunk.title for chunk in chunks]
    metadata_list = [chunk.metadata for chunk in chunks]

    # 先初始化 GLM 模型来检测向量维度
    logger.info("初始化 GLM Embedding 模型并检测向量维度...")
    dense_model = GLMEmbedding(
        api_key=config.glm.api_key,
        model=config.glm.model,
        auto_detect_dim=True
    )
    actual_dim = dense_model.dimension
    logger.info(f"检测到的向量维度: {actual_dim}")

    # 初始化 Milvus 客户端（使用实际检测到的维度）
    milvus_client = MilvusClient(
        uri=config.milvus.uri,
        collection_name=config.milvus.collection_name,
        dense_dim=actual_dim
    )

    # 创建 Collection
    milvus_client.create_collection(drop_existing=True)

    # 生成 Dense 向量
    logger.info(f"正在生成 {len(texts)} 个 Dense 向量...")
    dense_vectors = dense_model.encode(texts, batch_size=config.glm.batch_size)
    logger.info(f"Dense 向量生成完成: {dense_vectors.shape}")

    # 生成 Sparse 向量
    logger.info(f"正在生成 Sparse 向量...")
    sparse_model = BM25Sparse(k1=1.5, b=0.75)
    sparse_model.fit(texts)
    sparse_vectors = sparse_model.encode_documents()
    logger.info(f"Sparse 向量生成完成: {len(sparse_vectors)} 个")

    # 插入 Milvus
    logger.info("正在插入数据到 Milvus...")
    milvus_client.insert_data(
        ids=chunk_ids,
        doc_ids=doc_ids,
        chunk_ids=chunk_ids,
        titles=titles,
        contents=texts,
        metadata_list=metadata_list,
        dense_vectors=dense_vectors.tolist(),
        sparse_vectors=sparse_vectors
    )

    # 加载到内存
    milvus_client.load_collection()

    # 获取统计信息
    stats = milvus_client.get_stats()
    logger.info(f"Milvus 索引构建完成: {stats}")

    milvus_client.disconnect()

    return dense_model, sparse_model


def build_es_index(config, chunks: list):
    """构建 ES 索引"""
    logger.info("=" * 50)
    logger.info("开始构建 ES 索引")
    logger.info("=" * 50)

    # 初始化 ES 客户端
    try:
        es_client = ESClient(
            host=config.elasticsearch.host,
            port=config.elasticsearch.port,
            index_name=config.elasticsearch.index_name
        )
    except Exception as e:
        logger.error(f"无法连接到 Elasticsearch: {e}")
        logger.warning("请确保 Elasticsearch 已启动:")
        logger.warning("  docker start elasticsearch")
        logger.warning("  或使用: docker run -d -p 9200:9200 -e \"discovery.type=single-node\" -e \"xpack.security.enabled=false\" docker.elastic.co/elasticsearch/elasticsearch:8.11.0")
        return None

    # 创建索引
    try:
        es_client.create_index(drop_existing=True, use_ik_analyzer=False)
        logger.info("ES 索引创建成功")
    except Exception as e:
        logger.error(f"ES 索引创建失败: {e}")
        es_client.close()
        return None

    # 准备文档
    documents = []
    for chunk in chunks:
        doc = {
            "id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "content": chunk.content,
        }
        documents.append(doc)

    # 插入 ES
    logger.info("正在插入数据到 ES...")
    es_client.insert_documents(documents)

    # 获取统计信息
    stats = es_client.get_stats()
    logger.info(f"ES 索引构建完成: {stats}")

    es_client.close()

    return es_client


def save_chunks(chunks: list, output_path: Path):
    """保存分块数据"""
    chunks_data = [chunk.to_dict() for chunk in chunks]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    logger.info(f"分块数据已保存: {output_path}")


def main():
    """主函数"""
    # 加载配置
    config = get_config()

    # 配置日志
    logger.add(
        config.logging.file,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        level=config.logging.level
    )

    logger.info("开始构建索引")
    logger.info(f"项目: {config.project}")

    # 项目路径
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    # 加载所有文档数据
    all_docs_file = data_dir / "raw" / "all_docs.json"
    logger.info(f"加载文档数据: {all_docs_file}")
    documents = load_all_documents(data_dir)
    logger.info(f"文档数量: {len(documents)}")

    # 加载查询数据
    test_queries_file = data_dir / "queries" / "test_queries.json"
    logger.info(f"加载查询数据: {test_queries_file}")
    queries = load_queries(test_queries_file)
    logger.info(f"查询数量: {len(queries)}")

    # 加载混合查询（如果存在）
    mixed_queries_file = data_dir / "queries" / "mixed_queries.json"
    if mixed_queries_file.exists():
        mixed_queries = load_queries(mixed_queries_file)
        queries.extend(mixed_queries)
        logger.info(f"加载混合查询: {len(mixed_queries)} 个")
        logger.info(f"总查询数: {len(queries)}")

    # 文档分块
    logger.info("=" * 50)
    logger.info("开始文档分块")
    logger.info("=" * 50)

    chunker = DocumentChunker(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        max_chunks_per_doc=config.chunking.max_chunks_per_doc
    )

    chunks = chunker.chunk(documents)
    logger.info(f"分块完成: {len(documents)} 个文档 -> {len(chunks)} 个块")

    # 保存分块数据
    chunks_file = data_dir / "processed" / "chunks.json"
    save_chunks(chunks, chunks_file)

    # 文档分块
    logger.info("=" * 50)
    logger.info("开始文档分块")
    logger.info("=" * 50)

    chunker = DocumentChunker(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        max_chunks_per_doc=config.chunking.max_chunks_per_doc
    )

    chunks = chunker.chunk(documents)
    logger.info(f"分块完成: {len(documents)} 个文档 -> {len(chunks)} 个块")

    # 保存分块数据
    save_chunks(chunks, chunks_file)

    # 构建 Milvus 索引
    dense_model, sparse_model = build_milvus_index(config, documents, chunks)

    # 构建 ES 索引
    es_client = build_es_index(config, chunks)

    logger.info("=" * 50)
    logger.info("索引构建完成！")
    logger.info("=" * 50)
    logger.info(f"Milvus Collection: {config.milvus.uri}")
    logger.info(f"ES Index: {config.elasticsearch.host}:{config.elasticsearch.port}/{config.elasticsearch.index_name}")
    logger.info(f"\n下一步: 执行检索脚本")
    logger.info(f"  python3 scripts/03_run_search.py")


if __name__ == "__main__":
    main()
