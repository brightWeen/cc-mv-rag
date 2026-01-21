#!/usr/bin/env python3
"""
检索执行脚本

执行 Dense、Sparse、Hybrid 和 ES 检索
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from tqdm import tqdm

from src.config.settings import get_config
from src.models.dense_embedding import GLMEmbedding
from src.models.sparse_embedding import BM25Sparse
from src.search.hybrid_search import HybridSearcher
from src.search.es_mv_hybrid import ESMVHybridSearcher
from src.database.milvus_client import MilvusClient
from src.database.es_client import ESClient


def load_queries(data_path: Path) -> list:
    """加载查询数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks(data_path: Path) -> list:
    """加载分块数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def milvus_dense_search(
    hybrid_searcher: HybridSearcher,
    dense_model: GLMEmbedding,
    queries: list,
    top_k: int
) -> Dict[str, List[str]]:
    """执行 Dense 检索"""
    logger.info("=" * 50)
    logger.info("执行 Dense 向量检索")
    logger.info("=" * 50)

    results = {}

    for query_item in tqdm(queries, desc="Dense 检索"):
        query_id = query_item["query_id"]
        query_text = query_item["query"]

        # 生成查询向量
        query_vector = dense_model.encode_single(query_text)

        # 执行检索
        search_results = hybrid_searcher.dense_search(query_vector, top_k=top_k)

        # 保存结果
        results[query_id] = [r.doc_id for r in search_results]

    logger.info(f"Dense 检索完成: {len(results)} 个查询")
    return results


def milvus_sparse_search(
    hybrid_searcher: HybridSearcher,
    sparse_model: BM25Sparse,
    queries: list,
    top_k: int
) -> Dict[str, List[str]]:
    """执行 Sparse 检索"""
    logger.info("=" * 50)
    logger.info("执行 Sparse 向量检索")
    logger.info("=" * 50)

    results = {}

    for query_item in tqdm(queries, desc="Sparse 检索"):
        query_id = query_item["query_id"]
        query_text = query_item["query"]

        # 生成稀疏查询向量
        query_sparse = sparse_model.encode_query(query_text)

        # 执行检索
        search_results = hybrid_searcher.sparse_search(query_sparse, top_k=top_k)

        # 保存结果
        results[query_id] = [r.doc_id for r in search_results]

    logger.info(f"Sparse 检索完成: {len(results)} 个查询")
    return results


def milvus_hybrid_search(
    hybrid_searcher: HybridSearcher,
    dense_model: GLMEmbedding,
    sparse_model: BM25Sparse,
    queries: list,
    top_k: int,
    fusion_method: str = "rrf",
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4
) -> Dict[str, List[str]]:
    """执行混合检索"""
    logger.info("=" * 50)
    logger.info(f"执行混合检索 (融合方法: {fusion_method}, Dense权重: {dense_weight}, Sparse权重: {sparse_weight})")
    logger.info("=" * 50)

    results = {}

    for query_item in tqdm(queries, desc="Hybrid 检索"):
        query_id = query_item["query_id"]
        query_text = query_item["query"]

        # 生成查询向量
        query_dense = dense_model.encode_single(query_text)
        query_sparse = sparse_model.encode_query(query_text)

        # 执行混合检索
        search_results = hybrid_searcher.hybrid_search(
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=top_k,
            fusion_method=fusion_method,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )

        # 保存结果
        results[query_id] = [r.doc_id for r in search_results]

    logger.info(f"混合检索完成: {len(results)} 个查询")
    return results


def es_bm25_search(
    es_client: ESClient,
    queries: list,
    top_k: int
) -> Dict[str, List[str]]:
    """执行 ES BM25 检索"""
    logger.info("=" * 50)
    logger.info("执行 ES BM25 检索")
    logger.info("=" * 50)

    results = {}

    for query_item in tqdm(queries, desc="ES BM25 检索"):
        query_id = query_item["query_id"]
        query_text = query_item["query"]

        try:
            # 执行检索
            search_results = es_client.search(query_text, top_k=top_k)

            # 保存结果
            results[query_id] = [r["doc_id"] for r in search_results]
        except Exception as e:
            logger.warning(f"ES 检索失败: {e}")
            results[query_id] = []

    logger.info(f"ES BM25 检索完成: {len(results)} 个查询")
    return results


def es_mv_hybrid_search(
    es_mv_searcher: ESMVHybridSearcher,
    dense_model: GLMEmbedding,
    queries: list,
    top_k: int,
    fusion_method: str = "rrf"
) -> Dict[str, List[str]]:
    """执行 ES + MV 混合检索（应用层融合）"""
    logger.info("=" * 50)
    logger.info(f"执行 ES + MV 混合检索 (融合方法: {fusion_method})")
    logger.info("=" * 50)

    results = {}

    for query_item in tqdm(queries, desc="ES+MV Hybrid 检索"):
        query_id = query_item["query_id"]
        query_text = query_item["query"]

        # 生成 Dense 查询向量
        query_dense = dense_model.encode_single(query_text)

        try:
            # 执行混合检索
            search_results = es_mv_searcher.hybrid_search(
                query=query_text,
                query_dense=query_dense,
                top_k=top_k,
                fusion_method=fusion_method
            )

            # 保存结果
            results[query_id] = [r["doc_id"] for r in search_results]
        except Exception as e:
            logger.warning(f"ES+MV 混合检索失败: {e}")
            results[query_id] = []

    logger.info(f"ES+MV 混合检索完成: {len(results)} 个查询")
    return results


def save_results(results: dict, output_path: Path):
    """保存检索结果"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"检索结果已保存: {output_path}")


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

    logger.info("开始执行检索")
    logger.info(f"项目: {config.project}")

    # 项目路径
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载查询数据（包含 test_queries 和 mixed_queries）
    queries_file = data_dir / "queries" / "test_queries.json"
    logger.info(f"加载查询数据: {queries_file}")
    queries = load_queries(queries_file)
    logger.info(f"基础查询数量: {len(queries)}")

    # 加载混合查询
    mixed_queries_file = data_dir / "queries" / "mixed_queries.json"
    if mixed_queries_file.exists():
        mixed_queries = load_queries(mixed_queries_file)
        queries.extend(mixed_queries)
        logger.info(f"混合查询数量: {len(mixed_queries)}")
        logger.info(f"总查询数量: {len(queries)}")

    # 加载分块数据（用于训练 Sparse 模型）
    chunks_file = data_dir / "processed" / "chunks.json"
    logger.info(f"加载分块数据: {chunks_file}")
    chunks = load_chunks(chunks_file)
    logger.info(f"分块数量: {len(chunks)}")

    # 初始化 Milvus
    logger.info("连接 Milvus...")
    milvus_client = MilvusClient(
        uri=config.milvus.uri,
        collection_name=config.milvus.collection_name,
        dense_dim=config.glm.dimension
    )
    milvus_client.load_collection()
    collection = milvus_client.get_collection()

    # 初始化 Hybrid Searcher
    hybrid_searcher = HybridSearcher(
        collection=collection,
        dense_search_field=config.milvus.dense_vector_field,
        sparse_search_field=config.milvus.sparse_vector_field
    )

    # 初始化 Dense 模型
    logger.info("初始化 GLM Embedding 模型...")
    dense_model = GLMEmbedding(
        api_key=config.glm.api_key,
        model=config.glm.model
    )

    # 初始化 Sparse 模型
    logger.info("初始化 BM25 Sparse 模型...")
    sparse_model = BM25Sparse(k1=1.5, b=0.75)
    texts = [chunk["content"] for chunk in chunks]
    sparse_model.fit(texts)
    logger.info(f"BM25 模型训练完成，词汇表大小: {sparse_model.get_vocab_size()}")

    # 初始化 ES 客户端
    es_client = None
    es_mv_searcher = None
    try:
        logger.info("连接 Elasticsearch...")
        es_client = ESClient(
            host=config.elasticsearch.host,
            port=config.elasticsearch.port,
            index_name=config.elasticsearch.index_name
        )
        # 初始化 ES+MV 混合检索器
        es_mv_searcher = ESMVHybridSearcher(
            es_client=es_client,
            milvus_collection=collection,
            dense_search_field=config.milvus.dense_vector_field
        )
    except Exception as e:
        logger.warning(f"无法连接 Elasticsearch: {e}")
        logger.warning("将跳过 ES 相关检索")

    # 执行检索
    top_k = config.search.default_top_k
    all_results = {}

    # Dense 检索
    all_results["dense"] = milvus_dense_search(
        hybrid_searcher, dense_model, queries, top_k
    )
    save_results(all_results["dense"], output_dir / "dense_results.json")

    # Sparse 检索
    all_results["sparse"] = milvus_sparse_search(
        hybrid_searcher, sparse_model, queries, top_k
    )
    save_results(all_results["sparse"], output_dir / "sparse_results.json")

    # Hybrid 检索 (RRF)
    all_results["hybrid_rrf"] = milvus_hybrid_search(
        hybrid_searcher, dense_model, sparse_model, queries, top_k, fusion_method="rrf"
    )
    save_results(all_results["hybrid_rrf"], output_dir / "hybrid_rrf_results.json")

    # Hybrid 检索 (Weighted - Dense 优先)
    all_results["hybrid_weighted"] = milvus_hybrid_search(
        hybrid_searcher, dense_model, sparse_model, queries, top_k, fusion_method="weighted"
    )
    save_results(all_results["hybrid_weighted"], output_dir / "hybrid_weighted_results.json")

    # ES BM25 检索
    if es_client:
        all_results["es_bm25"] = es_bm25_search(
            es_client, queries, top_k
        )
        save_results(all_results["es_bm25"], output_dir / "es_bm25_results.json")

    # ES + MV 混合检索 (应用层 RRF 融合)
    if es_mv_searcher:
        all_results["es_mv_hybrid_rrf"] = es_mv_hybrid_search(
            es_mv_searcher, dense_model, queries, top_k, fusion_method="rrf"
        )
        save_results(all_results["es_mv_hybrid_rrf"], output_dir / "es_mv_hybrid_rrf_results.json")

    # 关闭 ES 连接
    if es_client:
        es_client.close()

    # 保存所有结果
    save_results(all_results, output_dir / "all_results.json")

    logger.info("=" * 50)
    logger.info("检索完成！")
    logger.info("=" * 50)
    logger.info(f"结果已保存到: {output_dir}")
    logger.info(f"\n下一步: 执行评估脚本")
    logger.info(f"  python3 scripts/04_evaluate.py")

    # 关闭连接
    milvus_client.disconnect()


if __name__ == "__main__":
    main()
