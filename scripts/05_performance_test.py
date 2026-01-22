#!/usr/bin/env python3
"""
性能测试脚本 - 对比不同检索方法的执行时间
"""

import sys
import time
from pathlib import Path
from typing import List, Dict
from statistics import mean, median, stdev
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import load_config
from src.data.qa_dataset import load_qa_dataset
from src.search.milvus_hybrid import MilvusHybridSearcher
from pymilvus import MilvusClient
from src.database.seekdb_client import SeekDBClient
from src.search.seekdb_hybrid import SeekDBHybridSearcher


def measure_search_time(searcher, query_text: str, top_k: int = 10, method_name: str = "", iterations: int = 5) -> Dict:
    """测量单次查询的执行时间"""
    times = []

    for i in range(iterations):
        start = time.perf_counter()

        if hasattr(searcher, 'hybrid_search'):
            # SeekDB 或 Milvus Hybrid
            if method_name == "milvus_weighted":
                results = searcher.hybrid_search(
                    query_text=query_text,
                    top_k=top_k,
                    fusion_method="weighted"
                )
            else:
                results = searcher.hybrid_search(
                    query_text=query_text,
                    top_k=top_k,
                    fusion_method="rrf"
                )
        elif hasattr(searcher, 'dense_search'):
            # Milvus 单独检索
            results = searcher.dense_search(query_text=query_text, top_k=top_k)

        end = time.perf_counter()
        times.append((end - start) * 1000)  # 转换为毫秒

    return {
        "mean": mean(times),
        "std": stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "median": median(times)
    }


def main():
    config = load_config()
    top_k = 10
    iterations = 3  # 每个查询重复次数

    # 加载测试查询（使用前5个查询进行性能测试）
    queries = load_qa_dataset(config.project["queries_path"])[:5]

    logger.info("=" * 60)
    logger.info("性能测试开始")
    logger.info(f"测试查询数: {len(queries)}")
    logger.info(f"每个查询重复: {iterations} 次")
    logger.info("=" * 60)

    results = {}

    # 1. Milvus Dense 检索
    logger.info("\n[1/4] 测试 Milvus Dense 检索...")
    try:
        milvus_client = MilvusClient(
            uri=config.milvus.uri,
            token=config.milvus.token,
        )
        milvus_searcher = MilvusHybridSearcher(
            milvus_client=milvus_client,
            collection_name=config.milvus.dense_collection_name
        )

        times = []
        for query in queries:
            stats = measure_search_time(
                milvus_searcher,
                query["query"],
                top_k,
                "milvus_dense",
                iterations
            )
            times.append(stats)

        results["milvus_dense"] = {
            "avg_ms": mean([t["mean"] for t in times]),
            "std_ms": mean([t["std"] for t in times]),
            "min_ms": min([t["min"] for t in times]),
            "max_ms": max([t["max"] for t in times]),
        }
        logger.info(f"  平均延迟: {results['milvus_dense']['avg_ms']:.2f} ms")
    except Exception as e:
        logger.warning(f"  Milvus Dense 测试失败: {e}")

    # 2. Milvus Hybrid RRF
    logger.info("\n[2/4] 测试 Milvus Hybrid RRF...")
    try:
        milvus_hybrid_searcher = MilvusHybridSearcher(
            milvus_client=milvus_client,
            dense_collection_name=config.milvus.dense_collection_name,
            sparse_collection_name=config.milvus.sparse_collection_name
        )

        times = []
        for query in queries:
            stats = measure_search_time(
                milvus_hybrid_searcher,
                query["query"],
                top_k,
                "milvus_rrf",
                iterations
            )
            times.append(stats)

        results["milvus_hybrid_rrf"] = {
            "avg_ms": mean([t["mean"] for t in times]),
            "std_ms": mean([t["std"] for t in times]),
            "min_ms": min([t["min"] for t in times]),
            "max_ms": max([t["max"] for t in times]),
        }
        logger.info(f"  平均延迟: {results['milvus_hybrid_rrf']['avg_ms']:.2f} ms")
    except Exception as e:
        logger.warning(f"  Milvus Hybrid RRF 测试失败: {e}")

    # 3. Milvus Hybrid Weighted
    logger.info("\n[3/4] 测试 Milvus Hybrid Weighted...")
    try:
        times = []
        for query in queries:
            stats = measure_search_time(
                milvus_hybrid_searcher,
                query["query"],
                top_k,
                "milvus_weighted",
                iterations
            )
            times.append(stats)

        results["milvus_hybrid_weighted"] = {
            "avg_ms": mean([t["mean"] for t in times]),
            "std_ms": mean([t["std"] for t in times]),
            "min_ms": min([t["min"] for t in times]),
            "max_ms": max([t["max"] for t in times]),
        }
        logger.info(f"  平均延迟: {results['milvus_hybrid_weighted']['avg_ms']:.2f} ms")
    except Exception as e:
        logger.warning(f"  Milvus Hybrid Weighted 测试失败: {e}")

    # 4. SeekDB Hybrid RRF
    logger.info("\n[4/4] 测试 SeekDB Hybrid RRF...")
    try:
        seekdb_client = SeekDBClient(
            db_path=config.seekdb.db_path,
            collection_name=config.seekdb.collection_name,
            host=config.seekdb.host,
            port=config.seekdb.port,
            user=config.seekdb.user,
            password=config.seekdb.password,
            use_server=config.seekdb.use_server,
        )
        collection = seekdb_client.get_collection()
        seekdb_searcher = SeekDBHybridSearcher(collection=collection)

        times = []
        for query in queries:
            stats = measure_search_time(
                seekdb_searcher,
                query["query"],
                top_k,
                "seekdb_rrf",
                iterations
            )
            times.append(stats)

        results["seekdb_hybrid_rrf"] = {
            "avg_ms": mean([t["mean"] for t in times]),
            "std_ms": mean([t["std"] for t in times]),
            "min_ms": min([t["min"] for t in times]),
            "max_ms": max([t["max"] for t in times]),
        }
        logger.info(f"  平均延迟: {results['seekdb_hybrid_rrf']['avg_ms']:.2f} ms")
    except Exception as e:
        logger.warning(f"  SeekDB Hybrid RRF 测试失败: {e}")

    # 输出结果
    logger.info("\n" + "=" * 60)
    logger.info("性能测试结果汇总")
    logger.info("=" * 60)

    print("\n┌─────────────────────────┬──────────┬──────────┬──────────┬──────────┐")
    print("│ 方法                    │ 平均(ms) │ 标准差   │ 最小(ms) │ 最大(ms) │")
    print("├─────────────────────────┼──────────┼──────────┼──────────┼──────────┤")

    for name, data in sorted(results.items(), key=lambda x: x[1]["avg_ms"]):
        print(f"│ {name:23s} │ {data['avg_ms']:8.2f} │ {data['std_ms']:8.2f} │ {data['min_ms']:8.2f} │ {data['max_ms']:8.2f} │")

    print("└─────────────────────────┴──────────┴──────────┴──────────┴──────────┘")

    # 计算 QPS
    print("\n【QPS 估算】(基于平均延迟)")
    for name, data in sorted(results.items(), key=lambda x: x[1]["avg_ms"]):
        qps = 1000 / data["avg_ms"]
        print(f"  {name:25s}: {qps:.2f} queries/sec")

    # 计算相对倍数
    if results:
        fastest = min(results.items(), key=lambda x: x[1]["avg_ms"])
        print(f"\n【相对性能】(以 {fastest[0]} 为基准)")
        for name, data in results.items():
            ratio = data["avg_ms"] / fastest[1]["avg_ms"]
            print(f"  {name:25s}: {ratio:.2f}x")


if __name__ == "__main__":
    main()
