"""
ES + MV 混合检索器

方案 B: ES 全文检索 + MV Dense 向量检索
- 文本检索: Elasticsearch (全文检索能力)
- 语义检索: Milvus Dense (GLM Embedding)
- 结果融合: 应用层 RRF
"""

from collections import defaultdict
from typing import Dict, List, Optional

from loguru import logger
from pymilvus import Collection

from src.database.es_client import ESClient


class ESMVHybridSearcher:
    """
    ES + MV 混合检索器

    使用 Elasticsearch 进行全文检索
    使用 Milvus 进行 Dense 向量检索
    在应用层进行 RRF 融合
    """

    def __init__(
        self,
        es_client: ESClient,
        milvus_collection: Collection,
        dense_search_field: str = "dense_vector"
    ):
        """
        初始化 ES + MV 混合检索器

        Args:
            es_client: Elasticsearch 客户端
            milvus_collection: Milvus Collection 对象
            dense_search_field: Dense 向量字段名
        """
        self.es_client = es_client
        self.collection = milvus_collection
        self.dense_search_field = dense_search_field

        logger.info("ES+MV 混合检索器初始化完成")

    def es_fulltext_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Elasticsearch 全文检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[Dict]: 检索结果列表
        """
        try:
            results = self.es_client.search(query, top_k=top_k)

            # 转换为统一格式
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "doc_id": r["doc_id"],
                    "chunk_id": r["chunk_id"],
                    "title": r["title"],
                    "content": r["content"],
                    "score": r["score"]
                })

            return formatted_results

        except Exception as e:
            logger.error(f"ES 检索失败: {e}")
            return []

    def dense_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        expr: Optional[str] = None
    ) -> List[Dict]:
        """
        Milvus Dense 向量检索

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            expr: 过滤表达式

        Returns:
            List[Dict]: 检索结果列表
        """
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 256}
        }

        results = self.collection.search(
            data=[query_vector],
            anns_field=self.dense_search_field,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["doc_id", "chunk_id", "title", "content", "metadata"]
        )

        # 转换为统一格式
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "doc_id": hit.entity.get("doc_id"),
                "chunk_id": hit.entity.get("chunk_id"),
                "title": hit.entity.get("title"),
                "content": hit.entity.get("content"),
                "score": hit.score
            })

        return formatted_results

    def hybrid_search(
        self,
        query: str,
        query_dense: List[float],
        top_k: int = 10,
        rrf_k: int = 60,
        es_weight: float = 0.5,
        dense_weight: float = 0.5,
        fusion_method: str = "rrf"
    ) -> List[Dict]:
        """
        ES + MV 混合检索（应用层 RRF 融合）

        Args:
            query: 原始查询文本（用于 ES）
            query_dense: Dense 查询向量（用于 MV）
            top_k: 返回结果数量
            rrf_k: RRF 参数 k
            es_weight: ES 权重（用于 weighted 融合）
            dense_weight: Dense 权重（用于 weighted 融合）
            fusion_method: 融合方法，"rrf" 或 "weighted"

        Returns:
            List[Dict]: 融合后的检索结果列表
        """
        # 执行两路检索，获取更多结果用于融合
        es_results = self.es_fulltext_search(query, top_k=top_k * 2)
        dense_results = self.dense_search(query_dense, top_k=top_k * 2)

        # 结果融合
        if fusion_method == "rrf":
            fused_results = self._rrf_fusion(es_results, dense_results, top_k, rrf_k)
        else:
            fused_results = self._weighted_fusion(
                es_results, dense_results, top_k, es_weight, dense_weight
            )

        logger.debug(f"ES+MV 混合检索完成: ES={len(es_results)}, Dense={len(dense_results)}, Fused={len(fused_results)}")
        return fused_results

    def _rrf_fusion(
        self,
        es_results: List[Dict],
        dense_results: List[Dict],
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """
        应用层 RRF 融合算法

        score = sum(1 / (k + rank))

        Args:
            es_results: ES 检索结果
            dense_results: Dense 检索结果
            top_k: 返回结果数量
            k: RRF 参数

        Returns:
            List[Dict]: 融合后的结果
        """
        scores = defaultdict(float)
        doc_data: Dict[str, Dict] = {}

        # 处理 ES 结果
        for rank, result in enumerate(es_results, 1):
            doc_id = result["chunk_id"]
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in doc_data:
                doc_data[doc_id] = result

        # 处理 Dense 结果
        for rank, result in enumerate(dense_results, 1):
            doc_id = result["chunk_id"]
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in doc_data:
                doc_data[doc_id] = result

        # 排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 创建融合结果
        fused_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            result = doc_data[doc_id].copy()
            result["fusion_score"] = rrf_score
            result["fusion_method"] = "es_mv_rrf"
            result["source"] = "ES+MV"
            fused_results.append(result)

        return fused_results

    def _weighted_fusion(
        self,
        es_results: List[Dict],
        dense_results: List[Dict],
        top_k: int,
        es_weight: float,
        dense_weight: float
    ) -> List[Dict]:
        """
        加权融合算法

        Args:
            es_results: ES 检索结果
            dense_results: Dense 检索结果
            top_k: 返回结果数量
            es_weight: ES 权重
            dense_weight: Dense 权重

        Returns:
            List[Dict]: 融合后的结果
        """
        scores = defaultdict(float)
        doc_data: Dict[str, Dict] = {}

        # 获取分数并归一化
        es_scores = {r["chunk_id"]: r["score"] for r in es_results}
        dense_scores = {r["chunk_id"]: r["score"] for r in dense_results}

        max_es = max(es_scores.values()) if es_scores else 1.0
        max_dense = max(dense_scores.values()) if dense_scores else 1.0

        # 融合 ES 结果
        for result in es_results:
            doc_id = result["chunk_id"]
            es_norm = result["score"] / max_es
            scores[doc_id] += es_weight * es_norm
            if doc_id not in doc_data:
                doc_data[doc_id] = result

        # 融合 Dense 结果
        for result in dense_results:
            doc_id = result["chunk_id"]
            dense_norm = result["score"] / max_dense
            if doc_id in scores:
                scores[doc_id] += dense_weight * dense_norm
            else:
                scores[doc_id] = dense_weight * dense_norm
                doc_data[doc_id] = result

        # 排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 创建融合结果
        fused_results = []
        for doc_id, fusion_score in sorted_docs[:top_k]:
            result = doc_data[doc_id].copy()
            result["fusion_score"] = fusion_score
            result["fusion_method"] = "es_mv_weighted"
            result["source"] = "ES+MV"
            fused_results.append(result)

        return fused_results


if __name__ == "__main__":
    print("ES+MV 混合检索器模块")
    print("\n方案对比:")
    print("方案 A: MV 单库 - MV Sparse (BM25) + MV Dense (GLM)")
    print("方案 B: ES+MV 双库 - ES 全文检索 + MV Dense (GLM)")
    print("\n差异: 方案 B 使用 ES 的全文检索能力（BM25 + Phrase + Fuzzy + 通配符）")
