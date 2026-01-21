"""
混合检索和 RRF 融合模块
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from loguru import logger
from pymilvus import Collection


class SearchResult:
    """检索结果"""

    def __init__(
        self,
        doc_id: str,
        chunk_id: str,
        title: str,
        content: str,
        score: float,
        distance: Optional[float] = None,
        metadata: Optional[dict] = None
    ):
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.title = title
        self.content = content
        self.score = score
        self.distance = distance
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "score": self.score,
            "distance": self.distance,
            "metadata": self.metadata
        }

    @classmethod
    def from_milvus_hit(cls, hit) -> "SearchResult":
        """从 Milvus 命中结果创建"""
        return cls(
            doc_id=hit.entity.get("doc_id"),
            chunk_id=hit.entity.get("chunk_id"),
            title=hit.entity.get("title"),
            content=hit.entity.get("content"),
            score=hit.score,
            distance=hit.distance,
            metadata=hit.entity.get("metadata")
        )


class HybridSearcher:
    """
    混合检索器

    支持 Dense 向量检索、Sparse 向量检索和混合检索（RRF 融合）
    """

    def __init__(
        self,
        collection: Collection,
        dense_search_field: str = "dense_vector",
        sparse_search_field: str = "sparse_vector"
    ):
        """
        初始化混合检索器

        Args:
            collection: Milvus Collection 对象
            dense_search_field: Dense 向量字段名
            sparse_search_field: Sparse 向量字段名
        """
        self.collection = collection
        self.dense_search_field = dense_search_field
        self.sparse_search_field = sparse_search_field

        logger.info("混合检索器初始化完成")

    def dense_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        expr: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Dense 向量检索

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            expr: 过滤表达式

        Returns:
            List[SearchResult]: 检索结果列表
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

        return [SearchResult.from_milvus_hit(hit) for hit in results[0]]

    def sparse_search(
        self,
        query_sparse: Dict[int, float],
        top_k: int = 10,
        expr: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Sparse 向量检索

        Args:
            query_sparse: 查询稀疏向量
            top_k: 返回结果数量
            expr: 过滤表达式

        Returns:
            List[SearchResult]: 检索结果列表
        """
        search_params = {
            "metric_type": "IP",
            "params": {"drop_ratio_search": 0.1}
        }

        results = self.collection.search(
            data=[query_sparse],
            anns_field=self.sparse_search_field,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["doc_id", "chunk_id", "title", "content", "metadata"]
        )

        return [SearchResult.from_milvus_hit(hit) for hit in results[0]]

    def hybrid_search(
        self,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        top_k: int = 10,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> List[SearchResult]:
        """
        混合检索（Dense + Sparse）

        Args:
            query_dense: Dense 查询向量
            query_sparse: Sparse 查询向量
            top_k: 返回结果数量
            fusion_method: 融合方法，"rrf" 或 "weighted"
            rrf_k: RRF 参数 k
            dense_weight: Dense 权重（用于 weighted 融合）
            sparse_weight: Sparse 权重（用于 weighted 融合）

        Returns:
            List[SearchResult]: 融合后的检索结果列表
        """
        # 执行两路检索，获取更多结果用于融合
        dense_results = self.dense_search(query_dense, top_k=top_k * 2)
        sparse_results = self.sparse_search(query_sparse, top_k=top_k * 2)

        # 结果融合
        if fusion_method == "rrf":
            fused_results = self._rrf_fusion(dense_results, sparse_results, top_k, rrf_k)
        else:
            fused_results = self._weighted_fusion(
                dense_results, sparse_results, top_k, dense_weight, sparse_weight
            )

        logger.debug(f"混合检索完成: Dense={len(dense_results)}, Sparse={len(sparse_results)}, Fused={len(fused_results)}")
        return fused_results

    def _rrf_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int,
        k: int = 60
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF) 算法

        score = sum(1 / (k + rank))

        Args:
            dense_results: Dense 检索结果
            sparse_results: Sparse 检索结果
            top_k: 返回结果数量
            k: RRF 参数

        Returns:
            List[SearchResult]: 融合后的结果
        """
        scores = defaultdict(float)
        doc_data: Dict[str, SearchResult] = {}

        # 处理 Dense 结果
        for rank, result in enumerate(dense_results, 1):
            doc_id = result.chunk_id  # 使用 chunk_id 唯一标识
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in doc_data:
                doc_data[doc_id] = result

        # 处理 Sparse 结果
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result.chunk_id
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in doc_data:
                doc_data[doc_id] = result

        # 排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 创建融合结果
        fused_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            result = doc_data[doc_id]
            # 更新分数为 RRF 分数
            result.score = rrf_score
            result.metadata["fusion_score"] = rrf_score
            result.metadata["fusion_method"] = "rrf"
            fused_results.append(result)

        return fused_results

    def _weighted_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int,
        dense_weight: float,
        sparse_weight: float
    ) -> List[SearchResult]:
        """
        加权融合算法

        Args:
            dense_results: Dense 检索结果
            sparse_results: Sparse 检索结果
            top_k: 返回结果数量
            dense_weight: Dense 权重
            sparse_weight: Sparse 权重

        Returns:
            List[SearchResult]: 融合后的结果
        """
        scores = defaultdict(float)
        doc_data: Dict[str, SearchResult] = {}

        # 获取分数并归一化
        dense_scores = {r.chunk_id: r.score for r in dense_results}
        sparse_scores = {r.chunk_id: r.score for r in sparse_results}

        max_dense = max(dense_scores.values()) if dense_scores else 1.0
        max_sparse = max(sparse_scores.values()) if sparse_scores else 1.0

        # 融合 Dense 结果
        for result in dense_results:
            doc_id = result.chunk_id
            dense_norm = result.score / max_dense
            scores[doc_id] += dense_weight * dense_norm
            if doc_id not in doc_data:
                doc_data[doc_id] = result

        # 融合 Sparse 结果
        for result in sparse_results:
            doc_id = result.chunk_id
            sparse_norm = result.score / max_sparse
            if doc_id in scores:
                scores[doc_id] += sparse_weight * sparse_norm
            else:
                scores[doc_id] = sparse_weight * sparse_norm
                doc_data[doc_id] = result

        # 排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 创建融合结果
        fused_results = []
        for doc_id, fusion_score in sorted_docs[:top_k]:
            result = doc_data[doc_id]
            result.score = fusion_score
            result.metadata["fusion_score"] = fusion_score
            result.metadata["fusion_method"] = "weighted"
            fused_results.append(result)

        return fused_results


if __name__ == "__main__":
    # 测试代码
    print("HybridSearcher 测试")
    print("请在实际使用时连接到 Milvus Collection 后测试")
    print()
    print("RRF 融合算法示例:")
    print("score = 1 / (k + rank_dense) + 1 / (k + rank_sparse)")
