"""
SeekDB 混合检索模块 - 使用内置 hybrid_search API
"""

from typing import List, Optional
from loguru import logger

from src.search.hybrid_search import SearchResult


class SeekDBHybridSearcher:
    """SeekDB 混合检索器，使用内置 hybrid_search API"""

    def __init__(self, collection):
        """
        初始化 SeekDB 混合检索器

        Args:
            collection: SeekDB Collection 对象（已配置好 embedding_function）
        """
        self.collection = collection
        logger.info("SeekDB 混合检索器初始化完成")

    def dense_search(
        self,
        query_text: str,
        top_k: int = 10,
        where: Optional[dict] = None
    ) -> List[SearchResult]:
        """
        Dense 向量检索（语义检索）

        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            where: 过滤条件
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where
        )

        return self._format_results(results)

    def sparse_search(
        self,
        query_text: str,
        top_k: int = 10,
        where: Optional[dict] = None
    ) -> List[SearchResult]:
        """
        Sparse 向量检索（全文检索）
        """
        results = self.collection.query(
            query_texts=[""],  # 空查询文本
            n_results=top_k,
            where_document={"$contains": query_text},
            where=where
        )

        return self._format_results(results)

    def hybrid_search(
        self,
        query_text: str,
        top_k: int = 10,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> List[SearchResult]:
        """
        混合检索（使用 SeekDB 内置 hybrid_search API）

        注意：SeekDB 只支持 RRF 融合，不支持加权融合
        """
        # 使用 SeekDB 内置的 hybrid_search
        results = self.collection.hybrid_search(
            query={"where_document": {"$contains": query_text}, "n_results": top_k * 2},
            knn={"query_texts": [query_text], "n_results": top_k * 2},
            rank={"rrf": {}},
            n_results=top_k
        )

        return self._format_hybrid_results(results)

    def _format_results(self, results) -> List[SearchResult]:
        """格式化查询结果"""
        formatted = []
        if not results['ids'][0]:
            return formatted

        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
            # 将距离转换为分数
            if results.get('distances') and results['distances'][0]:
                distance = results['distances'][0][i]
                score = 1.0 / (1.0 + distance)
            else:
                score = 1.0 / (1.0 + i)

            # 从 metadata 中获取 doc_id
            doc_id = metadata.get("doc_id", "")
            if not doc_id:
                # 尝试从 chunk_id 中提取 doc_id
                chunk_id = metadata.get("chunk_id", results['ids'][0][i])
                doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id

            formatted.append(SearchResult(
                doc_id=doc_id,
                chunk_id=metadata.get("chunk_id", results['ids'][0][i]),
                title=metadata.get("title", ""),
                content=results['documents'][0][i] if results.get('documents') else "",
                score=score,
                distance=results['distances'][0][i] if results.get('distances') else None,
                metadata=metadata
            ))
        return formatted

    def _format_hybrid_results(self, results) -> List[SearchResult]:
        """格式化混合检索结果"""
        formatted = []
        if not results['ids'][0]:
            return formatted

        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i] if results.get('metadatas') else {}

            # hybrid_search 返回的是融合后的排名
            # 使用排名作为分数（排名越前分数越高）
            score = 1.0 / (1.0 + i)

            # 从 metadata 中获取 doc_id
            doc_id = metadata.get("doc_id", "")
            if not doc_id:
                chunk_id = metadata.get("chunk_id", results['ids'][0][i])
                doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id

            formatted.append(SearchResult(
                doc_id=doc_id,
                chunk_id=metadata.get("chunk_id", results['ids'][0][i]),
                title=metadata.get("title", ""),
                content=results['documents'][0][i] if results.get('documents') else "",
                score=score,
                metadata=metadata
            ))
        return formatted
