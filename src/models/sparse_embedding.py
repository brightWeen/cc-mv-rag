"""
BM25 Sparse Embedding 模型
"""

from collections import defaultdict
from typing import Dict, List, Set

import jieba
from loguru import logger
from rank_bm25 import BM25Okapi
import numpy as np


class BM25Sparse:
    """
    BM25 稀疏向量生成器

    用于将文本转换为稀疏向量格式，支持 Milvus SPARSE_FLOAT_VECTOR 类型
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25 模型

        Args:
            k1: 调节词频饱和度的参数，默认 1.5
            b: 调节文档长度归一化的参数，默认 0.75
        """
        self.k1 = k1
        self.b = b
        self.bm25: BM25Okapi = None
        self.corpus: List[str] = None
        self.tokenized_corpus: List[List[str]] = None
        self.vocab: Dict[str, int] = {}  # 词到索引的映射
        self.idf: Dict[str, float] = {}  # 词的 IDF 值

        logger.info(f"BM25 Sparse 模型初始化完成: k1={k1}, b={b}")

    def fit(self, corpus: List[str]):
        """
        在语料库上训练 BM25 模型

        Args:
            corpus: 文档列表
        """
        self.corpus = corpus

        # 分词
        logger.info("正在对语料库进行分词...")
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]

        # 构建词汇表
        logger.info("正在构建词汇表...")
        tokens = set()
        for doc_tokens in self.tokenized_corpus:
            tokens.update(doc_tokens)

        self.vocab = {token: idx for idx, token in enumerate(sorted(tokens))}
        logger.info(f"词汇表大小: {len(self.vocab)}")

        # 训练 BM25
        logger.info("正在训练 BM25 模型...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

        # 计算 IDF
        self._compute_idf()

        logger.info("BM25 模型训练完成")

    def _compute_idf(self):
        """计算每个词的 IDF 值"""
        n_docs = len(self.tokenized_corpus)
        doc_freq = defaultdict(int)

        for doc_tokens in self.tokenized_corpus:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        for token in self.vocab:
            self.idf[token] = np.log((n_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5) + 1)

    def encode_documents(self) -> List[Dict[int, float]]:
        """
        将训练文档转换为稀疏向量

        用于插入 Milvus 的 sparse_vector 字段

        Returns:
            List[Dict[int, float]]: 稀疏向量列表，每个向量是 {索引: 值} 的字典
        """
        if self.bm25 is None:
            raise ValueError("请先调用 fit() 方法训练模型")

        sparse_vectors = []

        for doc_tokens in self.tokenized_corpus:
            sparse_vec = {}

            # 计算每个词的 BM25 分数
            for token in set(doc_tokens):
                if token in self.vocab:
                    idx = self.vocab[token]
                    # 使用 TF-IDF 作为稀疏向量值
                    tf = doc_tokens.count(token)
                    sparse_vec[idx] = tf * self.idf.get(token, 1.0)

            # 归一化
            if sparse_vec:
                max_val = max(abs(v) for v in sparse_vec.values())
                if max_val > 0:
                    sparse_vec = {k: v / max_val for k, v in sparse_vec.items()}

            sparse_vectors.append(sparse_vec)

        return sparse_vectors

    def encode_query(self, query: str) -> Dict[int, float]:
        """
        将查询转换为稀疏向量

        Args:
            query: 查询文本

        Returns:
            Dict[int, float]: 稀疏向量 {词索引: 词权重}
        """
        if self.bm25 is None:
            raise ValueError("请先调用 fit() 方法训练模型")

        # 分词
        query_tokens = list(jieba.cut(query))

        # 构建查询词的稀疏向量（正确格式：词索引 -> 词权重）
        sparse_vec = {}
        token_counts = {}

        # 统计查询词频
        for token in query_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # 计算每个查询词的权重（使用 BM25 查询词权重公式）
        for token, count in token_counts.items():
            if token in self.vocab:
                idx = self.vocab[token]
                # BM25 查询词权重 = IDF * (tf * (k1 + 1)) / (tf + k1)
                tf = count
                idf = self.idf.get(token, 1.0)
                weight = idf * (tf * (self.k1 + 1)) / (tf + self.k1)
                sparse_vec[idx] = weight

        # 归一化
        if sparse_vec:
            max_val = max(abs(v) for v in sparse_vec.values())
            if max_val > 0:
                sparse_vec = {k: v / max_val for k, v in sparse_vec.items()}

        return sparse_vec

    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.vocab)


class SparseEmbedding:
    """
    稀疏向量基类接口
    用于与 Dense Embedding 保持一致的接口
    """

    def __init__(self, sparse_model: BM25Sparse):
        self.model = sparse_model

    def encode(self, texts: List[str]) -> List[Dict[int, float]]:
        """编码多个文本"""
        if self.model.bm25 is None:
            raise ValueError("请先训练模型")
        # 对于查询，使用 encode_query
        return [self.model.encode_query(text) for text in texts]

    def encode_single(self, text: str) -> Dict[int, float]:
        """编码单个文本"""
        return self.model.encode_query(text)


if __name__ == "__main__":
    # 测试代码
    corpus = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务",
        "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的表示",
        "自然语言处理是人工智能的重要应用领域，涉及计算机与人类语言之间的交互",
        "机器学习是人工智能的核心技术之一，使计算机能够从数据中学习",
        "计算机视觉是人工智能的另一个重要分支，使计算机能够理解和分析图像"
    ]

    # 创建并训练 BM25 模型
    bm25 = BM25Sparse(k1=1.5, b=0.75)
    bm25.fit(corpus)

    # 获取文档稀疏向量
    doc_vectors = bm25.encode_documents()
    print(f"文档稀疏向量数量: {len(doc_vectors)}")
    print(f"第一个文档稀疏向量: {list(doc_vectors[0].items())[:5]}")

    # 编码查询
    query = "什么是深度学习"
    query_vector = bm25.encode_query(query)
    print(f"\n查询稀疏向量: {list(query_vector.items())[:5]}")

    # 使用 BM25 获取分数
    scores = bm25.bm25.get_scores(list(jieba.cut(query)))
    print(f"\nBM25 分数: {scores}")
