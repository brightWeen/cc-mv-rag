"""
检索评估指标模块
"""

from typing import Dict, List, Set

import numpy as np


class RetrievalMetrics:
    """检索评估指标计算类"""

    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Recall@K: 召回率

        前 K 个结果中相关文档占所有相关文档的比例

        Args:
            retrieved_docs: 检索到的文档 ID 列表
            relevant_docs: 相关文档 ID 集合
            k: 前K个结果

        Returns:
            float: Recall@K 值
        """
        if not relevant_docs:
            return 0.0
        retrieved_at_k = set(retrieved_docs[:k])
        return len(retrieved_at_k & relevant_docs) / len(relevant_docs)

    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Precision@K: 精确率

        前 K 个结果中相关文档的比例

        Args:
            retrieved_docs: 检索到的文档 ID 列表
            relevant_docs: 相关文档 ID 集合
            k: 前K个结果

        Returns:
            float: Precision@K 值
        """
        if k == 0:
            return 0.0
        retrieved_at_k = set(retrieved_docs[:k])
        return len(retrieved_at_k & relevant_docs) / k

    @staticmethod
    def mrr(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """
        MRR: Mean Reciprocal Rank

        首个相关文档排名的倒数

        Args:
            retrieved_docs: 检索到的文档 ID 列表
            relevant_docs: 相关文档 ID 集合

        Returns:
            float: MRR 值
        """
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain

        归一化折损累积增益

        Args:
            retrieved_docs: 检索到的文档 ID 列表
            relevant_docs: 相关文档 ID 集合
            k: 前K个结果

        Returns:
            float: NDCG@K 值
        """
        # 计算 DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k], 1):
            if doc_id in relevant_docs:
                dcg += 1.0 / np.log2(i + 1)

        # 计算 IDCG (理想情况)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant_docs), k) + 1))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def map_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        MAP@K: Mean Average Precision

        平均精度均值

        Args:
            retrieved_docs: 检索到的文档 ID 列表
            relevant_docs: 相关文档 ID 集合
            k: 前K个结果

        Returns:
            float: MAP@K 值
        """
        precisions = []
        hit_count = 0

        for i, doc_id in enumerate(retrieved_docs[:k], 1):
            if doc_id in relevant_docs:
                hit_count += 1
                precisions.append(hit_count / i)

        return np.mean(precisions) if precisions else 0.0

    @staticmethod
    def f1_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        F1@K: F1 分数

        精确率和召回率的调和平均

        Args:
            retrieved_docs: 检索到的文档 ID 列表
            relevant_docs: 相关文档 ID 集合
            k: 前K个结果

        Returns:
            float: F1@K 值
        """
        precision = RetrievalMetrics.precision_at_k(retrieved_docs, relevant_docs, k)
        recall = RetrievalMetrics.recall_at_k(retrieved_docs, relevant_docs, k)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class Evaluator:
    """
    检索评估器

    用于评估检索结果的质量
    """

    def __init__(self, k_values: List[int] = None):
        """
        初始化评估器

        Args:
            k_values: 评估的 K 值列表
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.metrics = RetrievalMetrics()

    def evaluate_single_query(
        self,
        retrieved_docs: List[str],
        relevant_docs: Set[str]
    ) -> Dict[str, float]:
        """
        评估单个查询的检索结果

        Args:
            retrieved_docs: 检索到的文档 ID 列表
            relevant_docs: 相关文档 ID 集合

        Returns:
            Dict[str, float]: 各指标值
        """
        results = {
            "mrr": self.metrics.mrr(retrieved_docs, relevant_docs),
            "ndcg@10": self.metrics.ndcg_at_k(retrieved_docs, relevant_docs, 10),
            "map@10": self.metrics.map_at_k(retrieved_docs, relevant_docs, 10),
        }

        for k in self.k_values:
            results[f"recall@{k}"] = self.metrics.recall_at_k(retrieved_docs, relevant_docs, k)
            results[f"precision@{k}"] = self.metrics.precision_at_k(retrieved_docs, relevant_docs, k)
            results[f"f1@{k}"] = self.metrics.f1_at_k(retrieved_docs, relevant_docs, k)

        return results

    def evaluate_all_queries(
        self,
        all_results: Dict[str, List[str]],
        all_relevant: Dict[str, Set[str]]
    ) -> Dict[str, float]:
        """
        评估所有查询的平均结果

        Args:
            all_results: 查询 ID -> 检索结果列表
            all_relevant: 查询 ID -> 相关文档集合

        Returns:
            Dict[str, float]: 各指标的平均值
        """
        metric_sums = {}
        metric_counts = {}
        query_count = 0

        for query_id, retrieved_docs in all_results.items():
            relevant_docs = all_relevant.get(query_id, set())
            if not relevant_docs:
                continue

            metrics = self.evaluate_single_query(retrieved_docs, relevant_docs)

            for metric_name, value in metrics.items():
                if metric_name not in metric_sums:
                    metric_sums[metric_name] = 0.0
                    metric_counts[metric_name] = 0
                metric_sums[metric_name] += value
                metric_counts[metric_name] += 1

            query_count += 1

        # 计算平均值
        avg_metrics = {}
        for metric_name in metric_sums:
            avg_metrics[metric_name] = metric_sums[metric_name] / metric_counts[metric_name]

        avg_metrics["query_count"] = query_count

        return avg_metrics

    def compare_results(
        self,
        methods_results: Dict[str, Dict[str, List[str]]],
        all_relevant: Dict[str, Set[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        比较多个检索方法的结果

        Args:
            methods_results: 方法名 -> {查询ID -> 检索结果}
            all_relevant: 查询 ID -> 相关文档集合

        Returns:
            Dict[str, Dict[str, float]]: 方法名 -> 指标值
        """
        comparison = {}

        for method_name, all_results in methods_results.items():
            avg_metrics = self.evaluate_all_queries(all_results, all_relevant)
            comparison[method_name] = avg_metrics

        return comparison


if __name__ == "__main__":
    # 测试代码
    print("=== 检索评估指标测试 ===\n")

    # 模拟检索结果
    retrieved = ["doc_005", "doc_002", "doc_008", "doc_001", "doc_010"]
    relevant = {"doc_001", "doc_002", "doc_005"}

    evaluator = Evaluator(k_values=[1, 3, 5, 10])

    metrics = evaluator.evaluate_single_query(retrieved, relevant)

    print("单个查询评估结果:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    print("\n=== 多方法对比测试 ===\n")

    # 模拟多个方法的结果
    methods_results = {
        "Dense": {
            "q_001": ["doc_005", "doc_002", "doc_008", "doc_001", "doc_010"],
            "q_002": ["doc_003", "doc_007", "doc_001", "doc_009"],
        },
        "Sparse": {
            "q_001": ["doc_001", "doc_005", "doc_002", "doc_008", "doc_010"],
            "q_002": ["doc_001", "doc_007", "doc_003", "doc_009"],
        },
        "Hybrid": {
            "q_001": ["doc_001", "doc_005", "doc_002", "doc_008", "doc_010"],
            "q_002": ["doc_001", "doc_003", "doc_007", "doc_009"],
        },
    }

    all_relevant = {
        "q_001": {"doc_001", "doc_002", "doc_005"},
        "q_002": {"doc_001", "doc_003", "doc_007"},
    }

    comparison = evaluator.compare_results(methods_results, all_relevant)

    print("多方法对比结果:")
    for method, metrics in comparison.items():
        print(f"\n{method}:")
        for metric_name, value in metrics.items():
            if metric_name != "query_count":
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
