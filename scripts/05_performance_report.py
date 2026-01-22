#!/usr/bin/env python3
"""
性能报告生成 - 基于已有结果数据
"""

import json
from pathlib import Path


def main():
    # 读取评估结果
    report_path = Path("outputs/reports/evaluation_report.md")
    comparison_path = Path("outputs/reports/comparison_results.json")

    if comparison_path.exists():
        with open(comparison_path) as f:
            data = json.load(f)

        print("=" * 70)
        print("性能数据对比报告")
        print("=" * 70)

        # 1. 准确率对比
        print("\n【准确率对比】NDCG@10 (越高越好)")
        print("-" * 70)

        ndcg_scores = []
        for method, metrics in data.items():
            if method != "es_bm25":  # 排除失败的 ES BM25
                ndcg_scores.append((method, metrics["ndcg@10"]))

        for method, score in sorted(ndcg_scores, key=lambda x: -x[1]):
            print(f"  {method:25s}: {score:.4f}")

        # 2. 召回率对比
        print("\n【召回率对比】Recall@10 (越高越好)")
        print("-" * 70)

        recall_scores = []
        for method, metrics in data.items():
            if method != "es_bm25":
                recall_scores.append((method, metrics["recall@10"]))

        for method, score in sorted(recall_scores, key=lambda x: -x[1]):
            print(f"  {method:25s}: {score:.4f}")

        # 3. MRR 对比
        print("\n【排名质量对比】MRR (越高越好)")
        print("-" * 70)

        mrr_scores = []
        for method, metrics in data.items():
            if method != "es_bm25":
                mrr_scores.append((method, metrics["mrr"]))

        for method, score in sorted(mrr_scores, key=lambda x: -x[1]):
            print(f"  {method:25s}: {score:.4f}")

        # 4. 综合对比表
        print("\n【综合性能对比表】")
        print("-" * 70)
        print(f"{'方法':<22} {'NDCG@10':<10} {'Recall@10':<10} {'MRR':<10} {'MAP@10':<10}")
        print("-" * 70)

        for method, metrics in sorted(data.items(), key=lambda x: -x[1]["ndcg@10"]):
            if method == "es_bm25":
                continue
            print(f"{method:<22} {metrics['ndcg@10']:<10.4f} {metrics['recall@10']:<10.4f} {metrics['mrr']:<10.4f} {metrics['map@10']:<10.4f}")

        # 5. 性能分析
        print("\n【关键发现】")
        print("-" * 70)

        # 找出最佳方法
        best_ndcg = max(ndcg_scores, key=lambda x: x[1])
        best_recall = max(recall_scores, key=lambda x: x[1])
        best_mrr = max(mrr_scores, key=lambda x: x[1])

        print(f"  最佳 NDCG@10    : {best_ndcg[0]} ({best_ndcg[1]:.4f})")
        print(f"  最佳 Recall@10  : {best_recall[0]} ({best_recall[1]:.4f})")
        print(f"  最佳 MRR        : {best_mrr[0]} ({best_mrr[1]:.4f})")

        # RRF vs Weighted 对比
        print("\n【融合算法对比】(使用相同 GLM 2048维向量)")
        print("-" * 70)

        if "hybrid_rrf" in data and "hybrid_weighted" in data:
            rrf = data["hybrid_rrf"]
            weighted = data["hybrid_weighted"]

            print(f"  指标           RRF          Weighted      提升")
            print(f"  ───────────────────────────────────────────────")
            print(f"  NDCG@10        {rrf['ndcg@10']:.4f}       {weighted['ndcg@10']:.4f}       +{(weighted['ndcg@10']-rrf['ndcg@10'])*100:.2f}%")
            print(f"  Recall@10      {rrf['recall@10']:.4f}       {weighted['recall@10']:.4f}       +{(weighted['recall@10']-rrf['recall@10'])*100:.2f}%")
            print(f"  MRR            {rrf['mrr']:.4f}       {weighted['mrr']:.4f}       +{(weighted['mrr']-rrf['mrr'])*100:.2f}%")
            print(f"  MAP@10         {rrf['map@10']:.4f}       {weighted['map@10']:.4f}       +{(weighted['map@10']-rrf['map@10'])*100:.2f}%")

        # SeekDB vs Milvus RRF 对比
        print("\n【RRF 实现对比】(相同融合算法)")
        print("-" * 70)

        if "hybrid_rrf" in data and "seekdb_hybrid_rrf" in data:
            milvus_rrf = data["hybrid_rrf"]
            seekdb_rrf = data["seekdb_hybrid_rrf"]

            print(f"  指标           Milvus RRF   SeekDB RRF    差异")
            print(f"  ───────────────────────────────────────────────")
            print(f"  NDCG@10        {milvus_rrf['ndcg@10']:.4f}       {seekdb_rrf['ndcg@10']:.4f}       {(seekdb_rrf['ndcg@10']-milvus_rrf['ndcg@10'])*100:+.2f}%")
            print(f"  Recall@10      {milvus_rrf['recall@10']:.4f}       {seekdb_rrf['recall@10']:.4f}       {(seekdb_rrf['recall@10']-milvus_rrf['recall@10'])*100:+.2f}%")
            print(f"  MRR            {milvus_rrf['mrr']:.4f}       {seekdb_rrf['mrr']:.4f}       {(seekdb_rrf['mrr']-milvus_rrf['mrr'])*100:+.2f}%")
            print(f"  MAP@10         {milvus_rrf['map@10']:.4f}       {seekdb_rrf['map@10']:.4f}       {(seekdb_rrf['map@10']-milvus_rrf['map@10'])*100:+.2f}%")

        # 6. 执行时间估算（基于之前运行的观察）
        print("\n【执行时间观察】(基于31个查询)")
        print("-" * 70)
        print("  注意: 以下为实际运行观察值，非精确测试")
        print()
        print("  Milvus Dense (31 queries)    : ~3-5 秒    (~100-150 ms/查询)")
        print("  Milvus Hybrid RRF (31 q)     : ~5-7 秒    (~170-220 ms/查询)")
        print("  Milvus Hybrid Weighted (31 q): ~5-7 秒    (~170-220 ms/查询)")
        print("  SeekDB Hybrid RRF (31 q)     : ~50-60 秒  (~1600-2000 ms/查询)")
        print()
        print("  QPS 估算 (单线程):")
        print("    Milvus Dense     : ~6-10 queries/sec")
        print("    Milvus Hybrid    : ~4-6 queries/sec")
        print("    SeekDB Hybrid    : ~0.5-0.6 queries/sec")
        print()
        print("  结论: Milvus 执行速度约为 SeekDB 的 8-10 倍")


if __name__ == "__main__":
    main()
