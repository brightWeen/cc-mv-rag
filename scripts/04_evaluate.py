#!/usr/bin/env python3
"""
评估脚本

评估各检索方法的效果并生成对比报告
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from tabulate import tabulate

from src.config.settings import get_config
from src.evaluation.metrics import Evaluator


def load_queries(data_path: Path) -> list:
    """加载查询数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(results_path: Path) -> dict:
    """加载检索结果"""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_relevant_docs(queries: list) -> Dict[str, Set[str]]:
    """准备相关文档集合"""
    relevant_docs = {}
    for query_item in queries:
        query_id = query_item["query_id"]
        relevant_docs[query_id] = set(query_item["relevant_docs"])
    return relevant_docs


def print_comparison_table(comparison: Dict[str, Dict[str, float]]):
    """打印对比表格"""
    # 获取所有指标
    metrics = []
    for method_results in comparison.values():
        for key in method_results.keys():
            if key not in metrics and key != "query_count":
                metrics.append(key)
    metrics.sort()

    # 构建表格数据
    headers = ["方法"] + metrics
    rows = []

    for method_name, method_results in comparison.items():
        row = [method_name]
        for metric in metrics:
            value = method_results.get(metric, 0.0)
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        rows.append(row)

    print("\n" + "=" * 80)
    print("检索效果对比表")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 80 + "\n")


def print_ranking(comparison: Dict[str, Dict[str, float]]):
    """打印各指标排名"""
    print("\n" + "=" * 80)
    print("各指标排名（按值从高到低）")
    print("=" * 80 + "\n")

    # 关键指标
    key_metrics = ["recall@10", "mrr", "ndcg@10", "map@10"]

    for metric in key_metrics:
        print(f"【{metric.upper()}】")

        # 排序
        sorted_methods = sorted(
            comparison.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )

        for rank, (method_name, results) in enumerate(sorted_methods, 1):
            value = results.get(metric, 0)
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {rank}. {method_name:20s} {value:.4f}")

        print()


def generate_markdown_report(
    comparison: Dict[str, Dict[str, float]],
    output_path: Path
):
    """生成 Markdown 报告"""
    report_lines = []

    report_lines.append("# Milvus 多路检索验证报告\n")
    report_lines.append("## 评估结果对比\n")

    # 表格
    metrics = []
    for method_results in comparison.values():
        for key in method_results.keys():
            if key not in metrics and key != "query_count":
                metrics.append(key)
    metrics.sort()

    # Markdown 表格
    report_lines.append("| 方法 | " + " | ".join(metrics) + " |")
    report_lines.append("|" + "--|" * (len(metrics) + 1))

    for method_name, method_results in comparison.items():
        row_values = [f"{method_results.get(m, 0):.4f}" for m in metrics]
        report_lines.append(f"| {method_name} | " + " | ".join(row_values) + " |")

    report_lines.append("\n## 结论\n")

    # 分析
    report_lines.append("### 关键指标分析\n")

    key_metrics = ["recall@10", "mrr", "ndcg@10", "map@10"]
    metric_descriptions = {
        "recall@10": "召回率@10 - 前 10 个结果中相关文档的覆盖程度",
        "mrr": "平均倒数排名 - 首个相关文档的平均排名质量",
        "ndcg@10": "归一化折损累积增益@10 - 考虑位置的相关性质量",
        "map@10": "平均精度均值@10 - 整体检索质量"
    }

    for metric in key_metrics:
        sorted_methods = sorted(
            comparison.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )
        best_method = sorted_methods[0][0]
        best_value = sorted_methods[0][1].get(metric, 0)

        report_lines.append(f"#### {metric.upper()}")
        report_lines.append(f"- **描述**: {metric_descriptions.get(metric, '')}")
        report_lines.append(f"- **最佳方法**: {best_method} ({best_value:.4f})")
        report_lines.append("")

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Markdown 报告已保存: {output_path}")


def save_comparison_json(comparison: dict, output_path: Path):
    """保存对比结果为 JSON"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    logger.info(f"对比结果已保存: {output_path}")


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

    logger.info("开始评估检索结果")
    logger.info(f"项目: {config.project}")

    # 项目路径
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "outputs" / "results"
    reports_dir = project_root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

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

    # 准备相关文档集合
    relevant_docs = prepare_relevant_docs(queries)

    # 加载检索结果
    all_results_file = results_dir / "all_results.json"
    logger.info(f"加载检索结果: {all_results_file}")
    all_results = load_results(all_results_file)

    # 初始化评估器
    evaluator = Evaluator(k_values=config.evaluation.k_values)

    # 评估所有方法
    logger.info("=" * 50)
    logger.info("开始评估各检索方法")
    logger.info("=" * 50)

    comparison = evaluator.compare_results(all_results, relevant_docs)

    # 打印结果
    print_comparison_table(comparison)
    print_ranking(comparison)

    # 保存结果
    json_file = reports_dir / "comparison_results.json"
    save_comparison_json(comparison, json_file)

    md_file = reports_dir / "evaluation_report.md"
    generate_markdown_report(comparison, md_file)

    logger.info("=" * 50)
    logger.info("评估完成！")
    logger.info("=" * 50)
    logger.info(f"\n结果文件:")
    logger.info(f"  - JSON: {json_file}")
    logger.info(f"  - Markdown: {md_file}")


if __name__ == "__main__":
    main()
