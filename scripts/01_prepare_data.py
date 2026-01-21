#!/usr/bin/env python3
"""
数据准备脚本

生成示例文档数据、产品文档数据和测试查询数据
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_products(data_path: Path) -> list:
    """加载产品文档数据"""
    products_file = data_path / "raw" / "products.json"
    if products_file.exists():
        with open(products_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def generate_sample_documents() -> list:
    """
    生成示例文档数据

    Returns:
        list: 文档列表
    """
    documents = [
        {
            "doc_id": "doc_001",
            "title": "人工智能基础",
            "content": "人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务。这些任务包括视觉感知、语音识别、决策制定、语言翻译等。AI 的核心目标之一是让机器具备类似人类的思维能力。",
            "category": "技术",
            "metadata": {"author": "AI 研究所", "date": "2024-01-01"}
        },
        {
            "doc_id": "doc_002",
            "title": "机器学习入门",
            "content": "机器学习（Machine Learning）是人工智能的核心技术之一。它使计算机能够从数据中学习，而不是通过明确的编程。机器学习算法可以分为监督学习、无监督学习和强化学习三大类。监督学习使用标注数据训练模型，无监督学习从未标注数据中发现模式，强化学习通过与环境交互学习最优策略。",
            "category": "技术",
            "metadata": {"author": "数据科学团队", "date": "2024-01-02"}
        },
        {
            "doc_id": "doc_003",
            "title": "深度学习详解",
            "content": "深度学习（Deep Learning）是机器学习的一个子领域，它使用多层神经网络来学习数据的表示。深度神经网络可以自动提取特征，无需人工设计特征工程。卷积神经网络（CNN）在图像识别中表现出色，循环神经网络（RNN）和 Transformer 在自然语言处理领域取得了突破性进展。",
            "category": "技术",
            "metadata": {"author": "深度学习实验室", "date": "2024-01-03"}
        },
        {
            "doc_id": "doc_004",
            "title": "自然语言处理",
            "content": "自然语言处理（Natural Language Processing，简称 NLP）是人工智能的重要应用领域，涉及计算机与人类语言之间的交互。NLP 任务包括文本分类、命名实体识别、情感分析、机器翻译、问答系统等。近年来，基于 Transformer 的预训练模型（如 BERT、GPT）在 NLP 任务上取得了显著成果。",
            "category": "技术",
            "metadata": {"author": "NLP 研究组", "date": "2024-01-04"}
        },
        {
            "doc_id": "doc_005",
            "title": "计算机视觉",
            "content": "计算机视觉（Computer Vision）是人工智能的另一个重要分支，使计算机能够理解和分析图像和视频。主要任务包括图像分类、目标检测、图像分割、人脸识别、姿态估计等。卷积神经网络（CNN）是计算机视觉的主流架构，ResNet、EfficientNet 等模型在各种视觉任务中达到了最先进的性能。",
            "category": "技术",
            "metadata": {"author": "视觉计算中心", "date": "2024-01-05"}
        },
        {
            "doc_id": "doc_006",
            "title": "强化学习基础",
            "content": "强化学习（Reinforcement Learning）是一种通过与环境交互来学习最优策略的机器学习方法。智能体（Agent）通过执行动作、观察状态和接收奖励来学习。强化学习在游戏 AI（如 AlphaGo）、机器人控制、自动驾驶等领域有广泛应用。Q-Learning、策略梯度、Actor-Critic 是常用的强化学习算法。",
            "category": "技术",
            "metadata": {"author": "机器人实验室", "date": "2024-01-06"}
        },
        {
            "doc_id": "doc_007",
            "title": "知识图谱概述",
            "content": "知识图谱（Knowledge Graph）是一种结构化的知识表示方法，用图的形式存储实体及其关系。知识图谱由节点（实体）和边（关系）组成，可用于语义搜索、问答系统、推荐系统等。知识图谱的构建包括实体抽取、关系抽取、知识融合等技术。Google 知识图谱是知识图谱应用的典型代表。",
            "category": "技术",
            "metadata": {"author": "数据挖掘团队", "date": "2024-01-07"}
        },
        {
            "doc_id": "doc_008",
            "title": "大语言模型",
            "content": "大语言模型（Large Language Models，简称 LLM）是近年来 AI 领域的重大突破。这些模型（如 GPT-4、Claude、文心一言）通过在海量文本数据上预训练，获得了强大的语言理解和生成能力。LLM 可以进行对话、写作、编程、推理等多种任务。提示工程（Prompt Engineering）和微调（Fine-tuning）是使用 LLM 的关键技术。",
            "category": "技术",
            "metadata": {"author": "AI 研究院", "date": "2024-01-08"}
        },
        {
            "doc_id": "doc_009",
            "title": "向量数据库",
            "content": "向量数据库（Vector Database）是专门用于存储和检索高维向量的数据库系统。它在 AI 应用中扮演着重要角色，特别是在语义搜索、推荐系统和 RAG（检索增强生成）等场景中。向量数据库使用近似最近邻（ANN）算法来快速检索相似向量。常见的向量数据库包括 Milvus、Pinecone、Weaviate、Chroma 等。",
            "category": "技术",
            "metadata": {"author": "数据库团队", "date": "2024-01-09"}
        },
        {
            "doc_id": "doc_010",
            "title": "RAG 技术介绍",
            "content": "RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索和文本生成的技术。RAG 首先从知识库中检索相关文档，然后将检索结果作为上下文输入到大语言模型，生成更准确的回答。RAG 可以解决 LLM 的知识滞后和幻觉问题，在企业知识库、客服机器人、智能问答等场景中有广泛应用。",
            "category": "技术",
            "metadata": {"author": "应用开发组", "date": "2024-01-10"}
        },
    ]

    return documents


def generate_test_queries() -> list:
    """
    生成测试查询数据

    Returns:
        list: 查询列表
    """
    queries = [
        {
            "query_id": "q_001",
            "query": "什么是人工智能？",
            "relevant_docs": ["doc_001"],
            "query_type": "factual"
        },
        {
            "query_id": "q_002",
            "query": "机器学习有哪些类型？",
            "relevant_docs": ["doc_002"],
            "query_type": "factual"
        },
        {
            "query_id": "q_003",
            "query": "深度学习和机器学习的区别",
            "relevant_docs": ["doc_002", "doc_003"],
            "query_type": "comparison"
        },
        {
            "query_id": "q_004",
            "query": "自然语言处理有哪些应用",
            "relevant_docs": ["doc_004"],
            "query_type": "application"
        },
        {
            "query_id": "q_005",
            "query": "CNN 是什么？",
            "relevant_docs": ["doc_003", "doc_005"],
            "query_type": "technical"
        },
        {
            "query_id": "q_006",
            "query": "强化学习的应用场景",
            "relevant_docs": ["doc_006"],
            "query_type": "application"
        },
        {
            "query_id": "q_007",
            "query": "知识图谱有什么用",
            "relevant_docs": ["doc_007"],
            "query_type": "application"
        },
        {
            "query_id": "q_008",
            "query": "GPT 是大语言模型吗",
            "relevant_docs": ["doc_008"],
            "query_type": "factual"
        },
        {
            "query_id": "q_009",
            "query": "向量数据库是用来做什么的",
            "relevant_docs": ["doc_009"],
            "query_type": "application"
        },
        {
            "query_id": "q_010",
            "query": "RAG 技术解决了什么问题",
            "relevant_docs": ["doc_010"],
            "query_type": "factual"
        },
        {
            "query_id": "q_011",
            "query": "AI 在图像识别中的应用",
            "relevant_docs": ["doc_005"],
            "query_type": "application"
        },
        {
            "query_id": "q_012",
            "query": "Transformer 模型",
            "relevant_docs": ["doc_003", "doc_004", "doc_008"],
            "query_type": "technical"
        },
    ]

    return queries


def main():
    """主函数"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    queries_dir = data_dir / "queries"
    processed_dir = data_dir / "processed"

    for dir_path in [raw_dir, queries_dir, processed_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 生成示例文档
    print("正在生成示例文档数据...")
    documents = generate_sample_documents()
    docs_file = raw_dir / "sample_docs.json"
    with open(docs_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"已生成 {len(documents)} 个文档 -> {docs_file}")

    # 加载产品文档
    products = load_products(data_dir)
    if products:
        print(f"\n加载 {len(products)} 个产品文档")
        documents.extend(products)

    # 保存合并后的文档
    all_docs_file = raw_dir / "all_docs.json"
    with open(all_docs_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"已保存合并文档 -> {all_docs_file}")

    # 生成测试查询
    print("\n正在生成测试查询数据...")
    queries = generate_test_queries()
    queries_file = queries_dir / "test_queries.json"
    with open(queries_file, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print(f"已生成 {len(queries)} 个查询 -> {queries_file}")

    # 加载混合查询
    mixed_queries_file = queries_dir / "mixed_queries.json"
    if mixed_queries_file.exists():
        with open(mixed_queries_file, "r", encoding="utf-8") as f:
            mixed_queries = json.load(f)
        print(f"已加载 {len(mixed_queries)} 个混合查询 -> {mixed_queries_file}")
        queries.extend(mixed_queries)

    print("\n数据准备完成！")
    print(f"\n文档统计:")
    print(f"  - 技术文档: {len([d for d in documents if d['doc_id'].startswith('doc_')])}")
    print(f"  - 产品文档: {len([d for d in documents if d['doc_id'].startswith('product_')])}")
    print(f"  - 总文档数: {len(documents)}")
    print(f"  - 总字符数: {sum(len(doc['content']) for doc in documents)}")
    print(f"\n查询统计:")
    print(f"  - 总查询数: {len(queries)}")

    # 按类型统计查询
    query_type_counts = {}
    for q in queries:
        qt = q.get('query_type', 'unknown')
        query_type_counts[qt] = query_type_counts.get(qt, 0) + 1
    print(f"  - 查询类型分布: {query_type_counts}")


if __name__ == "__main__":
    main()
