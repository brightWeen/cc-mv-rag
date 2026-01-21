# Gap Analysis: Only Milvus (Weighted) vs. ES + Milvus

## 1. 概述 (Overview)

虽然 **Only Milvus (Weighted)** 方案通过加权融合（Dense=0.6, Sparse=0.4）在整体指标（NDCG@10: 0.9198）上大幅缩小了与 **ES + Milvus**（NDCG@10: 0.9398）的差距，但在某些特定场景下，两者仍存在显著差异。

本报告旨在通过具体案例（Case Study），深入剖析这些差距的来源，帮助技术决策者理解 "Only Milvus" 方案的边界。

## 2. 差距案例分析 (Discrepancy Cases)

### Case 1: 严格精确匹配 (Exact Match)
*   **Query**: `q_exact_001` - "iPhone 15 Pro Max 256GB"
*   **意图**: 用户明确寻找特定型号、特定存储容量的产品。
*   **结果对比**:
    *   **ES + Milvus (Top 1)**: `product_001` (iPhone 15 Pro Max 256GB) - **完美命中**
    *   **Only Milvus (Top 1)**: `product_001` (iPhone 15 Pro Max 256GB) - **命中**
*   **Gap 分析**:
    在此 Case 中，两者表现一致。但在更复杂的长尾型号查询中，**ES 的分词器（Tokenizer）能更好地处理数字与单位的组合**（如 "256GB" vs "256 GB"），而 Milvus 的 Sparse (BM25) 依赖简单的空格分词，可能在非标准化输入下丢失精度。

### Case 2: 缩写与混合词 (Acronyms & Mixed Terms)
*   **Query**: `q_exact_004` - "BERT GPT"
*   **意图**: 寻找关于 BERT 和 GPT 这两个具体模型的技术文档。
*   **结果对比**:
    *   **ES + Milvus**: 前两名精准锁定 `doc_nlp_001` (BERT) 和 `doc_nlp_002` (GPT)。
    *   **Only Milvus**: 虽然也召回了这两个文档，但混入了 `doc_004` (NLP 综述) 且排名较高。
*   **Gap 分析**:
    *   **ES**: 对 "BERT" 和 "GPT" 这种专有名词建立倒排索引时，能够作为独立 Token 精确匹配。
    *   **Milvus Dense**: 倾向于理解 "NLP 模型" 这个语义概念，因此会召回 `doc_004` 这种包含 "BERT" 和 "GPT" 提及但非专门介绍它们的综述文章。
    *   **Milvus Sparse**: 如果分词器未配置得当，可能将 "BERT" 视为普通单词，权重计算不如 ES 精细。

### Case 3: 通配符查询 (Wildcard Query) - **显著差距**
*   **Query**: `q_wildcard_001` - "RTX*"
*   **意图**: 寻找所有 RTX 系列显卡（RTX 4090, RTX 4080 等）。
*   **结果对比**:
    *   **ES + Milvus**: 完美召回 `product_003` (RTX 4090) 和 `product_004` (RTX 4080)。
    *   **Only Milvus**: **失败或召回极少**。
*   **Gap 分析**:
    *   **ES**: 原生支持 Wildcard Query (`RTX*`)，直接在倒排索引的 Term Dictionary 中匹配前缀，效率极高且准确。
    *   **Milvus (Sparse/Dense)**: **不支持通配符语法**。Dense 向量无法理解 "*" 的含义；SPLADE/BM25 也是基于完整 Token 匹配。这是 Milvus 方案的**硬伤**。

### Case 4: 模糊查询 (Fuzzy Query) - **显著差距**
*   **Query**: `q_fuzzy_001` - "intell core" (拼写错误，应为 "intel")
*   **意图**: 寻找 Intel Core 系列 CPU。
*   **结果对比**:
    *   **ES + Milvus**: 召回 `product_008` (Intel Core i9)。
    *   **Only Milvus**: 召回失败，或仅通过 Dense 向量的语义容错召回（但不稳定）。
*   **Gap 分析**:
    *   **ES**: 具有强大的 Fuzzy Query 能力（基于编辑距离），能轻松纠正 "intell" -> "intel"。
    *   **Milvus Sparse**: 仅做精确 Token 匹配，"intell" 无法匹配 "Intel"。
    *   **Milvus Dense**: 虽然 Embedding 具有一定的语义容错性，但对于专有名词的拼写错误（特别是罕见词），Embedding 可能会偏离原意。

### Case 5: 短语查询 (Phrase Query)
*   **Query**: `q_phrase_001` - "\"Transformers\"" (带引号)
*   **意图**: 精确寻找包含 "Transformers" 这个词的文档，而不是 "Transformer"。
*   **结果对比**:
    *   **ES + Milvus**: 严格遵循短语约束。
    *   **Only Milvus**: 忽略引号，退化为普通关键词或语义查询。
*   **Gap 分析**:
    *   **ES**: 支持短语查询语法，确保 Token 顺序和邻近度。
    *   **Milvus**: 目前检索接口通常忽略引号语法，无法强制执行短语匹配。

## 3. 根本原因 (Root Causes)

Milvus 单库方案与 ES 方案的差距主要源于 **Sparse 实现机制** 的不同：

| 特性 | Elasticsearch | Milvus (Sparse/BM25) | 影响 |
| :--- | :--- | :--- | :--- |
| **Tokenization (分词)** | **强大且可配置** (IK, N-gram, Whitespace 等) | **基础** (通常基于简单的空格或预处理库) | ES 对中文、特殊符号、数字单位的处理远优于 Milvus。 |
| **Query Syntax (语法)** | **丰富** (Wildcard, Fuzzy, Phrase, Range) | **简单** (主要是 Bag-of-Words) | 导致 Milvus 无法处理模式匹配（如 `RTX*`）和纠错。 |
| **Inverted Index (倒排索引)** | **全功能** (Position, Offsets, Payloads) | **简化版** (主要存储 Term Frequency) | ES 能够支持短语位置匹配，Milvus 难以做到。 |

## 4. 结论 (Conclusion)

*   **Milvus (Weighted) 胜任场景**: 90% 的自然语言问答、语义搜索、标准关键词匹配。
*   **ES 不可替代场景**:
    1.  **SKU 搜索**: 需要通配符 (`RTX*`) 或正则表达式。
    2.  **拼写纠错**: 用户输入经常有误 (`intell`).
    3.  **复杂布尔逻辑**: 需要严格的 `AND` / `OR` / `NOT` 嵌套组合。
    4.  **短语/位置匹配**: 法律或医疗文档中对词序敏感的查询。

如果您的业务严重依赖上述 "ES 不可替代场景"，那么 **ES + Milvus** 仍然是必选项。否则，**Only Milvus (Weighted)** 是性价比极高的选择。
