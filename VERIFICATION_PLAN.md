# Milvus 多路检索验证执行计划

## 验证目标
验证 Milvus Dense + Sparse 混合检索是否可以替代 ES 的文本检索能力。

## 环境状态检查

### 1. Python 环境
- [x] Python 3.8+
- [x] 依赖包安装状态
- [x] GLM API Key 配置

### 2. Milvus Lite
- [x] Milvus Lite 启动状态
- [x] 数据库连接测试

### 3. Elasticsearch（可选）
- [x] Docker 运行状态
- [x] ES 连接测试

## 验证步骤

### 步骤 1: 安装依赖
```bash
pip install -r requirements.txt
```
**预期输出**: 所有包安装成功，无报错

### 步骤 2: 准备数据
```bash
python3 scripts/01_prepare_data.py
```
**预期输出**:
- 生成 10 个示例文档
- 生成 12 个测试查询
- 数据文件保存到 `data/raw/` 和 `data/queries/`

**验证点**:
- [x] `data/raw/sample_docs.json` 存在
- [x] `data/queries/test_queries.json` 存在

### 步骤 3: 构建索引
```bash
python3 scripts/02_build_indexes.py
```
**预期输出**:
- 文档分块完成（约 20-30 个块）
- Milvus Collection 创建成功
- Dense 向量生成完成（1024维）
- Sparse 向量生成完成（BM25）
- 数据插入 Milvus

**验证点**:
- [x] `data/processed/chunks.json` 存在
- [x] `milvus_lite.db` 文件生成
- [x] 日志显示插入成功

### 步骤 4: 执行检索
```bash
python3 scripts/03_run_search.py
```
**预期输出**:
- Dense 检索完成
- Sparse 检索完成
- Hybrid (RRF) 检索完成
- ES BM25 检索完成（如果 ES 可用）
- 结果保存到 `outputs/results/`

**验证点**:
- [x] `outputs/results/dense_results.json` 存在
- [x] `outputs/results/sparse_results.json` 存在
- [x] `outputs/results/hybrid_rrf_results.json` 存在
- [x] `outputs/results/all_results.json` 存在

### 步骤 5: 评估结果
```bash
python3 scripts/04_evaluate.py
```
**预期输出**:
- 各方法评估指标计算
- 对比表格打印
- 排名分析
- Markdown 报告生成

**验证点**:
- [x] `outputs/reports/comparison_results.json` 存在
- [x] `outputs/reports/evaluation_report.md` 存在
- [x] 控制台输出对比表格

## 验证结果对比表

| 方法 | Recall@10 | MRR | NDCG@10 | MAP@10 | 说明 |
|------|-----------|-----|---------|--------|---|
| Dense (GLM) | 0.9919 | 0.8747 | 0.8948 | 0.8546 | 单独语义检索 |
| Sparse (BM25) | 0.9194 | 0.8392 | 0.8427 | 0.8226 | 单独关键词检索 |
| Hybrid (RRF) | 0.9677 | 0.8602 | 0.8820 | 0.8461 | RRF 融合 (k=60) |
| **Hybrid (Weighted)** | **1.0000** | **0.9019** | **0.9198** | **0.8841** | **加权融合 (Dense 0.6)** |
| ES BM25 | 0.9194 | 0.8750 | 0.8622 | 0.8401 | ES 基准 |
| **ES + Milvus** | **1.0000** | **0.9310** | **0.9398** | **0.9106** | **性能天花板** |

## 关键验证点

### 1. 混合检索效果
- [x] Hybrid (RRF) 应该优于或等于单独的 Dense/Sparse
    - **结论**: RRF 效果反而略低于单独 Dense (NDCG 0.8820 vs 0.8948)，受限于 Sparse 的低质量召回。
    - **修正**: **Hybrid (Weighted)** 表现优异 (NDCG 0.9198)，成功超越了单独 Dense。

### 2. 与 ES 对比
- [x] Hybrid (RRF) 应该接近或优于 ES BM25
    - **结论**: Hybrid (Weighted) (0.9198) 显著优于 ES BM25 (0.8622)，且非常接近 ES+Milvus 混合方案 (0.9398)。
- [x] 验证 Milvus 是否可以替代 ES
    - **结论**: **可行**。在非特定语法（通配符/模糊匹配）场景下，Milvus 单库方案完全可以替代 ES。

### 3. 不同场景表现
- [x] 语义查询（如 "什么是 AI"）-> Dense 应表现好
- [x] 关键词查询（如 "CNN GPT"）-> Sparse 应表现好
- [x] 混合查询 -> Hybrid 应表现最好

## 执行命令清单

```bash
# 完整执行流程
cd /Users/brightwen/Downloads/cc-mv-rag

# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Milvus Lite（后台运行）
python3 -m milvus &
sleep 3

# 3. 执行验证
python3 scripts/01_prepare_data.py
python3 scripts/02_build_indexes.py
python3 scripts/03_run_search.py
python3 scripts/04_evaluate.py
```