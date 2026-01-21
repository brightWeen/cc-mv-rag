# Milvus 多路检索验证

验证 Milvus 是否可以替代 Elasticsearch 的文本检索能力。

## 目标

通过 Dense + Sparse 混合检索与 ES BM25 进行效果对比，验证：
- Dense 向量语义检索效果
- Sparse 向量关键词检索效果
- 混合检索 (Hybrid Search) 效果
- 与 ES BM25 的效果对比

## 技术栈

- **Embedding 模型**: GLM Embedding API (智谱 AI)
- **稀疏向量**: BM25 算法
- **向量数据库**: Milvus Lite (本地)
- **对比基准**: Elasticsearch + IK 分词器
- **结果融合**: RRF (Reciprocal Rank Fusion)

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 复制环境变量配置
cp .env.example .env
# 编辑 .env，填入 GLM API Key

# 启动 Milvus Lite
python3 -m milvus

# 启动 Elasticsearch (Docker)
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.11.0
```

## 使用

```bash
# 1. 准备数据
python3 scripts/01_prepare_data.py

# 2. 构建索引
python3 scripts/02_build_indexes.py

# 3. 执行检索
python3 scripts/03_run_search.py

# 4. 评估结果
python3 scripts/04_evaluate.py
```

## 验证结论

经过详细对比验证，得出以下核心结论：

1.  **加权融合优于 RRF**: 在 Milvus 单库方案中，使用 **加权融合 (Dense=0.6, Sparse=0.4)** 的效果显著优于 RRF 融合，NDCG@10 达到 **0.9198**，非常接近 ES+Milvus 方案 (0.9398)。
2.  **Milvus 单库可行性**: 对于绝大多数语义检索和标准关键词匹配场景，**Only Milvus (Weighted)** 方案具备极高的性价比，足以替代 ES。
3.  **ES 的不可替代性**: 在通配符查询 (`RTX*`)、模糊纠错 (`intell`) 和严格短语匹配等特定场景下，ES 凭借其强大的倒排索引和分词能力仍然具有不可替代的优势。

**详细报告:**
- 📄 [汇总对比报告 (Summary Report)](outputs/reports/milvus_vs_es_milvus_summary.md)
- 🔍 [差距分析报告 (Gap Analysis)](outputs/reports/gap_analysis_cases.md)

## 项目结构

```
cc-mv-rag/
├── src/                    # 源代码
│   ├── config/            # 配置管理
│   ├── models/            # 向量模型
│   ├── database/          # 数据库连接
│   ├── pipeline/          # 数据处理流程
│   ├── search/            # 检索逻辑
│   └── evaluation/        # 评估模块
├── scripts/               # 脚本目录
├── data/                  # 数据目录
└── outputs/               # 输出目录
```
