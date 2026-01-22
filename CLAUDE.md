# Claude AI 协作指南

## 项目概述

本项目是一个 Milvus 多路检索验证框架，用于对比 Milvus、Elasticsearch 和 OceanBase SeekDB 的检索性能。

## 代码规范

### Python 执行
- **使用 `python3`**: 项目使用 Python 3，执行脚本时请使用 `python3` 命令
  ```bash
  python3 scripts/01_prepare_data.py
  python3 scripts/02_build_indexes.py
  ```

### 项目结构
```
cc-mv-rag/
├── src/                    # 源代码
│   ├── config/            # 配置管理
│   ├── models/            # 向量模型
│   ├── database/          # 数据库连接 (Milvus, ES, SeekDB)
│   ├── pipeline/          # 数据处理流程
│   ├── search/            # 检索逻辑
│   └── evaluation/        # 评估模块
├── scripts/               # 脚本目录
├── data/                  # 数据目录
└── outputs/               # 输出目录
    ├── results/           # 检索结果
    └── reports/           # 评估报告
```

## 工作流程

### 1. 数据准备
```bash
python3 scripts/01_prepare_data.py
```
生成测试文档和查询数据。

### 2. 索引构建
```bash
# 构建 Milvus 索引
python3 scripts/02_build_indexes.py

# 构建 SeekDB 索引（需要 SeekDB 运行）
python3 scripts/02_build_indexes.py --seekdb
```

### 3. 检索执行
```bash
# 执行 Milvus 检索
python3 scripts/03_run_search.py

# 执行 SeekDB 检索
python3 scripts/03_run_search.py --seekdb
```

### 4. 结果评估
```bash
python3 scripts/04_evaluate.py
```

## 关键配置

### 环境变量
- `GLM_API_KEY`: 智谱 AI Embedding API 密钥

### 数据库配置
- **Milvus**: 本地文件 `milvus_lite.db`
- **Elasticsearch**: `http://localhost:9200`
- **SeekDB**: `http://localhost:2881`

## 验证结果总结

### 准确率排名 (NDCG@10)
1. Milvus Hybrid Weighted: **0.9198**
2. Milvus Dense: 0.8948
3. Milvus Hybrid RRF: 0.8820
4. SeekDB Hybrid RRF: 0.8527

### 性能对比
- **Milvus**: ~170ms/查询，QPS 4-6
- **SeekDB**: ~1600ms/查询，QPS 0.5-0.6

## 重要说明

1. **向量模型**: 使用 GLM embedding-3 (2048维)
2. **融合算法**:
   - RRF (Reciprocal Rank Fusion)
   - Weighted Fusion (Dense=0.6, Sparse=0.4)
3. **SeekDB 限制**: 只支持 RRF 融合，不支持加权融合

## 报告文件

- `outputs/reports/evaluation_report.md`: 主评估报告
- `outputs/reports/seekdb_integration/`: SeekDB 集成专用报告
- `outputs/reports/comparison_results.json`: 详细对比数据
