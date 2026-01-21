# Milvus vs Elasticsearch RAG Verification

## Project Overview

This project is a verification framework designed to compare the retrieval performance of **Milvus** (using Hybrid Search with Dense + Sparse vectors) against **Elasticsearch** (using BM25) in a RAG (Retrieval-Augmented Generation) context.

The primary goal is to determine if a "Milvus Only" architecture can effectively replace a complex "Milvus + Elasticsearch" dual-database architecture while maintaining high retrieval quality.

**Key Technologies:**
*   **Language:** Python 3.8+
*   **Vector Database:** Milvus Lite (local)
*   **Search Engine:** Elasticsearch 8.11.0 (Docker)
*   **Embedding Model:** ZhipuAI GLM Embedding
*   **Sparse Retrieval:** BM25 (via Milvus Sparse Vector)
*   **Fusion Algorithms:** RRF (Reciprocal Rank Fusion) and Weighted Fusion

## Architecture & Structure

*   **`src/`**: Core library code.
    *   `database/`: Clients for Milvus and Elasticsearch.
    *   `models/`: Embedding wrappers (GLM, BM25).
    *   `search/`: Implementation of Hybrid Search and Fusion logic.
    *   `evaluation/`: Metrics calculation (NDCG, MRR, Recall).
*   **`scripts/`**: Sequential workflow scripts.
    *   `01_prepare_data.py`: Generates synthetic test data and queries.
    *   `02_build_indexes.py`: Chunks data and builds indices in Milvus and ES.
    *   `03_run_search.py`: Executes search across multiple configurations (Dense, Sparse, Hybrid RRF/Weighted, ES).
    *   `04_evaluate.py`: Calculates metrics and generates comparison reports.
*   **`outputs/`**: Stores search results (`results/`) and generated reports (`reports/`).

## Setup & Usage

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure Environment Variables
cp .env.example .env
# Edit .env and add your GLM_API_KEY
```

### 2. Service Startup

```bash
# Start Milvus Lite (runs as a local process)
python3 -m milvus

# Start Elasticsearch (if comparing against ES)
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.11.0
```

### 3. Execution Workflow

The verification process is divided into 4 steps:

```bash
# Step 1: Generate test data
python3 scripts/01_prepare_data.py

# Step 2: Build Milvus and ES indices
python3 scripts/02_build_indexes.py

# Step 3: Run search benchmarks
python3 scripts/03_run_search.py

# Step 4: Generate evaluation reports
python3 scripts/04_evaluate.py
```

## Key Findings & Conclusions

Recent analysis (see `outputs/reports/`) has established the following:

1.  **Weighted Fusion > RRF**: For Milvus Hybrid Search, a weighted approach (Dense=0.6, Sparse=0.4) significantly outperforms standard RRF, achieving an NDCG@10 of **0.9198**, comparable to ES+Milvus (0.9398).
2.  **Milvus Viability**: For most semantic and keyword search scenarios, a Milvus-only architecture is a cost-effective and performant alternative to maintaining a separate ES cluster.
3.  **ES Edge Cases**: Elasticsearch remains superior for specific query types due to its advanced tokenization and query syntax:
    *   Wildcard queries (e.g., `RTX*`)
    *   Fuzzy matching/Spell check (e.g., `intell` -> `intel`)
    *   Exact phrase matching with punctuation support.

## Important Files

*   `README.md`: Main project entry point.
*   `VERIFICATION_PLAN.md`: Original plan for the verification task.
*   `outputs/reports/milvus_vs_es_milvus_summary.md`: Detailed comparison summary.
*   `outputs/reports/gap_analysis_cases.md`: Analysis of specific cases where ES outperforms Milvus.
