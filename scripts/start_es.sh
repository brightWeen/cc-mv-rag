#!/bin/bash
# Elasticsearch 启动脚本

echo "正在启动 Elasticsearch..."

# 检查是否已有 ES 容器在运行
if docker ps | grep -q elasticsearch; then
    echo "Elasticsearch 已在运行，停止旧容器..."
    docker stop elasticsearch
    docker rm elasticsearch
fi

# 拉取 ES 镜像（如果不存在）
if ! docker images | grep -q elasticsearch; then
    echo "正在拉取 Elasticsearch 镜像..."
    docker pull docker.elastic.co/elasticsearch/elasticsearch:8.11.0
fi

# 创建数据卷
docker volume create es_data 2>/dev/null

# 启动 Elasticsearch
echo "启动 Elasticsearch 容器..."
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.security.http.ssl.enabled=false" \
  -v es_data:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0

echo ""
echo "Elasticsearch 启动中..."
echo "健康检查: curl http://localhost:9200/_cluster/health"
echo ""
echo "等待约 30-60 秒让 ES 完全启动..."
