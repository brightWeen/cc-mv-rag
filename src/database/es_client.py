"""
Elasticsearch 客户端封装
"""

from typing import List, Dict, Optional

from elasticsearch import Elasticsearch
from loguru import logger

from .schemas import get_es_mapping


class ESClient:
    """Elasticsearch 客户端封装"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "doc_chunks",
        username: str = "",
        password: str = "",
        verify_certs: bool = False
    ):
        """
        初始化 ES 客户端

        Args:
            host: ES 主机地址
            port: ES 端口
            index_name: 索引名称
            username: 用户名
            password: 密码
            verify_certs: 是否验证证书
        """
        self.host = host
        self.port = port
        self.index_name = index_name

        # 构建连接配置
        if username and password:
            self.client = Elasticsearch(
                f"http://{host}:{port}",
                basic_auth=(username, password),
                verify_certs=verify_certs
            )
        else:
            self.client = Elasticsearch(
                f"http://{host}:{port}",
                verify_certs=verify_certs
            )

        # 测试连接
        if self.client.ping():
            logger.info(f"已连接到 Elasticsearch: {host}:{port}")
        else:
            logger.warning(f"无法连接到 Elasticsearch: {host}:{port}")

    def create_index(self, drop_existing: bool = False, use_ik_analyzer: bool = False) -> bool:
        """
        创建索引

        Args:
            drop_existing: 如果索引已存在是否删除
            use_ik_analyzer: 是否使用 IK 中文分词器

        Returns:
            bool: 是否创建成功
        """
        # 如果索引已存在
        if self.client.indices.exists(index=self.index_name):
            if drop_existing:
                self.client.indices.delete(index=self.index_name)
                logger.info(f"已删除现有索引: {self.index_name}")
            else:
                logger.info(f"索引已存在: {self.index_name}")
                return True

        # 获取 Mapping
        mapping = get_es_mapping(use_ik_analyzer=use_ik_analyzer)

        # 创建索引 (ES v8.x API)
        self.client.indices.create(
            index=self.index_name,
            body=mapping
        )
        logger.info(f"已创建索引: {self.index_name}")

        return True

    def insert_documents(self, documents: List[Dict]) -> int:
        """
        批量插入文档

        Args:
            documents: 文档列表

        Returns:
            int: 插入的文档数量
        """
        from elasticsearch.helpers import bulk

        actions = []
        for doc in documents:
            action = {
                "_index": self.index_name,
                "_id": doc.get("id", doc.get("chunk_id")),
                "_source": doc
            }
            actions.append(action)

        success_count, failed_items = bulk(
            self.client,
            actions,
            raise_on_error=False
        )

        logger.info(f"已插入 {success_count} 条文档到 ES")

        if failed_items:
            logger.warning(f"有 {len(failed_items)} 条文档插入失败")

        return success_count

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        执行 BM25 检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[Dict]: 检索结果
        """
        query_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "title": {
                                    "query": query,
                                    "boost": 2.0
                                }
                            }
                        },
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "boost": 1.0
                                }
                            }
                        }
                    ]
                }
            },
            "size": top_k
        }

        response = self.client.search(index=self.index_name, body=query_body)

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "doc_id": hit["_source"].get("doc_id"),
                "chunk_id": hit["_source"].get("chunk_id"),
                "title": hit["_source"].get("title"),
                "content": hit["_source"].get("content"),
                "score": hit["_score"]
            })

        return results

    def get_stats(self) -> dict:
        """获取索引统计信息"""
        try:
            stats = self.client.indices.stats(index=self.index_name)
            doc_count = stats["indices"][self.index_name]["primaries"]["docs"]["count"]
            return {
                "index_name": self.index_name,
                "doc_count": doc_count
            }
        except Exception as e:
            logger.error(f"获取索引统计失败: {e}")
            return {
                "index_name": self.index_name,
                "doc_count": 0
            }

    def delete_index(self):
        """删除索引"""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            logger.info(f"已删除索引: {self.index_name}")

    def close(self):
        """关闭客户端"""
        self.client.close()
        logger.info("已关闭 ES 连接")


if __name__ == "__main__":
    # 测试代码
    import os
    from dotenv import load_dotenv

    load_dotenv()

    client = ESClient(
        host=os.getenv("ES_HOST", "localhost"),
        port=int(os.getenv("ES_PORT", 9200)),
        index_name="test_index"
    )

    # 创建索引
    client.create_index(drop_existing=True)

    # 插入测试文档
    test_docs = [
        {
            "id": "test_001",
            "doc_id": "doc_001",
            "chunk_id": "chunk_001",
            "title": "测试文档",
            "content": "这是一个测试文档的内容，用于验证 Elasticsearch 连接和检索功能。"
        }
    ]
    client.insert_documents(test_docs)

    # 获取统计
    stats = client.get_stats()
    print(f"ES 索引统计: {stats}")

    # 测试检索
    results = client.search("测试", top_k=5)
    print(f"检索结果: {len(results)} 条")

    client.close()
