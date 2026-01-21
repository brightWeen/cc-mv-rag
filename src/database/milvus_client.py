"""
Milvus 客户端封装
"""

from typing import List, Optional
from pathlib import Path

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    connections,
    utility,
)
from pymilvus import FieldSchema, DataType

from .schemas import get_milvus_schema, DENSE_INDEX_CONFIG, SPARSE_INDEX_CONFIG


class MilvusClient:
    """Milvus 客户端封装"""

    def __init__(
        self,
        uri: str = "milvus_lite.db",
        collection_name: str = "doc_chunks",
        dense_dim: int = 1024
    ):
        """
        初始化 Milvus 客户端

        Args:
            uri: Milvus 连接 URI
            collection_name: Collection 名称
            dense_dim: Dense 向量维度
        """
        self.uri = uri
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self.collection: Optional[Collection] = None

        self._connect()

    def _connect(self):
        """连接到 Milvus"""
        connections.connect("default", uri=self.uri)
        logger.info(f"已连接到 Milvus: {self.uri}")

    def create_collection(self, drop_existing: bool = False) -> Collection:
        """
        创建 Collection

        Args:
            drop_existing: 如果 Collection 已存在是否删除

        Returns:
            Collection: 创建的 Collection 对象
        """
        # 如果已存在且需要删除
        if utility.has_collection(self.collection_name):
            if drop_existing:
                utility.drop_collection(self.collection_name)
                logger.info(f"已删除现有 Collection: {self.collection_name}")
            else:
                logger.info(f"Collection 已存在: {self.collection_name}")
                self.collection = Collection(self.collection_name)
                return self.collection

        # 创建 Schema
        schema = get_milvus_schema(dense_dim=self.dense_dim)

        # 创建 Collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        logger.info(f"已创建 Collection: {self.collection_name}")

        # 创建索引
        self._create_indexes()

        return self.collection

    def _create_indexes(self):
        """创建向量索引"""
        # Dense 向量索引
        self.collection.create_index(
            field_name="dense_vector",
            index_params=DENSE_INDEX_CONFIG
        )
        logger.info(f"已创建 Dense 向量索引: {DENSE_INDEX_CONFIG['index_type']}")

        # Sparse 向量索引
        self.collection.create_index(
            field_name="sparse_vector",
            index_params=SPARSE_INDEX_CONFIG
        )
        logger.info(f"已创建 Sparse 向量索引: {SPARSE_INDEX_CONFIG['index_type']}")

    def load_collection(self):
        """加载 Collection 到内存"""
        if self.collection is None:
            self.collection = Collection(self.collection_name)

        self.collection.load()
        logger.info(f"已加载 Collection 到内存: {self.collection_name}")

    def insert_data(
        self,
        ids: List[str],
        doc_ids: List[str],
        chunk_ids: List[str],
        titles: List[str],
        contents: List[str],
        metadata_list: List[dict],
        dense_vectors: List[List[float]],
        sparse_vectors: List[dict]
    ):
        """
        插入数据到 Collection

        Args:
            ids: 主键列表
            doc_ids: 文档 ID 列表
            chunk_ids: 块 ID 列表
            titles: 标题列表
            contents: 内容列表
            metadata_list: 元数据列表
            dense_vectors: Dense 向量列表
            sparse_vectors: Sparse 向量列表
        """
        data = [
            ids,
            doc_ids,
            chunk_ids,
            titles,
            contents,
            metadata_list,
            dense_vectors,
            sparse_vectors
        ]

        insert_result = self.collection.insert(data)
        self.collection.flush()

        logger.info(f"已插入 {len(ids)} 条数据到 Milvus")

        return insert_result

    def get_collection(self) -> Collection:
        """获取 Collection 对象"""
        if self.collection is None:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
            else:
                raise ValueError(f"Collection 不存在: {self.collection_name}，请先调用 create_collection()")
        return self.collection

    def get_stats(self) -> dict:
        """获取 Collection 统计信息"""
        self.load_collection()
        num_entities = self.collection.num_entities
        return {
            "collection_name": self.collection_name,
            "num_entities": num_entities
        }

    def disconnect(self):
        """断开连接"""
        connections.disconnect("default")
        logger.info("已断开 Milvus 连接")


if __name__ == "__main__":
    # 测试代码
    client = MilvusClient(
        uri="milvus_lite.db",
        collection_name="test_collection",
        dense_dim=1024
    )

    # 创建 Collection
    collection = client.create_collection(drop_existing=True)

    # 获取统计信息
    stats = client.get_stats()
    print(f"Collection 统计: {stats}")

    # 断开连接
    client.disconnect()
