"""
SeekDB 客户端封装，支持 GLM Embedding
"""

import pyseekdb
from loguru import logger
from typing import List, Optional, Union

from src.models.dense_embedding import GLMEmbedding


class GLMEmbeddingFunction:
    """
    GLM Embedding Function for SeekDB
    将 GLM Embedding API 适配为 SeekDB 的 EmbeddingFunction 接口
    """

    def __init__(self, api_key: str, model: str = "embedding-3"):
        """
        初始化 GLM Embedding Function

        Args:
            api_key: GLM API Key
            model: GLM 模型名称
        """
        self.api_key = api_key
        self.model = model
        self._model = None

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        if self._model is None:
            self._model = GLMEmbedding(
                api_key=self.api_key,
                model=self.model,
                auto_detect_dim=True
            )
        return self._model.dimension

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        生成向量嵌入

        Args:
            input: 单个文本或文本列表

        Returns:
            向量列表
        """
        if self._model is None:
            self._model = GLMEmbedding(
                api_key=self.api_key,
                model=self.model,
                auto_detect_dim=True
            )

        # 处理单个字符串输入
        if isinstance(input, str):
            input = [input]

        # 处理空输入
        if not input:
            return []

        # 批量生成向量
        embeddings = self._model.encode(input, batch_size=10)

        # 转换为列表格式
        return embeddings.tolist()


class SeekDBClient:
    """SeekDB 客户端封装，支持嵌入式模式和服务器模式"""

    def __init__(
        self,
        db_path: str = "seekdb.db",
        collection_name: str = "doc_chunks",
        host: Optional[str] = None,
        port: int = 2881,
        user: str = "root",
        password: str = "",
        use_server: bool = False,
        glm_api_key: Optional[str] = None,
        glm_model: str = "embedding-3",
        use_glm_embedding: bool = True,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.use_server = use_server
        self.glm_api_key = glm_api_key
        self.glm_model = glm_model
        self.use_glm_embedding = use_glm_embedding
        self.admin_client = None
        self.client = None
        self.collection = None
        self.embedding_function = None

        self._connect()
        self._init_embedding_function()

    def _init_embedding_function(self):
        """初始化 Embedding Function"""
        if self.use_glm_embedding and self.glm_api_key:
            try:
                self.embedding_function = GLMEmbeddingFunction(
                    api_key=self.glm_api_key,
                    model=self.glm_model
                )
                logger.info(f"SeekDB 将使用 GLM Embedding: {self.glm_model}, 维度: {self.embedding_function.dimension}")
            except Exception as e:
                logger.warning(f"GLM Embedding 初始化失败: {e}，将使用默认 Embedding Function")
                self.embedding_function = None
        else:
            self.embedding_function = None

    def _connect(self):
        """连接到 SeekDB（嵌入式模式或服务器模式）"""
        try:
            if self.use_server:
                # 服务器模式（用于 macOS/Windows）
                self._connect_server()
            else:
                # 嵌入式模式（仅 Linux）
                self._connect_embedded()
        except Exception as e:
            logger.error(f"SeekDB 连接失败: {e}")
            raise

    def _connect_embedded(self):
        """连接到嵌入式 SeekDB（仅 Linux）"""
        try:
            # 创建 Admin Client 和数据库
            self.admin_client = pyseekdb.AdminClient(path=self.db_path)
            db_name = self.db_path.replace(".db", "")
            try:
                self.admin_client.create_database(db_name)
            except Exception:
                pass  # 数据库可能已存在

            # 创建 Client
            self.client = pyseekdb.Client(path=self.db_path, database=db_name)
            logger.info(f"SeekDB 嵌入式模式连接成功: {self.db_path}")
        except Exception as e:
            logger.error(f"SeekDB 嵌入式模式连接失败: {e}")
            raise

    def _connect_server(self):
        """连接到远程 SeekDB 服务器（跨平台）"""
        try:
            # 服务器模式的 Admin Client
            self.admin_client = pyseekdb.AdminClient(
                host=self.host or "localhost",
                port=self.port,
                user=self.user,
                password=self.password
            )

            # 创建数据库（使用租户 'sys'）
            db_name = "hybrid_search_test"
            try:
                self.admin_client.create_database(db_name, tenant="sys")
                logger.info(f"SeekDB 数据库创建成功: {db_name}")
            except Exception:
                logger.info(f"SeekDB 数据库已存在: {db_name}")

            # 创建 Client
            self.client = pyseekdb.Client(
                host=self.host or "localhost",
                port=self.port,
                database=db_name,
                user=self.user,
                password=self.password
            )
            logger.info(f"SeekDB 服务器模式连接成功: {self.host or 'localhost'}:{self.port}")
        except Exception as e:
            logger.error(f"SeekDB 服务器模式连接失败: {e}")
            raise

    def create_collection(self, drop_existing: bool = False):
        """创建 Collection"""
        if drop_existing:
            self._drop_collection()

        # 使用 GLM Embedding Function 或默认的
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

        if self.embedding_function:
            logger.info(f"SeekDB Collection 创建成功: {self.collection_name} (GLM Embedding, {self.embedding_function.dimension}维)")
        else:
            logger.info(f"SeekDB Collection 创建成功: {self.collection_name} (默认 Embedding)")
        return self.collection

    def _drop_collection(self):
        """删除 Collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"SeekDB Collection 已删除: {self.collection_name}")
        except Exception:
            pass

    def insert_data(
        self,
        ids: List[str],
        doc_ids: List[str],
        chunk_ids: List[str],
        titles: List[str],
        contents: List[str],
        metadata_list: List[dict],
    ):
        """
        插入数据到 Collection（向量自动生成）
        """
        # 构建增强的 metadata
        enhanced_metadata = []
        for i, meta in enumerate(metadata_list):
            enhanced = {
                **meta,
                "doc_id": doc_ids[i],
                "chunk_id": chunk_ids[i],
                "title": titles[i]
            }
            enhanced_metadata.append(enhanced)

        self.collection.add(
            ids=ids,
            documents=contents,  # SeekDB 自动生成向量
            metadatas=enhanced_metadata
        )
        logger.info(f"SeekDB 插入数据: {len(ids)} 条")

    def get_collection(self):
        """获取 Collection 对象"""
        if self.collection is None:
            self.collection = self.client.get_collection(
                self.collection_name,
                embedding_function=self.embedding_function
            )
        return self.collection

    def get_stats(self) -> dict:
        """获取统计信息"""
        collection = self.get_collection()
        return {
            "collection_name": self.collection_name,
            "num_entities": collection.count()
        }

    def disconnect(self):
        """断开连接"""
        self.client = None
