"""
GLM Dense Embedding 模型
"""

import time
from typing import List, Union

import numpy as np
from loguru import logger
from zhipuai import ZhipuAI


class GLMEmbedding:
    """
    智谱 AI GLM Embedding 模型

    使用智谱 AI Embedding API 生成稠密向量
    """

    def __init__(self, api_key: str, model: str = "embedding-3", auto_detect_dim: bool = True):
        """
        初始化 GLM Embedding 模型

        Args:
            api_key: 智谱 AI API Key
            model: 模型名称，默认为 embedding-3
            auto_detect_dim: 是否自动检测向量维度
        """
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
        self.dimension = 1024  # 默认维度

        # 自动检测向量维度
        if auto_detect_dim:
            self.dimension = self._detect_dimension()

        logger.info(f"GLM Embedding 模型初始化完成: {model}, 维度: {self.dimension}")

    def _detect_dimension(self) -> int:
        """检测向量维度"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input="test"
            )
            dim = len(response.data[0].embedding)
            logger.info(f"自动检测到向量维度: {dim}")
            return dim
        except Exception as e:
            logger.warning(f"无法检测向量维度，使用默认值 1024: {e}")
            return 1024

    def encode(self, texts: List[str], batch_size: int = 10) -> np.ndarray:
        """
        批量生成稠密向量

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            np.ndarray: 向量数组，形状为 (len(texts), dimension)
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            for text in batch_texts:
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=text
                    )
                    embedding = response.data[0].embedding
                    embeddings.append(embedding)

                except Exception as e:
                    logger.error(f"生成向量失败: {e}, 文本: {text[:50]}...")
                    # 使用零向量作为降级方案
                    embeddings.append([0.0] * self.dimension)

                # 避免触发 API 限流
                time.sleep(0.1)

        return np.array(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """
        单个文本生成向量

        Args:
            text: 输入文本

        Returns:
            np.ndarray: 向量数组
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"生成向量失败: {e}, 文本: {text[:50]}...")
            return np.zeros(self.dimension, dtype=np.float32)

    @property
    def dim(self) -> int:
        """返回向量维度"""
        return self.dimension


if __name__ == "__main__":
    # 测试代码
    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        print("请设置 GLM_API_KEY 环境变量")
        exit(1)

    model = GLMEmbedding(api_key=api_key)

    # 测试单个文本
    text = "这是一个测试文本，用于验证 GLM Embedding 模型。"
    embedding = model.encode_single(text)
    print(f"向量维度: {embedding.shape}")
    print(f"向量前 10 个值: {embedding[:10]}")

    # 测试批量编码
    texts = [
        "人工智能是计算机科学的一个分支",
        "深度学习是机器学习的一个子领域",
        "自然语言处理是 AI 的重要应用"
    ]
    embeddings = model.encode(texts)
    print(f"批量编码结果形状: {embeddings.shape}")
