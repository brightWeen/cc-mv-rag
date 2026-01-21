"""
文档分块模块
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from loguru import logger


@dataclass
class Chunk:
    """文档块"""
    doc_id: str
    chunk_id: str
    title: str
    content: str
    metadata: dict

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata
        }


class DocumentChunker:
    """
    文档分块器

    将长文档按照指定大小和重叠进行分块，用于向量检索
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_chunks_per_doc: int = 50,
        separator: str = "\n\n"
    ):
        """
        初始化分块器

        Args:
            chunk_size: 每块的字符数
            chunk_overlap: 块之间的重叠字符数
            max_chunks_per_doc: 单个文档最大分块数
            separator: 分段分隔符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks_per_doc = max_chunks_per_doc
        self.separator = separator

        logger.info(f"文档分块器初始化: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def chunk(self, documents: List[Dict]) -> List[Chunk]:
        """
        对文档列表进行分块

        Args:
            documents: 文档列表，每个文档包含 doc_id, title, content, metadata

        Returns:
            List[Chunk]: 分块后的文档块列表
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_single(doc)
            all_chunks.extend(chunks)

        logger.info(f"分块完成: {len(documents)} 个文档 -> {len(all_chunks)} 个块")
        return all_chunks

    def chunk_single(self, document: Dict) -> List[Chunk]:
        """
        对单个文档进行分块

        Args:
            document: 文档字典

        Returns:
            List[Chunk]: 分块列表
        """
        doc_id = document.get("doc_id", "")
        title = document.get("title", "")
        content = document.get("content", "")
        metadata = document.get("metadata", {})

        # 如果标题不为空，将标题添加到内容中
        if title:
            full_content = f"{title}\n\n{content}"
        else:
            full_content = content

        # 按照分隔符分段
        sections = full_content.split(self.separator)
        sections = [s.strip() for s in sections if s.strip()]

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for section in sections:
            # 如果当前块加上新段落超过大小限制，先保存当前块
            if current_chunk and len(current_chunk) + len(section) + len(self.separator) > self.chunk_size:
                chunks.append(current_chunk.strip())
                # 保留重叠部分
                if self.chunk_overlap > 0:
                    current_chunk = current_chunk[-self.chunk_overlap:] + self.separator + section
                else:
                    current_chunk = section
                chunk_index += 1

                if len(chunks) >= self.max_chunks_per_doc:
                    break
            else:
                if current_chunk:
                    current_chunk += self.separator + section
                else:
                    current_chunk = section

        # 添加最后一个块
        if current_chunk and len(chunks) < self.max_chunks_per_doc:
            chunks.append(current_chunk.strip())

        # 创建 Chunk 对象
        chunk_objects = []
        for i, chunk_content in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i:03d}"
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk_content)
            }

            chunk_objects.append(Chunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                title=title,
                content=chunk_content,
                metadata=chunk_metadata
            ))

        return chunk_objects

    def chunk_text(self, text: str, doc_id: str = "doc_001") -> List[Chunk]:
        """
        对纯文本进行分块

        Args:
            text: 输入文本
            doc_id: 文档 ID

        Returns:
            List[Chunk]: 分块列表
        """
        document = {
            "doc_id": doc_id,
            "title": "",
            "content": text,
            "metadata": {}
        }
        return self.chunk_single(document)


if __name__ == "__main__":
    # 测试代码
    sample_text = """
    人工智能是计算机科学的一个分支，它致力于创建能够执行通常需要人类智能的任务。
    这些任务包括视觉感知、语音识别、决策制定和语言翻译等。

    机器学习是人工智能的核心技术之一。它使计算机能够从数据中学习，而不是通过明确的编程。
    常见的机器学习算法包括监督学习、无监督学习和强化学习。

    深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的表示。
    深度学习在图像识别、自然语言处理和语音识别等领域取得了显著的成果。
    """

    chunker = DocumentChunker(
        chunk_size=200,
        chunk_overlap=30,
        separator="\n\n"
    )

    chunks = chunker.chunk_text(sample_text, doc_id="test_doc")

    print(f"总共生成 {len(chunks)} 个块:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ({chunk.chunk_id}) ---")
        print(f"内容: {chunk.content[:100]}...")
        print(f"长度: {len(chunk.content)}")
        print()
