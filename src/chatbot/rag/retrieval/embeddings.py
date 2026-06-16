"""Azure-style embeddings wrapper used by the vector store."""
from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings
from openai import AzureOpenAI

from chatbot.settings.app_config import ConfigManager

def build_azure_embeddings(config_manager: ConfigManager) -> Embeddings:
    emb = config_manager.get_embedding_config()
    llm = config_manager.get_llm_config()
    return AzureEmbeddings(
        endpoint=config_manager.endpoint_url,
        api_key=config_manager.api_key,
        deployment=str(emb.deployment_name),
        api_version=str(llm.api_version or ""),
        batch_size=max(1, emb.chunk_size),
    )

class AzureEmbeddings(Embeddings):
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str,
        batch_size: int,
    ) -> None:
        self._deployment = deployment
        self._batch_size = batch_size
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text or ""])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self._batch_size):
            chunk = [t if t is not None else "" for t in texts[start : start + self._batch_size]]
            resp = self._client.embeddings.create(model=self._deployment, input=chunk)
            by_index = sorted(resp.data, key=lambda row: row.index)
            vectors.extend(row.embedding for row in by_index)
        return vectors
