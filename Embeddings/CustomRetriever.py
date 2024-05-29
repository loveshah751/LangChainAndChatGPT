from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain_community.vectorstores.chroma import Chroma


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def _get_relevant_documents(self, query, **kwargs):
        return []

    def get_relevant_documents(self, query, **kwargs):
        currentEmbeddings = self.embeddings.embed_query(query)
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=currentEmbeddings,
            lambda_mult=0.8)

    async def aget_relevant_documents(self, query, **kwargs):
        return []
