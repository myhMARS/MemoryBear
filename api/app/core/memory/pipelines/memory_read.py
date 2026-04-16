from app.core.memory.enums import SearchStrategy, StorageType
from app.core.memory.models.service_models import MemorySearchResult
from app.core.memory.pipelines.base_pipeline import ModelClientMixin, DBRequiredPipeline
from app.core.memory.read_services.content_search import Neo4jSearchService, RAGSearchService
from app.core.memory.read_services.query_preprocessor import QueryPreprocessor


class ReadPipeLine(ModelClientMixin, DBRequiredPipeline):
    async def run(self, query: str, search_switch: SearchStrategy, limit: int = 10, includes=None) -> MemorySearchResult:
        query = QueryPreprocessor.process(query)
        if self.ctx.storage_type == StorageType.RAG:
            return await self._rag_read(query, limit)
        match search_switch:
            case SearchStrategy.DEEP:
                return await self._deep_read(query, limit, includes)
            case SearchStrategy.NORMAL:
                return await self._normal_read(query, limit, includes)
            case SearchStrategy.QUICK:
                return await self._quick_read(query, limit, includes)
            case _:
                raise RuntimeError("Unsupported search strategy")

    async def _rag_read(self, query: str, limit: int) -> MemorySearchResult:
        service = RAGSearchService(
            self.ctx
        )
        return await service.search()

    async def _deep_read(self, query: str, limit: int, includes=None) -> MemorySearchResult:
        pass

    async def _normal_read(self, query: str, limit: int, includes=None) -> MemorySearchResult:
        pass

    async def _quick_read(self, query: str, limit: int, includes=None) -> MemorySearchResult:
        search_service = Neo4jSearchService(
            self.ctx,
            self.get_embedding_client(self.db, self.ctx.memory_config.embedding_model_id),
            includes=includes,
        )
        return await search_service.search(query, limit)
