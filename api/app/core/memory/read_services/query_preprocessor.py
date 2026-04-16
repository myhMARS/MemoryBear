import logging
import re

from app.core.memory.prompt import prompt_manager
from app.core.memory.utils.llm.llm_utils import StructResponse
from app.core.models import RedBearLLM
from app.schemas.memory_agent_schema import AgentMemoryDataset

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    @staticmethod
    def process(query: str) -> str:
        text = query.strip()
        if not text:
            return text

        text = re.sub(rf"{"|".join(AgentMemoryDataset.PRONOUN)}", AgentMemoryDataset.NAME, text)
        return text

    @staticmethod
    async def split(query: str, llm_client: RedBearLLM):
        system_prompt = prompt_manager.render(
            name="problem_split",
            history=[],
            sentence=query,
        )
        messages = [{"role": "system", "content": system_prompt}]
        try:
            sub_queries = await llm_client.ainvoke(messages) | StructResponse(mode='json')
        except Exception as e:
            logger.error(f"[QueryPreprocessor] Sub-question segmentation failed - {e}")
            sub_queries = None
        return sub_queries or query

    @staticmethod
    async def extension(query: str, llm_client: RedBearLLM):
        pass
