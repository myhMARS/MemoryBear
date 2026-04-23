"""TripletExtractionStep — critical step for extracting entities and triplets.

Replaces the legacy ``TripletExtractor`` with the unified ExtractionStep paradigm.
Predicate filtering against the ontology whitelist is performed in ``parse_response``.
"""

import logging
from typing import Any

from app.core.memory.models.triplet_models import TripletExtractionResponse
from app.core.memory.utils.data.ontology import PREDICATE_DEFINITIONS, Predicate
from app.core.memory.utils.prompt.prompt_utils import render_triplet_extraction_prompt

from .base import ExtractionStep, StepContext
from .schema import EntityItem, TripletItem, TripletStepInput, TripletStepOutput

logger = logging.getLogger(__name__)


class TripletExtractionStep(ExtractionStep[TripletStepInput, TripletStepOutput]):
    """Extract knowledge triplets and entities from a single statement.

    This is a **critical** step — failure aborts the pipeline after retries.

    Config params bound at init (from ``StepContext.config``):
        * ``ontology_types`` — predefined ontology types for entity classification
        * ``predicate_instructions`` — predicate definition guidance for the LLM
        * ``json_schema`` — JSON schema for the expected LLM output
    """

    def __init__(
        self,
        context: StepContext,
        ontology_types: Any = None,
    ) -> None:
        super().__init__(context)
        self.ontology_types = ontology_types
        self.predicate_instructions = PREDICATE_DEFINITIONS
        self.json_schema = TripletExtractionResponse.model_json_schema()
        self._allowed_predicates = {p.value for p in Predicate}

    # ── Identity ──

    @property
    def name(self) -> str:
        return "triplet_extraction"

    @property
    def is_critical(self) -> bool:
        return True

    # ── Lifecycle ──

    async def render_prompt(self, input_data: TripletStepInput) -> str:
        # Build chunk_content from supporting_context for pronoun resolution
        chunk_content = "\n".join(
            f"{m.role}: {m.msg}" for m in input_data.supporting_context.msgs
        ) if input_data.supporting_context.msgs else ""

        return await render_triplet_extraction_prompt(
            statement=input_data.statement_text,
            chunk_content=chunk_content,
            json_schema=self.json_schema,
            predicate_instructions=self.predicate_instructions,
            language=self.language,
            ontology_types=self.ontology_types,
            speaker=input_data.speaker,
        )

    async def call_llm(self, prompt: Any) -> Any:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at extracting knowledge triplets and entities "
                    "from text. Follow the provided instructions carefully and return valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return await self.llm_client.response_structured(
            messages, TripletExtractionResponse
        )

    async def parse_response(
        self, raw_response: Any, input_data: TripletStepInput
    ) -> TripletStepOutput:
        if not hasattr(raw_response, "triplets"):
            return self.get_default_output()

        # Filter triplets to allowed predicates from ontology whitelist
        filtered_triplets = [
            TripletItem(
                subject_name=t.subject_name,
                subject_id=t.subject_id,
                predicate=t.predicate,
                object_name=t.object_name,
                object_id=t.object_id,
            )
            for t in raw_response.triplets
            if getattr(t, "predicate", "") in self._allowed_predicates
        ]

        entities = [
            EntityItem(
                entity_idx=e.entity_idx,
                name=e.name,
                type=e.type,
                description=e.description,
                is_explicit_memory=getattr(e, "is_explicit_memory", False),
            )
            for e in (raw_response.entities or [])
        ]

        return TripletStepOutput(entities=entities, triplets=filtered_triplets)

    def get_default_output(self) -> TripletStepOutput:
        return TripletStepOutput(entities=[], triplets=[])
