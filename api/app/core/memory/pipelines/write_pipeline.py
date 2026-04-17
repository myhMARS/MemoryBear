"""
WritePipeline — 记忆写入流水线

编排完整的写入流程：预处理 → 萃取 → 存储 → 聚类 → 摘要。
不包含业务逻辑实现，只做步骤编排和数据传递。

设计原则：
- Pipeline 不直接操作数据库，通过 Engine / Repository 完成
- Pipeline 不包含 LLM 调用逻辑，通过 ExtractionOrchestrator 完成
- Pipeline 负责资源生命周期管理（客户端初始化 / 连接关闭）
- Pipeline 负责错误边界划分（哪些错误中断流程，哪些吞掉继续）

依赖方向：Facade → Pipeline → Engine → Repository（单向，不允许反向调用）
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from app.core.memory.models.graph_models import ExtractedEntityNode
    from app.core.memory.models.message_models import DialogData
    from app.schemas.memory_config_schema import MemoryConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────


@dataclass
class ExtractionResult:
    """萃取步骤的结构化输出，替代 ExtractionOrchestrator.run() 返回的裸元组。

    字段与 ExtractionOrchestrator.run() 的 10 元素返回值一一对应：
      [0] dialogue_nodes      → self.dialogue_nodes
      [1] chunk_nodes         → self.chunk_nodes
      [2] statement_nodes     → self.statement_nodes
      [3] entity_nodes        → self.entity_nodes
      [4] perceptual_nodes    → self.perceptual_nodes
      [5] stmt_chunk_edges    → self.stmt_chunk_edges
      [6] stmt_entity_edges   → self.stmt_entity_edges
      [7] entity_entity_edges → self.entity_entity_edges
      [8] perceptual_edges    → self.perceptual_edges
      [9] dialog_data_list    → self.dialog_data_list

    注意：字段类型使用 List[Any] 而非具体的 graph_models 类型，
    避免在模块加载时触发循环依赖。Pipeline 只做数据传递，不检查具体类型。
    """

    dialogue_nodes: List[Any]
    chunk_nodes: List[Any]
    statement_nodes: List[Any]
    entity_nodes: List[Any]
    perceptual_nodes: List[Any]
    stmt_chunk_edges: List[Any]
    stmt_entity_edges: List[Any]
    entity_entity_edges: List[Any]
    perceptual_edges: List[Any]
    dialog_data_list: List[Any]

    @property
    def stats(self) -> Dict[str, int]:
        """返回统计摘要，用于 WriteResult 和日志"""
        return {
            "dialogue_count": len(self.dialogue_nodes),
            "chunk_count": len(self.chunk_nodes),
            "statement_count": len(self.statement_nodes),
            "entity_count": len(self.entity_nodes),
            "perceptual_count": len(self.perceptual_nodes),
            "relation_count": len(self.entity_entity_edges),
        }


@dataclass
class WriteResult:
    """写入流水线的最终输出，返回给 MemoryService / MemoryAgentService"""

    status: str  # "success" | "pilot_complete" | "failed"
    extraction: Optional[Dict[str, int]] = None  # ExtractionResult.stats
    error: Optional[str] = None  # 失败时的错误信息
    elapsed_seconds: float = 0.0  # 总耗时（秒）


# ──────────────────────────────────────────────
# WritePipeline
# ──────────────────────────────────────────────


class WritePipeline:
    """
    记忆写入流水线

    编排完整的写入流程：预处理 → 萃取 → 存储 → 聚类 → 摘要。
    """

    def __init__(
        self,
        memory_config: MemoryConfig,
        end_user_id: str,
        language: str = "zh",
        progress_callback: Optional[
            Callable[[str, str, Optional[Dict[str, Any]]], Awaitable[None]]
        ] = None,
    ):
        """
        Args:
            memory_config: 不可变的记忆配置对象（从数据库加载）
            end_user_id: 终端用户 ID
            language: 语言 ("zh" | "en")
            progress_callback: 可选的进度回调，签名 (stage, message, data?) -> Awaitable[None]
        """
        self.memory_config = memory_config
        self.end_user_id = end_user_id
        self.language = language
        self.progress_callback = progress_callback

        # 延迟初始化的客户端
        self._llm_client = None
        self._embedder_client = None
        self._neo4j_connector = None

    # ──────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────

    async def run(
        self,
        messages: List[dict],
        ref_id: str = "",
        is_pilot_run: bool = False,
    ) -> WriteResult:
        """
        执行完整的写入流水线。

        Args:
            messages: 结构化消息 [{"role": "user"/"assistant", "content": "..."}]
            ref_id: 引用 ID，为空则自动生成
            is_pilot_run: 试运行模式（只萃取不写入）

        Returns:
            WriteResult 包含状态和统计信息
        """
        if not ref_id:
            ref_id = uuid.uuid4().hex

        mode = "试运行" if is_pilot_run else "正式"
        pipeline_start = time.time()

        logger.info(
            f"[WritePipeline] 开始 ({mode}) "
            f"config={self.memory_config.config_name}, "
            f"end_user={self.end_user_id}"
        )

        try:
            # 初始化客户端和连接
            self._init_clients()
            self._init_neo4j_connector()

            # Step 1: 预处理 - 消息分块
            step_start = time.time()
            chunked_dialogs = await self._preprocess(messages, ref_id)
            chunks_count = sum(len(d.chunks) for d in chunked_dialogs)
            logger.info(
                f"[WritePipeline] [1/5] 预处理：消息分块 "
                f"✔ {time.time() - step_start:.2f}s  chunks={chunks_count}"
            )

            # Step 2: 萃取 - 知识提取
            step_start = time.time()
            extraction_result = await self._extract(
                chunked_dialogs, is_pilot_run
            )
            stats = extraction_result.stats
            logger.info(
                f"[WritePipeline] [2/5] 萃取：知识提取 "
                f"✔ {time.time() - step_start:.2f}s  "
                f"entities={stats['entity_count']}, "
                f"statements={stats['statement_count']}, "
                f"relations={stats['relation_count']}"
            )

            # 试运行模式到此结束
            if is_pilot_run:
                elapsed = time.time() - pipeline_start
                logger.info(
                    f"[WritePipeline] 完成（试运行） ✔ {elapsed:.2f}s"
                )
                return WriteResult(
                    status="pilot_complete",
                    extraction=extraction_result.stats,
                    elapsed_seconds=elapsed,
                )

            # Step 3: 存储 - 写入 Neo4j
            step_start = time.time()
            await self._store(extraction_result)
            logger.info(
                f"[WritePipeline] [3/5] 存储：写入 Neo4j "
                f"✔ {time.time() - step_start:.2f}s"
            )

            # Step 4: 聚类 - 增量更新社区（异步，不阻塞）
            step_start = time.time()
            await self._cluster(extraction_result)
            logger.info(
                f"[WritePipeline] [4/5] 聚类：增量更新社区 "
                f"✔ {time.time() - step_start:.2f}s  mode=async"
            )

            # Step 5: 摘要 - 生成情景记忆摘要
            step_start = time.time()
            await self._summarize(chunked_dialogs)
            logger.info(
                f"[WritePipeline] [5/5] 摘要：生成情景记忆 "
                f"✔ {time.time() - step_start:.2f}s"
            )

            # 更新活动统计缓存
            await self._update_stats_cache(extraction_result)

            elapsed = time.time() - pipeline_start
            logger.info(
                f"[WritePipeline] 完成 ✔ {elapsed:.2f}s"
            )
            return WriteResult(
                status="success",
                extraction=extraction_result.stats,
                elapsed_seconds=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - pipeline_start
            logger.error(
                f"[WritePipeline] 失败 ✘ {elapsed:.2f}s  error={e}",
                exc_info=True,
            )
            raise

        finally:
            await self._cleanup()

    # ──────────────────────────────────────────────
    # Step 1: 预处理
    # ──────────────────────────────────────────────

    async def _preprocess(
        self, messages: List[dict], ref_id: str
    ) -> List[DialogData]:
        """
        预处理：消息校验 → 语义剪枝 → 对话分块。

        委托给 get_chunked_dialogs()，保持现有预处理逻辑不变。
        get_dialogs.py 内部已包含：
          - 消息格式校验（role/content 必填）
          - 语义剪枝（根据 config 中 pruning_enabled 决定）
          - DialogueChunker 分块
        """
        from app.core.memory.agent.utils.get_dialogs import get_chunked_dialogs

        return await get_chunked_dialogs(
            chunker_strategy=self.memory_config.chunker_strategy,
            end_user_id=self.end_user_id,
            messages=messages,
            ref_id=ref_id,
            config_id=str(self.memory_config.config_id),
        )

    # ──────────────────────────────────────────────
    # Step 2: 萃取
    # ──────────────────────────────────────────────

    async def _extract(
        self,
        chunked_dialogs: List[DialogData],
        is_pilot_run: bool,
    ) -> ExtractionResult:
        """
        萃取：初始化引擎 → 执行知识提取 → 返回结构化结果。

        ExtractionOrchestrator 作为萃取引擎被调用，
        Pipeline 不关心引擎内部的并行策略和提取细节。
        """
        from app.core.memory.storage_services.extraction_engine.extraction_orchestrator import (
            ExtractionOrchestrator,
        )
        from app.core.memory.utils.config.config_utils import get_pipeline_config

        pipeline_config = get_pipeline_config(self.memory_config)
        ontology_types = self._load_ontology_types()

        orchestrator = ExtractionOrchestrator(
            llm_client=self._llm_client,
            embedder_client=self._embedder_client,
            connector=self._neo4j_connector,
            config=pipeline_config,
            embedding_id=str(self.memory_config.embedding_model_id),
            language=self.language,
            ontology_types=ontology_types,
            progress_callback=self.progress_callback,
        )

        (
            dialogue_nodes,
            chunk_nodes,
            statement_nodes,
            entity_nodes,
            perceptual_nodes,
            stmt_chunk_edges,
            stmt_entity_edges,
            entity_entity_edges,
            perceptual_edges,
            dialog_data_list,
        ) = await orchestrator.run(chunked_dialogs, is_pilot_run=is_pilot_run)

        return ExtractionResult(
            dialogue_nodes=dialogue_nodes,
            chunk_nodes=chunk_nodes,
            statement_nodes=statement_nodes,
            entity_nodes=entity_nodes,
            perceptual_nodes=perceptual_nodes,
            stmt_chunk_edges=stmt_chunk_edges,
            stmt_entity_edges=stmt_entity_edges,
            entity_entity_edges=entity_entity_edges,
            perceptual_edges=perceptual_edges,
            dialog_data_list=dialog_data_list,
        )

    # ──────────────────────────────────────────────
    # Step 3: 存储
    # ──────────────────────────────────────────────

    async def _store(self, result: ExtractionResult) -> None:
        """
        存储：别名清洗 → Neo4j 写入（含死锁重试）。

        错误策略：
        - 别名清洗失败 → 警告日志，继续写入
        - Neo4j 写入死锁 → 指数退避重试 3 次
        - Neo4j 写入非死锁异常 → 直接抛出，中断流程
        """
        from app.repositories.neo4j.graph_saver import (
            save_dialog_and_statements_to_neo4j,
        )

        # 1. 写入前别名清洗（失败不中断）
        await self._clean_cross_role_aliases(result.entity_nodes)

        # 2. Neo4j 写入（含死锁重试）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                success = await save_dialog_and_statements_to_neo4j(
                    dialogue_nodes=result.dialogue_nodes,
                    chunk_nodes=result.chunk_nodes,
                    statement_nodes=result.statement_nodes,
                    entity_nodes=result.entity_nodes,
                    perceptual_nodes=result.perceptual_nodes,
                    statement_chunk_edges=result.stmt_chunk_edges,
                    statement_entity_edges=result.stmt_entity_edges,
                    entity_edges=result.entity_entity_edges,
                    perceptual_edges=result.perceptual_edges,
                    connector=self._neo4j_connector,
                )
                if success:
                    logger.info("Successfully saved all data to Neo4j")
                    return
                # 写入返回 False（部分失败）
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Neo4j 写入部分失败，重试 ({attempt + 2}/{max_retries})"
                    )
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    logger.error(
                        f"Neo4j 写入在 {max_retries} 次尝试后仍部分失败"
                    )
            except Exception as e:
                if self._is_deadlock(e) and attempt < max_retries - 1:
                    logger.warning(
                        f"Neo4j 死锁，重试 ({attempt + 2}/{max_retries})"
                    )
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    raise

    # ──────────────────────────────────────────────
    # Step 4: 聚类
    # ──────────────────────────────────────────────

    async def _cluster(self, result: ExtractionResult) -> None:
        """
        聚类：提交 Celery 异步任务进行增量社区更新。

        聚类不阻塞主写入流程，失败不影响写入结果。
        通过 Celery 异步执行，由 LabelPropagationEngine 完成实际计算。
        """
        if not result.entity_nodes:
            return

        try:
            from app.tasks import run_incremental_clustering

            new_entity_ids = [e.id for e in result.entity_nodes]
            task = run_incremental_clustering.apply_async(
                kwargs={
                    "end_user_id": self.end_user_id,
                    "new_entity_ids": new_entity_ids,
                    "llm_model_id": (
                        str(self.memory_config.llm_model_id)
                        if self.memory_config.llm_model_id
                        else None
                    ),
                    "embedding_model_id": (
                        str(self.memory_config.embedding_model_id)
                        if self.memory_config.embedding_model_id
                        else None
                    ),
                },
                priority=3,
            )
            logger.info(
                f"[Clustering] 增量聚类任务已提交 - "
                f"task_id={task.id}, entity_count={len(new_entity_ids)}"
            )
        except Exception as e:
            logger.error(
                f"[Clustering] 提交聚类任务失败（不影响主流程）: {e}",
                exc_info=True,
            )

    # ──────────────────────────────────────────────
    # Step 5: 摘要
    # （+ entity_description）
    # ──────────────────────────────────────────────

    async def _summarize(self, chunked_dialogs: List[DialogData]) -> None:
        """
        摘要：生成情景记忆摘要 → 写入 Neo4j。

        摘要生成失败不影响主流程（try/except 吞掉异常）。
        使用独立的 Neo4j 连接器，避免与主连接器的事务冲突。
        """
        from app.core.memory.storage_services.extraction_engine.knowledge_extraction.memory_summary import (
            memory_summary_generation,
        )
        from app.repositories.neo4j.add_edges import (
            add_memory_summary_statement_edges,
        )
        from app.repositories.neo4j.add_nodes import add_memory_summary_nodes
        from app.repositories.neo4j.neo4j_connector import Neo4jConnector

        try:
            summaries = await memory_summary_generation(
                chunked_dialogs,
                llm_client=self._llm_client,
                embedder_client=self._embedder_client,
                language=self.language,
            )
            ms_connector = Neo4jConnector()
            try:
                await add_memory_summary_nodes(summaries, ms_connector)
                await add_memory_summary_statement_edges(
                    summaries, ms_connector
                )
            finally:
                try:
                    await ms_connector.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Memory summary step failed: {e}", exc_info=True)

    # ──────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────

    def _init_clients(self) -> None:
        """
        从 MemoryConfig 构建 LLM 和 Embedding 客户端。

        使用 MemoryClientFactory 工厂模式，需要短暂的 DB session 来
        查询模型配置（API key、base_url 等），查询完毕立即释放。
        """
        from app.core.memory.utils.llm.llm_utils import MemoryClientFactory
        from app.db import get_db_context

        with get_db_context() as db:
            factory = MemoryClientFactory(db)
            self._llm_client = factory.get_llm_client_from_config(
                self.memory_config
            )
            self._embedder_client = factory.get_embedder_client_from_config(
                self.memory_config
            )
        logger.info("LLM and embedding clients constructed")

    def _init_neo4j_connector(self) -> None:
        """初始化 Neo4j 连接器。"""
        from app.repositories.neo4j.neo4j_connector import Neo4jConnector

        self._neo4j_connector = Neo4jConnector()

    def _load_ontology_types(self):
        """
        加载本体类型配置。

        如果 memory_config 中配置了 scene_id，则从数据库加载
        该场景关联的本体类型列表，用于指导三元组提取。
        """
        if not self.memory_config.scene_id:
            return None

        try:
            from app.core.memory.ontology_services.ontology_type_loader import (
                load_ontology_types_for_scene,
            )
            from app.db import get_db_context

            with get_db_context() as db:
                ontology_types = load_ontology_types_for_scene(
                    scene_id=self.memory_config.scene_id,
                    workspace_id=self.memory_config.workspace_id,
                    db=db,
                )
            if ontology_types:
                logger.info(
                    f"Loaded {len(ontology_types.types)} ontology types "
                    f"for scene_id: {self.memory_config.scene_id}"
                )
            return ontology_types
        except Exception as e:
            logger.warning(
                f"Failed to load ontology types for scene_id "
                f"{self.memory_config.scene_id}: {e}",
                exc_info=True,
            )
            return None

    async def _clean_cross_role_aliases(
        self, entity_nodes: List[ExtractedEntityNode]
    ) -> None:
        """
        清洗用户/AI助手实体之间的别名交叉污染。

        从 Neo4j 查询已有的 AI 助手别名，与本轮实体中的 AI 助手别名合并，
        确保用户实体的 aliases 不包含 AI 助手的名字。
        失败不中断主流程。
        """
        try:
            from app.core.memory.storage_services.extraction_engine.deduplication.deduped_and_disamb import (
                clean_cross_role_aliases,
                fetch_neo4j_assistant_aliases,
            )

            neo4j_assistant_aliases = set()
            if entity_nodes:
                eu_id = entity_nodes[0].end_user_id
                if eu_id:
                    neo4j_assistant_aliases = (
                        await fetch_neo4j_assistant_aliases(
                            self._neo4j_connector, eu_id
                        )
                    )
            clean_cross_role_aliases(
                entity_nodes,
                external_assistant_aliases=neo4j_assistant_aliases,
            )
            logger.info(
                f"别名清洗完成，AI助手别名排除集大小: "
                f"{len(neo4j_assistant_aliases)}"
            )
        except Exception as e:
            logger.warning(f"别名清洗失败（不影响主流程）: {e}")

    @staticmethod
    def _is_deadlock(e: Exception) -> bool:
        """判断异常是否为 Neo4j 死锁错误"""
        msg = str(e).lower()
        return "deadlockdetected" in msg or "deadlock" in msg

    async def _update_stats_cache(
        self, result: ExtractionResult
    ) -> None:
        """
        将提取统计写入 Redis 活动缓存，按 workspace_id 存储。
        失败不中断主流程。
        """
        try:
            from app.cache.memory.activity_stats_cache import (
                ActivityStatsCache,
            )

            stats = {
                "chunk_count": result.stats["chunk_count"],
                "statements_count": result.stats["statement_count"],
                "triplet_entities_count": result.stats["entity_count"],
                "triplet_relations_count": result.stats["relation_count"],
                "temporal_count": 0,
            }
            await ActivityStatsCache.set_activity_stats(
                workspace_id=str(self.memory_config.workspace_id),
                stats=stats,
            )
            logger.info(
                f"活动统计已写入 Redis: "
                f"workspace_id={self.memory_config.workspace_id}"
            )
        except Exception as e:
            logger.warning(
                f"写入活动统计缓存失败（不影响主流程）: {e}"
            )

    async def _cleanup(self) -> None:
        """
        清理资源：关闭 Neo4j 连接器和 HTTP 客户端。
        在 run() 的 finally 块中调用，确保资源释放。
        """
        # 关闭 Neo4j 连接器
        if self._neo4j_connector:
            try:
                await self._neo4j_connector.close()
            except Exception as e:
                logger.error(f"Error closing Neo4j connector: {e}")

        # 关闭 LLM/Embedder 底层 httpx 客户端
        # 防止 'RuntimeError: Event loop is closed' 在垃圾回收时触发
        for client_obj in (self._llm_client, self._embedder_client):
            try:
                underlying = getattr(
                    client_obj, "client", None
                ) or getattr(client_obj, "model", None)
                if underlying is None:
                    continue
                inner = getattr(underlying, "_model", underlying)
                http_client = getattr(inner, "async_client", None)
                if http_client is not None and hasattr(
                    http_client, "aclose"
                ):
                    await http_client.aclose()
            except Exception:
                pass
