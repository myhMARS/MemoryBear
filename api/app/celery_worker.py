"""
Celery Worker 入口点
用于启动 Celery Worker: celery -A app.celery_worker worker --loglevel=info
"""
from celery.signals import worker_process_init

from app.celery_app import celery_app
from app.core.logging_config import LoggingConfig, get_logger

# Initialize logging system for Celery worker
LoggingConfig.setup_logging()
logger = get_logger(__name__)
logger.info("Celery worker logging initialized")

# 导入任务模块以注册任务
import app.tasks


@worker_process_init.connect
def _reinit_db_pool(**kwargs):
    """
    prefork 子进程启动时重建 SQLAlchemy 连接池。
    
    fork() 后子进程继承了父进程的连接池和底层 TCP socket，
    多个子进程共享同一个 socket 会导致 PostgreSQL 连接损坏。
    dispose() 会关闭继承来的连接，后续操作会自动创建新连接。
    """
    from app.db import engine
    engine.dispose()
    logger.info("DB connection pool disposed for forked worker process")


__all__ = ['celery_app']
