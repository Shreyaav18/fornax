import asyncio
import os
import logging
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from alembic import context
from db.models import Base
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL environment variable is not set for Alembic migrations")

config.set_main_option("sqlalchemy.url", DATABASE_URL)


def run_migrations_offline() -> None:
    try:
        context.configure(
            url=DATABASE_URL,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            compare_type=True
        )

        with context.begin_transaction():
            context.run_migrations()

        logger.info("Offline migrations completed")

    except Exception as e:
        logger.error(f"Offline migration failed: {e}")
        raise


def do_run_migrations(connection: Connection) -> None:
    try:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True
        )

        with context.begin_transaction():
            context.run_migrations()

    except Exception as e:
        logger.error(f"Migration execution failed: {e}")
        raise


async def run_async_migrations() -> None:
    try:
        connectable = create_async_engine(DATABASE_URL, poolclass=pool.NullPool)

        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)

        await connectable.dispose()
        logger.info("Async migrations completed")

    except Exception as e:
        logger.error(f"Async migration failed: {e}")
        raise


def run_migrations_online() -> None:
    try:
        asyncio.run(run_async_migrations())
    except Exception as e:
        logger.error(f"Online migration runner failed: {e}")
        raise


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()