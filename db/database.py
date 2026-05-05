import logging
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from db.models import Base
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL environment variable is not set")

if not DATABASE_URL.startswith("postgresql+asyncpg://"):
    raise ValueError(
        f"DATABASE_URL must use postgresql+asyncpg:// scheme for async support, got: {DATABASE_URL[:30]}..."
    )

try:
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800
    )
    logger.info("Async database engine created")

except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

async def get_async_session() -> AsyncSession:
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()

        except SQLAlchemyError as e:
            logger.error(f"Database session error, rolling back: {e}")
            await session.rollback()
            raise

        except Exception as e:
            logger.error(f"Unexpected error in database session, rolling back: {e}")
            await session.rollback()
            raise

        finally:
            await session.close()

async def init_db():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def close_db():
    try:
        await engine.dispose()
        logger.info("Database engine disposed")

    except Exception as e:
        logger.error(f"Failed to dispose database engine: {e}")
        raise