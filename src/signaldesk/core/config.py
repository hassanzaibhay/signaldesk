"""The single typed settings object.

Every environment key the project reads is declared here once, with a type and a
default where a default is sensible. Django settings modules, the HTTP client,
the Celery app and the DuckDB factory all read from this object rather than from
``os.environ``, so there is exactly one place to look for what is configurable.

Keys that must be set explicitly (``DJANGO_SECRET_KEY``) carry no default, so a
missing value fails loudly at startup instead of silently running on a
placeholder.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveInt = Annotated[int, Field(gt=0)]


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


class Settings(BaseSettings):
    """Typed view of the environment, loaded once per process."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Django ----
    django_settings_module: str = "signaldesk.web.config.settings.dev"
    django_secret_key: str
    django_debug: bool = False
    django_allowed_hosts: str = "localhost,127.0.0.1"

    # ---- Postgres ----
    postgres_host: str = "postgres"
    postgres_port: PositiveInt = 5432
    postgres_db: str = "signaldesk"
    postgres_user: str = "signaldesk"
    postgres_password: str = ""

    # ---- Redis and Celery ----
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # ---- Paths inside the container ----
    data_dir: Path = Path("/data")
    cache_dir: Path = Path("/data/cache")
    model_dir: Path = Path("/data/models")

    # ---- Ingest ----
    faers_start_quarter: str = "2012Q4"
    openfda_api_key: str = ""
    ncbi_email: str = ""
    ncbi_api_key: str = ""
    http_user_agent: str = "signaldesk/0.1 (+https://github.com/hassanzaibhay/signaldesk)"

    # ---- Retrieval ----
    embedding_model: str = "ncbi/MedCPT-Article-Encoder"
    query_encoder_model: str = "ncbi/MedCPT-Query-Encoder"
    reranker_model: str = "ncbi/MedCPT-Cross-Encoder"
    ablation_dense_model: str = "BAAI/bge-m3"
    chunk_target_tokens: PositiveInt = 400
    chunk_overlap_tokens: int = Field(default=80, ge=0)
    dense_top_k: PositiveInt = 50
    sparse_top_k: PositiveInt = 50
    rrf_k: PositiveInt = 60
    rerank_top_k: PositiveInt = 8
    hnsw_ef_search: PositiveInt = 100
    context_token_budget: PositiveInt = 6000

    # ---- Model router, ordered failover ----
    llm_provider_chain: str = "gemini,groq,cerebras,ollama"
    llm_generation_model: str = "gemini-2.5-flash"
    llm_judge_provider: str = "groq"
    llm_judge_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    llm_max_attempts: PositiveInt = 3
    llm_timeout_seconds: PositiveInt = 90

    # ---- Agent budgets ----
    agent_max_steps: PositiveInt = 6
    agent_max_calls_per_tool: PositiveInt = 2
    agent_wall_clock_seconds: PositiveInt = 45

    @model_validator(mode="after")
    def _overlap_fits_inside_the_chunk(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            message = (
                "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS "
                f"(got {self.chunk_overlap_tokens} >= {self.chunk_target_tokens})"
            )
            raise ValueError(message)
        return self

    @property
    def allowed_hosts(self) -> list[str]:
        """``DJANGO_ALLOWED_HOSTS`` as Django wants it."""
        return list(_split_csv(self.django_allowed_hosts))

    @property
    def provider_chain(self) -> tuple[str, ...]:
        """Model providers in failover order."""
        return _split_csv(self.llm_provider_chain)

    @property
    def database_url(self) -> str:
        """libpq URL for the system-of-record database."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def http_cache_dir(self) -> Path:
        """Directory holding the on-disk HTTP response cache."""
        return self.cache_dir / "http"

    @property
    def duckdb_path(self) -> Path:
        """The analytics database file."""
        return self.data_dir / "analytics.duckdb"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, built on first use.

    Cached because reading and validating the environment repeatedly is wasted
    work and because callers should observe one consistent configuration.
    """
    # Every field is populated from the environment; the type checker cannot see
    # that, so it asks for the one field that has no default.
    return Settings()  # type: ignore[call-arg]
