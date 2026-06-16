"""Single source of truth for runtime configuration.

Loads ``config/app_config.yaml`` into a Pydantic ``AppConfig`` and exposes a
``ConfigManager`` that satisfies both:

* the agentic/MCP layer (``load_config()``-style access),
* the vendored RAG modules (which read ``cfg.app_settings``, ``cfg.api_key``,
  ``cfg.vector_db_dir`` and the like).
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "app_config.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class AppMetadata(BaseModel):
    name: str
    version: str
    log_level: str = "INFO"

class LLMConfig(BaseModel):
    model: str
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    request_timeout_s: float = 60.0

class EmbeddingConfig(BaseModel):
    model: str
    chunk_size: int = 2048
    deployment_name: str

class TextSplitterConfig(BaseModel):
    chunk_size: int = Field(..., gt=0)
    chunk_overlap: int = Field(..., ge=0)

    @model_validator(mode="after")
    def overlap_lt_chunk(self) -> "TextSplitterConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self

class StorageConfig(BaseModel):
    chroma_db_path: str = "chroma_db"
    evaluation_results_path: str = "evaluation_results"
    logs_path: str = "logs"
    chroma_collection_name: str = "chatbot_corpus"

class AccessControlConfig(BaseModel):
    department_categories: Dict[str, List[str]] = Field(default_factory=dict)

class EvaluationConfig(BaseModel):
    precision_k_values: List[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    recall_k_values: List[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    max_docs_for_evaluation: int = 10
    golden_dataset_path: str = "golden_dataset.json"
    relevancy_threshold: float = 0.7
    correctness_threshold: float = 0.5

class HybridSearchConfig(BaseModel):
    enabled: bool = True
    sparse_k: int = 50
    rrf_k: int = 60
    fusion: str = "rrf"
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    bm25_persist_filename: str = "bm25_index.json"

class StructuredOutputConfig(BaseModel):
    enabled: bool = True
    max_context_chars: int = 12000

class DocumentProcessingConfig(BaseModel):
    normalize_text: bool = True
    extract_headline_skills: bool = True
    multimodal_enabled: bool = True

class PIIRedactionConfig(BaseModel):
    enabled: bool = True
    patterns: List[str] = Field(default_factory=lambda: ["email", "phone", "ssn"])

class GuardrailsConfig(BaseModel):
    enabled: bool = True
    require_user: bool = False
    max_user_input_chars: int = 2000
    max_llm_output_chars: int = 16000
    max_tool_output_chars: int = 60000
    max_json_write_chars: int = 500000
    max_context_chars: int = 60000
    max_history_chars: int = 120000
    blocked_patterns: List[str] = Field(default_factory=list)
    pii_redaction: PIIRedactionConfig = Field(default_factory=PIIRedactionConfig)
    allowed_url_schemes: List[str] = Field(default_factory=lambda: ["http", "https"])

class MCPConfig(BaseModel):
    rag_server: str
    disaster_server: str
    healthcare_server: str
    startup_timeout_s: float = 30.0

class DisastersConfig(BaseModel):
    csv_files: List[str]
    max_query_limit: int = 200
    default_query_limit: int = 20
    indexing_max_rows: int = Field(
        500,
        ge=1,
        description="Cap on EM-DAT rows turned into RAG narrative documents.",
    )
    indexing_strategy: str = Field(
        "recent",
        description="Default row-selection strategy: 'recent' or 'impact'.",
    )

class HealthcareConfig(BaseModel):
    entity_types: List[str] = Field(default_factory=list)
    max_input_chars: int = 8000

class AgentSpec(BaseModel):
    name: str
    prompt_file: str

class AgentsConfig(BaseModel):
    max_turns: int = 10
    triage: AgentSpec
    rag: AgentSpec
    disaster: AgentSpec
    healthcare: AgentSpec

class TracingConfig(BaseModel):
    enabled: bool = True
    exporters: List[str] = Field(default_factory=lambda: ["console"])
    redact_outputs: bool = True
    max_output_chars: int = 500

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s %(name)s %(levelname)s %(message)s"

class AppConfig(BaseModel):
    """Parsed ``app_config.yaml`` content."""

    model_config = ConfigDict(extra="ignore")

    app: AppMetadata
    llm: LLMConfig
    embeddings: EmbeddingConfig
    text_splitter: TextSplitterConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    access_control: AccessControlConfig = Field(default_factory=AccessControlConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    structured_output: StructuredOutputConfig = Field(default_factory=StructuredOutputConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    mcp: MCPConfig
    disasters: DisastersConfig
    healthcare: HealthcareConfig = Field(default_factory=HealthcareConfig)
    agents: AgentsConfig
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @staticmethod
    def project_path(relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p

class EnvironmentSettings(BaseSettings):
    """Environment variables consumed by both layers."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(
        "",
        validation_alias=AliasChoices(
            "OPENAI_API_KEY", "API_KEY", "AZURE_OPENAI_API_KEY"
        ),
    )
    endpoint_url: str = Field(
        "",
        validation_alias=AliasChoices(
            "OPENAI_BASE_URL", "ENDPOINT_URL", "AZURE_OPENAI_ENDPOINT"
        ),
    )
    vector_db_dir: str = "./vector-db"
    results_dir: str = "./results"
    data_dir: str = "./dataset"
    bm25_hmac_key: str = Field("default-bm25-integrity-key", alias="BM25_HMAC_KEY")

def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value

def load_config(path: Optional[str | Path] = None) -> AppConfig:
    """Parse and validate ``app_config.yaml`` (cached)."""

    return _cached_load_config(str(Path(path)) if path else None)

@lru_cache(maxsize=4)
def _cached_load_config(path_str: Optional[str]) -> AppConfig:
    cfg_path = Path(path_str) if path_str else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    expanded = _expand_env(raw)
    try:
        return AppConfig.model_validate(expanded)
    except ValidationError as exc:
        logger.error("Configuration validation error: %s", exc)
        raise

def require_env(var_name: str) -> str:
    val = os.environ.get(var_name, "").strip()
    if not val:
        raise RuntimeError(
            f"Required environment variable '{var_name}' is not set."
        )
    return val

class ConfigManager:
    """RAG-compatible facade over :class:`AppConfig` + :class:`EnvironmentSettings`.

    Vendored RAG modules expect the same attribute surface as
    ``resume_rag.config.settings.ConfigManager``; this class provides it while
    sharing the same underlying YAML config used by the agent/MCP layer.
    """

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        env_settings: Optional[EnvironmentSettings] = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self._project_root = PROJECT_ROOT
        env_path = self._project_root / ".env"
        env_kw: Dict[str, Any] = {}
        if env_settings is None and env_path.is_file():
            env_kw["_env_file"] = str(env_path)
        self.env_settings = env_settings or EnvironmentSettings(**env_kw)
        self.app_settings = load_config(self.config_path)
        self._ensure_directories()

    @property
    def project_root(self) -> Path:
        return self._project_root

    def _resolve_path(self, raw: str) -> Path:
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (self._project_root / p).resolve()

    def _ensure_directories(self) -> None:
        for directory in (
            self.vector_db_dir / self.app_settings.storage.chroma_db_path,
            self.results_dir / self.app_settings.storage.evaluation_results_path,
            self.results_dir / self.app_settings.storage.logs_path,
            self.results_dir / "audit",
            self.data_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def api_key(self) -> str:
        return self.env_settings.api_key

    @property
    def endpoint_url(self) -> str:
        return self.env_settings.endpoint_url

    @property
    def vector_db_dir(self) -> Path:
        return self._resolve_path(self.env_settings.vector_db_dir)

    @property
    def results_dir(self) -> Path:
        return self._resolve_path(self.env_settings.results_dir)

    @property
    def data_dir(self) -> Path:
        return self._resolve_path(self.env_settings.data_dir)

    @property
    def chroma_persist_dir(self) -> str:
        return str(self.vector_db_dir / self.app_settings.storage.chroma_db_path)

    @property
    def chroma_collection_name(self) -> str:
        return self.app_settings.storage.chroma_collection_name

    @property
    def evaluation_results_dir(self) -> str:
        return str(self.results_dir / self.app_settings.storage.evaluation_results_path)

    @property
    def logs_dir(self) -> str:
        return str(self.results_dir / self.app_settings.storage.logs_path)

    def project_path(self, relative: str) -> Path:
        return AppConfig.project_path(relative)

    def get_llm_config(self) -> LLMConfig:
        return self.app_settings.llm

    def get_embedding_config(self) -> EmbeddingConfig:
        return self.app_settings.embeddings

    def get_text_splitter_config(self) -> TextSplitterConfig:
        return self.app_settings.text_splitter

    def get_access_control_config(self) -> AccessControlConfig:
        return self.app_settings.access_control

    def get_evaluation_config(self) -> EvaluationConfig:
        return self.app_settings.evaluation
