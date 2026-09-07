from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SettingsError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


@dataclass(frozen=True)
class Settings:
    project_root: Path
    mineru_api_token: str
    siliconflow_api_key: str
    vlm_model: str
    embedding_model: str
    rerank_model: str
    mysql_dsn: str
    milvus_uri: str
    config: dict[str, Any]

    @property
    def mineru(self) -> dict[str, Any]:
        return self.config["mineru"]

    @property
    def vlm(self) -> dict[str, Any]:
        return self.config["vlm"]

    @property
    def embedding(self) -> dict[str, Any]:
        return self.config["embedding"]

    @property
    def rerank(self) -> dict[str, Any]:
        return self.config["rerank"]

    @property
    def pipeline(self) -> dict[str, Any]:
        return self.config["pipeline"]

    @property
    def quality(self) -> dict[str, Any]:
        return self.config["quality"]


def _required(values: dict[str, str | None], key: str) -> str:
    value = values.get(key)
    if not value or not str(value).strip():
        raise SettingsError(f"Missing required setting: {key}")
    return str(value).strip()


def _resolve_milvus_uri(root: Path, raw_uri: str) -> str:
    if "://" in raw_uri or raw_uri.startswith("/"):
        return raw_uri
    return str((root / raw_uri).resolve())


def load_settings(
    env_path: Path | None = None,
    config_path: Path | None = None,
) -> Settings:
    root = PROJECT_ROOT
    env_path = env_path or root / ".env"
    config_path = config_path or root / "config" / "default.yaml"
    values = dotenv_values(env_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # ``MILVUS_URI`` is reserved by pymilvus itself.  A local Lite path in
    # that process environment is parsed as a remote server URI during import
    # and raises ConnectionConfigException, so the application uses a scoped
    # variable name.
    raw_milvus_uri = _required(values, "RAG_MILVUS_URI")
    return Settings(
        project_root=root,
        mineru_api_token=_required(values, "MINERU_API_TOKEN"),
        siliconflow_api_key=_required(values, "SILICONFLOW_API_KEY"),
        vlm_model=_required(values, "VLM_MODEL"),
        embedding_model=_required(values, "EMBEDDING_MODEL"),
        rerank_model=_required(values, "RERANK_MODEL"),
        mysql_dsn=_required(values, "MYSQL_DSN"),
        milvus_uri=_resolve_milvus_uri(root, raw_milvus_uri),
        config=config,
    )
