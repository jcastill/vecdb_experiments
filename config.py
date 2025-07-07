from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class MilvusConfig(BaseModel):
    """Configuration for Milvus vector database connection."""

    host: str = Field(default="localhost", description="Milvus server host")
    port: int = Field(default=19530, description="Milvus server port")
    username: Optional[str] = Field(
        default=None,
        description="Milvus username"
    )
    password: Optional[str] = Field(
        default=None,
        description="Milvus password"
    )
    database: str = Field(
        default="default",
        description="Milvus database name"
    )
    alias: str = Field(default="default", description="Connection alias")
    timeout: float = Field(
        default=30.0,
        description="Connection timeout in seconds"
    )
    collection_name: str = Field(
        default="vector_collection",
        description="Default collection name"
    )

    @field_validator('port')
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v


class SearchConfig(BaseModel):
    """Configuration for vector search parameters."""

    top_k: int = Field(
        default=10,
        description="Number of top results to return"
    )
    nprobe: int = Field(
        default=10,
        description="Number of clusters to search"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity threshold"
    )
    metric_type: str = Field(
        default="L2",
        description="Distance metric (L2, IP, COSINE)"
    )
    search_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        },
        description="Additional search parameters"
    )

    @field_validator('top_k')
    def validate_top_k(cls, v):
        if v <= 0:
            raise ValueError("top_k must be positive")
        return v

    @field_validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                "similarity_threshold must be between 0.0 and 1.0"
            )
        return v

    @field_validator('metric_type')
    def validate_metric_type(cls, v):
        allowed_metrics = ["L2", "IP", "COSINE", "HAMMING", "JACCARD"]
        if v not in allowed_metrics:
            raise ValueError(f"metric_type must be one of {allowed_metrics}")
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models and parameters."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    device: str = Field(
        default="cpu",
        description="Device to use for embedding (cpu, cuda, mps)"
    )


class AppConfig(BaseSettings):
    """Main application configuration that loads from environment variables."""

    # Component configurations
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    class Config:
        # Load from environment variables with prefix
        env_prefix = "VECDB_"
        env_nested_delimiter = "__"
        case_sensitive = False

        # Example environment variable names:
        # VECDB_MILVUS__HOST=localhost
        # VECDB_MILVUS__PORT=19530
        # VECDB_SEARCH__TOP_K=20
        # VECDB_EMBEDDING__MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> AppConfig:
    """Reload configuration from environment variables."""
    global config
    config = AppConfig()
    return config


# Convenience functions for common config access
def get_milvus_config() -> MilvusConfig:
    """Get Milvus configuration."""
    return config.milvus


def get_search_config() -> SearchConfig:
    """Get search configuration."""
    return config.search


def get_embedding_config() -> EmbeddingConfig:
    """Get embedding configuration."""
    return config.embedding


# Example usage and configuration validation
if __name__ == "__main__":
    # Print current configuration
    print("Current Configuration:")
    print(f"Milvus Host: {config.milvus.host}:{config.milvus.port}")
    print(f"Collection: {config.milvus.collection_name}")
    print(f"Search Top-K: {config.search.top_k}")
    print(f"Embedding Model: {config.embedding.model_name}")

    # Validate configuration
    try:
        config.model_validate(config.model_dump())
        print("✓ Configuration is valid")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
