from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
from sentence_transformers import SentenceTransformer
import os

# Import configuration
from config import (
    get_milvus_config,
    get_embedding_config,
    get_search_config
)

# Import custom exceptions
from cas.exceptions import (
    ConnectionError,
    ModelLoadError,
    ConfigurationError
)


def connect_milvus():
    """Connect to Milvus using configuration."""
    milvus_config = get_milvus_config()

    try:
        # Build connection parameters from configuration
        conn_params = {
            'alias': milvus_config.alias,
            'host': milvus_config.host,
            'port': milvus_config.port,
            'timeout': milvus_config.timeout
        }

        if milvus_config.username:
            conn_params['user'] = milvus_config.username
        if milvus_config.password:
            conn_params['password'] = milvus_config.password
        if milvus_config.database != "default":
            conn_params['db_name'] = milvus_config.database

        connections.connect(**conn_params)
        print(f"Connected to Milvus at "
              f"{milvus_config.host}:{milvus_config.port}")

    except Exception as e:
        raise ConnectionError(
            str(e),
            host=milvus_config.host,
            port=milvus_config.port
        )


def get_embedding_dimension():
    """Get the embedding dimension from the configured model."""
    embedding_config = get_embedding_config()

    # Map known models to their dimensions
    model_dimensions = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/all-distilroberta-v1": 768,
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-distilroberta-v1": 768,
    }

    return model_dimensions.get(embedding_config.model_name, 384)


def create_collection():
    """Create or get existing collection using configuration."""
    milvus_config = get_milvus_config()
    collection_name = milvus_config.collection_name
    embed_dim = get_embedding_dimension()

    if utility.has_collection(collection_name):
        return Collection(collection_name)

    fields = [
        FieldSchema(name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True),
        FieldSchema(name="command", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=embed_dim)
    ]
    schema = CollectionSchema(fields, description="Semantic search over docs")

    collection = Collection(collection_name, schema)

    # Use search config for index parameters
    search_config = get_search_config()
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": search_config.metric_type,
        "params": {"nlist": 128}
    }

    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Created collection '{collection_name}' with dimension {embed_dim}")
    return collection


def load_embedding_model():
    """Load embedding model using configuration."""
    embedding_config = get_embedding_config()

    try:
        model = SentenceTransformer(
            embedding_config.model_name,
            device=embedding_config.device
        )
        print(f"Loaded embedding model: {embedding_config.model_name}")
        return model
    except Exception as e:
        raise ModelLoadError(str(e), model_name=embedding_config.model_name)


def load_chunks(file_path, chunk_size=512):
    """Load and chunk text from file."""
    with open(file_path, "r") as f:
        text = f.read()

    lines = text.strip().split("\n")
    chunks, current = [], []

    for line in lines:
        current.append(line.strip())
        if len(" ".join(current)) > chunk_size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def insert_data(collection, command_name, texts, embeddings):
    """Insert data into collection."""
    data = [
        [command_name] * len(texts),  # command
        texts,                        # chunk
        embeddings                    # vector
    ]
    collection.insert(data)
    collection.flush()


def main():
    """Main function to index documents."""
    try:
        # Connect to Milvus using configuration
        connect_milvus()

        # Load embedding model using configuration
        model = load_embedding_model()

        # Create collection using configuration
        collection = create_collection()

        # Get collection name from config for final message
        milvus_config = get_milvus_config()
        collection_name = milvus_config.collection_name

        # TODO: Add logic to read from different directories
        data_dir = "data"

        if not os.path.exists(data_dir):
            print(f"Warning: Data directory '{data_dir}' does not exist")
            return

        processed_files = 0
        for fname in os.listdir(data_dir):
            if fname.endswith(".txt"):
                try:
                    command = fname.replace(".txt", "")
                    file_path = os.path.join(data_dir, fname)
                    chunks = load_chunks(file_path)
                    embeddings = model.encode(chunks).tolist()
                    insert_data(collection, command, chunks, embeddings)
                    processed_files += 1
                    print(f"✓ Processed {fname}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"✗ Error processing {fname}: {e}")

        print(f"✓ Processed {processed_files} files into Milvus "
              f"collection '{collection_name}'")

    except ConnectionError as e:
        print(f"Connection Error: {e}")
        return 1
    except ModelLoadError as e:
        print(f"Model Load Error: {e}")
        return 1
    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        exit(exit_code)

# vim: set et ts=4 sw=4 :
