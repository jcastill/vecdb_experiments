import sys
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# Import configuration
from config import (
    get_milvus_config,
    get_search_config,
    get_embedding_config
)

# Import custom exceptions
from cas.exceptions import (
    ConnectionError,
    ModelLoadError,
    SearchError,
    QueryProcessingError
)


def connect_to_milvus():
    """Connect to Milvus using configuration."""
    milvus_config = get_milvus_config()

    try:
        # Only pass non-None values to avoid type errors
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
        return True
    except Exception as e:
        raise ConnectionError(
            str(e),
            host=milvus_config.host,
            port=milvus_config.port
        )


def load_embedding_model():
    """Load embedding model based on configuration."""
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


def search_documents(query_vector, collection_name="docs"):
    """Search documents using configured parameters."""
    search_config = get_search_config()
    milvus_config = get_milvus_config()

    try:
        # Use configured collection name if available
        if milvus_config.collection_name != "vector_collection":
            col_name = milvus_config.collection_name
        else:
            col_name = collection_name

        col = Collection(col_name)

        # Build search parameters from configuration
        search_params = search_config.search_params.get(
            "params", {"nprobe": search_config.nprobe}
        )

        search_parameters = {
            "data": [query_vector],
            "anns_field": "embedding",
            "param": search_params,
            "limit": search_config.top_k,
            "output_fields": ["command", "chunk"],
            "consistency_level": "Strong"
        }

        results = col.search(**search_parameters)
        return results

    except Exception as e:
        raise SearchError(str(e), collection_name=col_name)


def display_results(results):
    """Display search results in a formatted way."""
    search_config = get_search_config()

    if not results or len(results) == 0:
        print("No results found.")
        return

    print(f"\n=== Search Results (Top {len(results[0])}) ===")

    for i, result in enumerate(results[0], 1):
        score = result.distance

        # Filter by similarity threshold if configured
        threshold = search_config.similarity_threshold

        # Note: For L2 distance, lower is better
        if search_config.metric_type == "L2":
            if score > (2.0 - threshold):
                continue
        elif search_config.metric_type in ["IP", "COSINE"]:
            if score < threshold:
                continue

        command = result.entity.get('command', 'N/A')
        chunk = result.entity.get('chunk', 'No content available')

        print(f"\n{i}. [{command} | Score: {score:.4f}]")
        print(f"   {chunk}")
        print("-" * 80)


def main():
    """Main CLI function."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python -m cas.cli <query>")
        print("Example: python -m cas.cli 'how to list files'")
        return 1

    query = " ".join(sys.argv[1:])
    print(f"Processing query: '{query}'")

    try:
        # Connect to Milvus
        connect_to_milvus()

        # Load embedding model
        model = load_embedding_model()

        # Encode query
        try:
            query_vector = model.encode(query).tolist()
            print(f"Query vector generated (dimension: {len(query_vector)})")
        except Exception as e:
            raise QueryProcessingError(str(e), query=query)

        # Search documents
        results = search_documents(query_vector)

        # Display results
        display_results(results)

        print("Search completed successfully")
        return 0

    except ConnectionError as e:
        print(f"Connection Error: {e}")
        return 1
    except ModelLoadError as e:
        print(f"Model Load Error: {e}")
        return 1
    except QueryProcessingError as e:
        print(f"Query Processing Error: {e}")
        return 1
    except SearchError as e:
        print(f"Search Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# vim: set et ts=4 sw=4 :
