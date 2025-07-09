import sys
import argparse
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility

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
    QueryProcessingError,
    ConfigurationError
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


def test_connection():
    """Test connection to localMilvus and other components."""
    print("=== Connection Test ===")

    # Test Milvus connection
    try:
        print("1. Testing Milvus connection...")
        connect_to_milvus()
        print("   ✓ Milvus connection successful")

        # Test if we can list collections
        print("2. Testing database access...")
        collections = utility.list_collections()
        print(f"   ✓ Database access successful. Found {len(collections)} "
              f"collections: {collections}")

        # Test collection access if it exists
        milvus_config = get_milvus_config()
        collection_name = milvus_config.collection_name

        print(f"3. Testing collection '{collection_name}' access...")
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            # Try to get collection stats
            collection.load()
            num_entities = collection.num_entities
            print(f"   ✓ Collection '{collection_name}' exists and has "
                  f"{num_entities} entities")
        else:
            print(f"   ⚠ Collection '{collection_name}' does not exist "
                  f"(this is normal for new setups)")

    except ConnectionError as e:
        print(f"   ✗ Milvus connection failed: {e}")
        return False
    except Exception as e:
        # Convert generic Milvus exceptions to ConnectionError for consistency
        conn_error = ConnectionError(str(e))
        print(f"   ✗ Milvus test failed: {conn_error}")
        return False

    # Test embedding model loading
    try:
        print("4. Testing embedding model loading...")
        model = load_embedding_model()
        print("   ✓ Embedding model loaded successfully")

        # Test encoding a sample text
        print("5. Testing embedding generation...")
        test_text = "This is a test query"
        test_vector = model.encode(test_text).tolist()
        print(f"   ✓ Embedding generated successfully "
              f"(dimension: {len(test_vector)})")

    except ModelLoadError as e:
        print(f"   ✗ Embedding model test failed: {e}")
        return False
    except Exception as e:
        # Convert generic embedding exceptions to ModelLoadError 
        # for consistency
        model_error = ModelLoadError(str(e))
        print(f"   ✗ Embedding test failed: {model_error}")
        return False

    # Test configuration
    try:
        print("6. Testing configuration...")
        config = get_milvus_config()
        search_config = get_search_config()
        embedding_config = get_embedding_config()

        print(f"   ✓ Milvus config: {config.host}:{config.port}, "
              f"db: {config.database}")
        print(f"   ✓ Search config: top_k={search_config.top_k}, "
              f"metric={search_config.metric_type}")
        print(f"   ✓ Embedding config: model={embedding_config.model_name}, "
              f"device={embedding_config.device}")

    except ConfigurationError as e:
        print(f"   ✗ Configuration test failed: {e}")
        return False
    except Exception as e:
        # Convert generic exceptions to ConfigurationError for consistency
        config_error = ConfigurationError(str(e))
        print(f"   ✗ Configuration test failed: {config_error}")
        return False

    print("\n=== All tests passed! ===")
    print("Your vector database setup is working correctly.")
    return True


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


def search_documents(query_vector, collection_name=None):
    """Search documents using configured parameters."""
    search_config = get_search_config()
    milvus_config = get_milvus_config()

    try:
        # Use configured collection name by default
        col_name = collection_name or milvus_config.collection_name

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

        command = getattr(result.entity, 'command', 'N/A')
        chunk = getattr(result.entity, 'chunk', 'No content available')

        print(f"\n{i}. [{command} | Score: {score:.4f}]")
        print(f"   {chunk}")
        print("-" * 80)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Vector Database Experiments CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m cas.cli test                   # Test connection and setup
    python -m cas.cli "how to list files"    # Search for query
    python -m cas.cli --help                 # Show this help
        """
    )

    parser.add_argument(
        'command_or_query',
        nargs='*',
        help='Command to run (test) or search query'
    )

    args = parser.parse_args()

    if not args.command_or_query:
        parser.print_help()
        return 1

    # Join all arguments to form the command or query
    input_text = " ".join(args.command_or_query)

    # Handle test command
    if input_text.lower() == 'test':
        try:
            if test_connection():
                return 0
            else:
                return 1
        except Exception as e:
            print(f"Test failed with unexpected error: {e}")
            return 1

    # Handle search query
    print(f"Processing query: '{input_text}'")

    try:
        # Connect to Milvus
        connect_to_milvus()

        # Load embedding model
        model = load_embedding_model()

        # Encode query
        try:
            query_vector = model.encode(input_text).tolist()
            print(f"Query vector generated (dimension: {len(query_vector)})")
        except Exception as e:
            raise QueryProcessingError(str(e), query=input_text)

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
    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# vim: set et ts=4 sw=4 :
