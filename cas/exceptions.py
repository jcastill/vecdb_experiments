class CASException(Exception):
    """ Base CAS Exception """
    pass


class ConnectionError(CASException):
    """Exception raised when connection to Milvus fails."""

    def __init__(self, message, host=None, port=None):
        self.host = host
        self.port = port
        if host and port:
            super().__init__(
                f"Failed to connect to Milvus at {host}:{port}: {message}"
            )
        else:
            super().__init__(f"Failed to connect to Milvus: {message}")


class ModelLoadError(CASException):
    """Exception raised when embedding model fails to load."""

    def __init__(self, message, model_name=None):
        self.model_name = model_name
        if model_name:
            super().__init__(
                f"Failed to load embedding model '{model_name}': {message}"
            )
        else:
            super().__init__(f"Failed to load embedding model: {message}")


class SearchError(CASException):
    """Exception raised when search operation fails."""

    def __init__(self, message, collection_name=None):
        self.collection_name = collection_name
        if collection_name:
            super().__init__(
                f"Search failed in collection '{collection_name}': {message}"
            )
        else:
            super().__init__(f"Search failed: {message}")


class QueryProcessingError(CASException):
    """Exception raised when query processing fails."""

    def __init__(self, message, query=None):
        self.query = query
        if query:
            super().__init__(f"Failed to process query '{query}': {message}")
        else:
            super().__init__(f"Failed to process query: {message}")


class ConfigurationError(CASException):
    """Exception raised when configuration is invalid or missing."""

    def __init__(self, message, config_key=None):
        self.config_key = config_key
        if config_key:
            super().__init__(
                f"Configuration error for '{config_key}': {message}"
            )
        else:
            super().__init__(f"Configuration error: {message}")

# vim: set et ts=4 sw=4 :
