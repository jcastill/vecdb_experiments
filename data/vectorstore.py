from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection
)
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "docs"
EMBED_DIM = 384
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"


def connect_milvus():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)


def create_collection():
    if Collection.exists(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True),
        FieldSchema(name="command", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=EMBED_DIM)
    ]
    schema = CollectionSchema(fields, description="Semantic search over docs")

    collection = Collection(COLLECTION_NAME, schema)
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT",
                      "metric_type": "COSINE",
                      "params": {"nlist": 128}}
    )
    return collection


def load_chunks(file_path, chunk_size=512):
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
    data = [
        [command_name] * len(texts),  # command
        texts,                        # chunk
        embeddings                    # vector
    ]
    collection.insert(data)
    collection.flush()


def main():
    connect_milvus()
    model = SentenceTransformer(MODEL_NAME)
    collection = create_collection()

    # TODO: Add a check to see if the collection already exists
    # TODO: Add logic to read from different directories
    data_dir = "data"
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            command = fname.replace(".txt", "")
            chunks = load_chunks(os.path.join(data_dir, fname))
            embeddings = model.encode(chunks).tolist()
            insert_data(collection, command, chunks, embeddings)

    print(f"✅ Data inserted into Milvus collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
