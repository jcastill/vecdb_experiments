import sys
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

connections.connect("default", host="localhost", port="19530")
model = SentenceTransformer("all-MiniLM-L6-v2")


def main():
    query = " ".join(sys.argv[1:])
    if not query:
        print("Query was missing. Please specify a query.")
        return
    vector = model.encode(query).tolist()
    col = Collection("docs")
    results = col.search(
        [vector],
        anns_field="embedding",
        param={"nprobe": 10},
        limit=5,  # Lets get 5 matches for now, we can go to 3 later
        output_fields=["command", "chunk"],
    )
    for r in results[0]:
        print(f"   [{r.entity['command']} Score={r.distance:.4f}]")
        print(r.entity['chunk'])


if __name__ == "__main__":
    main()

