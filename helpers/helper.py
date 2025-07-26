from elasticsearch import Elasticsearch


def delete_index(index_name):
    # Helper to delete a specific index

    # Connect to local Elasticsearch instance
    es = Elasticsearch("http://localhost:9200")

    # Check if the index exists and delete it
    if es.indices.exists(index=index_name, request_timeout=30):
        es.indices.delete(index=index_name, request_timeout=60)
        print(f"Index '{index_name}' deleted.")
    else:
        print(f"Index '{index_name}' does not exist.")


def list_indices():
    # Connect to local Elasticsearch instance
    es = Elasticsearch("http://localhost:9200")

    # Fetch and print all index names using keyword argument
    indices = es.indices.get_alias(index="*")
    print("Indices:")
    for index in indices:
        print(f"- {index}")


if __name__ == "__main__":
    list_indices()
