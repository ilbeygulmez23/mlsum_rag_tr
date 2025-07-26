from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Define index name
index_name = "mlsum_tr_semantic"

# Query to get first 10 documents
response = es.search(
    index=index_name,
    body={
        "query": {
            "match_all": {}
        },
        "size": 10
    }
)

print(response)
print(es.count(index=index_name))  # should return {'count': some_number, ...}

# Print the results
for hit in response["hits"]["hits"]:
    print(f"ID: {hit['_id']}")
    print(f"Title: {hit['_source']['title']}")
    print(f"Summary: {hit['_source']['summary']}")
    print(f"Date: {hit['_source']['date']}")
    print(f"Text: {hit['_source']['text'][:200]}...")  # Truncate long text
    print("=" * 80)
