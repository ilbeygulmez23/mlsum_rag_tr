from elasticsearch import Elasticsearch

# Connect to local Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Check connection
if es.ping():
    print("Connected to Elasticsearch!")
else:
    print("Cannot reach Elasticsearch.")
