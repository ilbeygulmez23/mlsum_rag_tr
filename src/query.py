from elasticsearch import Elasticsearch
from scripts.date_extractor import extract_turkish_date


def query_similar(prompt, model, k=10, index="mlsum_tr_semantic", host="localhost", port=9200, es=None):
    if not es:
        es = Elasticsearch(f"http://{host}:{port}")

    # Extract date from the prompt
    date = extract_turkish_date(prompt)
    lexical_query = prompt.replace(date, "").strip() if date else prompt

    # Embed the prompt
    embedding = embed_prompt(prompt, model)

    # Validate dimensions
    mapping = es.indices.get_mapping(index=index)
    dims = mapping[index]["mappings"]["properties"]["embedding"]["dims"]
    if len(embedding) != dims:
        raise ValueError(f"Dimension mismatch: got {len(embedding)}, expected {dims}")

    # Choose size of final result set
    size = 25

    # Build the hybrid RRF body
    if date:
        lexical = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": lexical_query,
                            "fields": ["summary^3", "title^2"],
                            "operator": "or"
                        }
                    },
                    {
                        "match": {
                            "date": {
                                "query": date,
                                "operator": "and"
                            }
                        }
                    }
                ]
            }
        }
    else:
        lexical = {
            "multi_match": {
                "query": lexical_query,
                "fields": ["summary^3", "title^2"],
                "operator": "or"
            }
        }

    body = {
        "size": size,
        "retriever": {
            "rrf": { # Burada syntax hatası alıyorum, gpt boyle bise yok diyor ama bence var.
                "retrievers": [
                    {
                        "standard": {
                            "query": lexical
                        }
                    },
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": embedding,
                            "k": k
                        }
                    }
                ]
            }
        }
    }

    return es.search(index=index, body=body)["hits"]["hits"]



def embed_prompt(prompt, model):
    return model.encode(prompt).tolist()


def print_retrievals(prompt, retrievals = None):
    if not retrievals:
        print("No results are retrieved.")
        exit(1)

    try:
        # Print top-k results
        for hit in retrievals:
            src = hit["_source"]
            print(prompt)
            print("=" * 80)
            print(f"ID={hit['_id']} Score={hit['_score']:.4f}")
            print(f"Title: {src.get('title')}")
            print(f"Topic: {src.get('topic')}")
            print(f"Summary: {src.get('summary')}")
            print(f"Text (truncated): {src.get('text', '')[:200]}...")
            print("=" * 80)
    except ValueError:
        print("The structure of the retrievals are not as expected. Skipped.")