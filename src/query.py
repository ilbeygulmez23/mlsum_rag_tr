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
    window_size = 50
    rank_constant = 60

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

    window_size_lex = 50
    window_size_vec = 100

    lexical_body = {
        "size": window_size_lex,
        "query": lexical
    }
    vec_body = {
        "size": window_size_vec,
        "knn": {
            "field": "embedding",
            "query_vector": embedding,
            "k": max(k, window_size),
            "num_candidates": max(100, window_size)
        }
    }

    lex_hits = es.search(index=index, body=lexical_body)["hits"]["hits"]
    vec_hits = es.search(index=index, body=vec_body)["hits"]["hits"]

    # Fuse the queries with RRF
    def rrf_merge(lex_hits, vec_hits, *, k_const=60, w_lex=1.0, w_vec=1.3, limit=25):
        """
        Client-side Reciprocal Rank Fusion (RRF) with per-retriever weights.
        - lex_hits, vec_hits: lists from ES (you control their sizes when querying)
        - k_const: rank_constant (larger = flatter influence)
        - w_lex, w_vec: weights to bias lexical vs vector
        - limit: final fused size
        """
        acc = {}

        def add_list(hits, weight):
            for r, h in enumerate(hits, start=1):
                _id = h["_id"]
                entry = acc.setdefault(_id, {"hit": h, "score": 0.0})
                entry["score"] += weight * (1.0 / (k_const + r))

        add_list(lex_hits, w_lex)
        add_list(vec_hits, w_vec)

        fused = sorted(acc.values(), key=lambda x: (-x["score"], x["hit"]["_id"]))

        # Expose RRF score as _score for inspection
        for e in fused:
            e["hit"]["_score"] = e["score"]
        return [e["hit"] for e in fused[:limit]]


    return rrf_merge(lex_hits, vec_hits)


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
            print(f"Date: {src.get('date')}")
            print(f"Summary: {src.get('summary')}")
            print(f"Text (truncated): {src.get('text', '')[:200]}...")
            print("=" * 80)
    except ValueError:
        print("The structure of the retrievals are not as expected. Skipped.")
        