from elasticsearch.helpers import streaming_bulk
from elasticsearch import Elasticsearch
from datasets import load_dataset
from tqdm import tqdm
from scripts.date_formatter import format_month_year

def index_data(model, index_name="mlsum_tr_semantic"):
    # Connect to Elasticsearch
    es = Elasticsearch("http://localhost:9200")

    # Check if the index already exists
    if es.indices.exists(index=index_name):
        print("Data is already indexed")
    else:
        # Create the index with appropriate mappings
        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "summary": {"type": "text"},
                        "title": {"type": "text"},
                        "date": {"type": "text"}, # Ocak 2024 # date_string custom field # 00/01/2010
                        "embedding": {
                            "type": "dense_vector",
                            "dims": model.get_sentence_embedding_dimension(), # index reload
                            "index": True,
                        "similarity": "cosine"
                    }
                    }
                }
            },
            request_timeout=60
        )

        # Load the Turkish portion of the MLSUM dataset
        dataset = load_dataset(
            "mlsum", "tu", split="train[:5%]", trust_remote_code=True)

        # Generate embeddings for each entry
        def batched_embed(batch):
            texts = [
                f"{title} {summary}"
                for title, summary in zip(batch["title"], batch["summary"])
            ]
            embeddings = model.encode(texts, batch_size=64).tolist()
            
            return {"embedding": embeddings}
        


        dataset = dataset.map(batched_embed, batched=True, batch_size=64, load_from_cache_file=False)

        def doc_generator(dataset, index_name):
            for idx, row in enumerate(dataset):
                yield {
                    "_index": index_name,
                    "_id": idx,
                    "_source": {
                        "text": row["text"],
                        "summary": row["summary"],
                        "title": row["title"],
                        "date": format_month_year(row["date"]),
                        "embedding": row["embedding"]
                    }
                }

        es.indices.put_settings(index=index_name, body={"index": {"refresh_interval": "-1"}})

        # Stream and monitor status
        success_count = 0
        for ok, response in tqdm(streaming_bulk(es, doc_generator(dataset, index_name), chunk_size=500)):
            if not ok:
                print("❌ Failed to index document:", response)
            else:
                success_count += 1

        es.indices.put_settings(index=index_name, body={"index": {"refresh_interval": "1s"}})
        print(f"✅ Indexed {success_count} documents successfully.")

    return es
