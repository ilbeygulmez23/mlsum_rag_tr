from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from datasets import load_dataset
from tqdm import tqdm

# count 0 indexed, smth is wrong

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
                        "date": {"type": "date"},
                        "embedding": {
                        "type": "dense_vector",
                        "dims": model.get_sentence_embedding_dimension(),
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

        dataset = dataset.map(batched_embed, batched=True, batch_size=64)

        def doc_generator(dataset, index_name):
            for idx, row in enumerate(dataset):
                yield {
                    "_index": index_name,
                    "_id": idx,
                    "_source": {
                        "text": row["text"],
                        "summary": row["summary"],
                        "title": row["title"],
                        "date": row["date"],
                        "embedding": row["embedding"]
                    }
                }

        es.indices.put_settings(index=index_name, body={"index": {"refresh_interval": "-1"}})

        actions = list(doc_generator(dataset, index_name))
        
        # Bulk index the documents -- must be changed for much much bigger docs, 4/5 scalability
        success, failed = bulk(es, actions, stats_only=True)
        print(f"✅ Indexed {success} documents. ❌ Failed: {failed}")

        es.indices.put_settings(index=index_name, body={"index": {"refresh_interval": "1s"}})


# Reformat the dates
turkish_months = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
    5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
    9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
}

def format_month_year(date_str):
    try:
        parts = date_str.split("/")  # expected: "00/MM/YYYY"
        month = int(parts[1])
        year = parts[2]
        month_name = turkish_months[month]
        return f"{month_name} {year}"  # e.g., "Ocak 2024"
    except Exception as e:
        return ""
