import sys
from sentence_transformers import SentenceTransformer
from src.indexing import index_data
from src.query import query_similar, print_retrievals
from src.reranker import CrossEncoderReranker


# Define evaluation prompts
PROMPTS = [
    "Asgari ücret zammı ile ilgili Erdoğan’ın yorumu nedir?",
    "Öğretmen atamaları konusunda sendikaların görüşü neydi?",
    "Yeni müfredat hakkında Milli Eğitim Bakanlığı ne söyledi?",
    "Elektrikli araçlara yönelik devlet teşviklerinden kimler yararlanabiliyor?",
    "Kira artışlarına karşı hükümetin aldığı önlemler nelerdir?",
    "Yerli aşı geliştirme süreci hakkında Sağlık Bakanı ne dedi?",
    "İstanbul’daki metro projeleriyle ilgili hangi açıklamalar yapıldı?",
    "Emeklilikte yaşa takılanlar (EYT) sorunu nasıl ele alındı?",
    "Yeni vergi düzenlemesi şirketleri nasıl etkileyecek?",
    "Üniversite sınav sistemiyle ilgili yapılan son değişiklikler nelerdir?"
]

def main():
    if len(sys.argv) > 2:
        print("Usage: python -m src.driver <embedding-model-name>")
        print("Example: python -m src.driver jinaai/jina-embeddings-v3")
        sys.exit(1)

    model_name = sys.argv[1] if len(sys.argv) == 2 else "jinaai/jina-embeddings-v3"
    print(f"\n>>> Loading embedding model: {model_name}")

    model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"\n>>> Embedding model successfully loaded: {model_name}")

    # Step 1: Index data
    print("\n>>> Indexing data into Elasticsearch...")
    es = index_data(model)

    # Step 2: Query similar results
    print("\n>>> Querying reranked retrievals with the prompts")
    reranker = CrossEncoderReranker()
    for prompt in PROMPTS:
        retrievals = query_similar(prompt, model, es=es)
        reranked_retrievals = reranker.rerank_with_metadata(prompt, retrievals)
        print_retrievals(prompt, reranked_retrievals)


    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
