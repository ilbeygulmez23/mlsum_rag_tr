import sys
import os
from sentence_transformers import SentenceTransformer
from src.indexing import index_data
from src.query import query_similar, print_retrievals


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

#"jinaai/jina-embeddings-v3"
def main():
    if len(sys.argv) != 2:
        print("Usage: python driver.py <embedding-model-name>")
        print("Example: python driver.py all-MiniLM-L6-v2")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"\n>>> Loading embedding model: {model_name}")

    model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"\n>>> Embedding model successfully loaded: {model_name}")

    # Step 1: Index data
    print("\n>>> Indexing data into Elasticsearch...")
    index_data(model)
    print("\n>>> Data is successfully indexed.")


    # Step 2: Query similar results and export top_k_results.csv
    print("\n>>> Querying with the prompts")
    if os.path.exists("top_k_results.csv"):
        os.remove("top_k_results.csv")

    for prompt in PROMPTS:
        retrievals = query_similar(prompt, model)
        print_retrievals(prompt, retrievals)

    print("\n✅ Pipeline complete. Results saved to top_k_results.csv")


if __name__ == "__main__":
    main()
