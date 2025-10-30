import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from src.indexing import index_data
from src.query import query_similar
from src.reranker import CrossEncoderReranker
from scripts.redis_key_generator import make_cache_key
from elasticsearch import Elasticsearch
import redis




# Page configuration
st.set_page_config(page_title="RAG Haber Asistanı", layout="wide")
st.title("📰 Tarafsız Haber Asistanı (RAG + LLM)")
st.caption("Elasticsearch + SentenceTransformer + CrossEncoderReranker + HuggingFace LLM")

# Global setup (first time only)
@st.cache_resource(show_spinner=True)
def initialize_rag_pipeline():
    st.write("🔄 Elasticsearch'e bağlanılıyor ve modeller yükleniyor...")

    # Elasticsearch connection (Docker's 9200 port)
    es = Elasticsearch("http://localhost:9200")

    # Embedding model
    model_name = "jinaai/jina-embeddings-v3"
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Index data preparation (if needed)
    index_data(model)  # if already exists, don't recreate

    # Reranker
    reranker = CrossEncoderReranker()

    # Redis connection
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    return model, es, reranker, redis_client


# setup only once (cache_resource)
model, es, reranker, redis_client = initialize_rag_pipeline()

# UI input field
prompt = st.text_area("Sorunuzu yazın:", height=100)

if st.button("Cevabı Al"):
    if not prompt.strip():
        st.warning("Lütfen bir soru girin.")
    else:
        with st.spinner("Veriler getiriliyor ve LLM çalıştırılıyor..."):
            # Retrieval + Reranker
            retrievals = query_similar(prompt, model, es=es, k=5)
            reranked_retrievals = reranker.rerank_with_metadata(prompt, retrievals)

            # Context creation
            rag_context = ""
            for idx, r in enumerate(reranked_retrievals, 1):
                src = r.get("_source", {})
                summary = src.get("summary", "")
                text = src.get("text", "")

                rag_context += f"{idx}. Özet: {summary}\n   Metin: {text[:500]}...\n"

            # Cache checking
            cache_key = make_cache_key(prompt, rag_context)
            cached = redis_client.get(cache_key)

            if cached:
                answer = cached.decode()
                st.success("⚡ Cevap (Redis Cache)")

            else:
                # LLM endpoint call
                llm_url = "https://ilbeygulmez-mlsum-rag-llm.hf.space/ask"

                system_prompt = f"""
                    Sen tarafsız bir haber asistanısın. 
                    Görevin, verilen bağlamı inceleyerek soruya yalnızca bu bilgilere dayanarak yanıt vermektir. 
                    Yanıtını Türkçe ve kısa, açık cümlelerle ver. Bağlamdaki bilgileri kullan, fakat bağlamın kendisine atıfta bulunma.

                    Örnek:
                    Soru: Türkiye’nin 2020 yılında ekonomik büyüme oranı neydi?
                    Bağlam:
                    1. Özet: Türkiye ekonomisi 2020 yılında pandemi etkisiyle daralma yaşasa da yılın son çeyreğinde toparlanma görüldü. Yıllık bazda %1,8 büyüme kaydedildi.
                    2. Metin: TÜİK verilerine göre, 2020 yılı büyüme oranı %1,8 olarak açıklandı.
                    Cevap: Türkiye ekonomisi 2020 yılında %1,8 oranında büyümüştür.

                    Şimdi senin sıran:

                    Soru: {prompt}

                    Bağlamlar:
                    {rag_context}

                    Cevap:
                    """

                try:
                    print("Prompt sent to LLM: ", system_prompt)
                    res = requests.post(llm_url, json={"prompt": system_prompt}, timeout=30)
                    res.raise_for_status()
                    answer = res.json().get("answer")

                    if answer:
                        # Cache the valid answer (30 minutes TTL)
                        redis_client.set(cache_key, answer, ex=600)
                    else:
                        answer = "❌ Cevap alınamadı."

                except Exception as e:
                    answer = f"❌ Hata oluştu: {e}"

            # Result display
            st.subheader("📄 Cevap:")
            st.write(answer)

            with st.expander("📚 Kullanılan Bağlamlar"):
                for idx, r in enumerate(reranked_retrievals, 1):
                    src = r.get("_source", {})
                    summary = src.get("summary", "")
                    text = src.get("text", "")
                    title = src.get("title", "")
                    date = src.get("date", "")

                    st.markdown(f"**{idx}. ({date}) {title}**")
                    st.markdown(f"**Özet:** {summary}")
                    st.markdown(f"**Metin:** {text[:500]}\n---")
