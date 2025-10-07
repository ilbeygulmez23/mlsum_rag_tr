from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple

class CrossEncoderReranker:
    """
    CrossEncoderReranker uses a pretrained cross-encoder model to score and rank
    a list of candidate texts based on their relevance to a given input query.

    This is typically used in RAG (Retrieval-Augmented Generation) pipelines to
    rerank retrieved passages before passing the top-k to a generator model.

    Attributes:
        model_name (str): Name of the HuggingFace model to use.
        top_k (int): Number of top-scoring candidates to return.
        device (str): Computation device, automatically set to 'cuda' if available.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5, device: str = None):
        """
        Initializes the CrossEncoderReranker with a specified model and top_k.

        Args:
            model_name (str): HuggingFace model identifier.
            top_k (int): Number of top results to return after reranking.
            device (str): Manually specified device ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.top_k = top_k

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Scores and reranks the candidate texts based on their relevance to the query.

        Args:
            query (str): The input query or question.
            candidates (List[str]): List of retrieved documents/passages to rerank.

        Returns:
            List[Tuple[str, float]]: A list of top-k (text, score) pairs, sorted by relevance.
        """
        if not candidates:
            return []

        # Tokenize the query-candidate pairs as input for the cross-encoder
        inputs = self.tokenizer(
            [query] * len(candidates),  # Duplicate query for pairwise scoring
            candidates,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Disable gradient tracking for inference
        with torch.no_grad():
            # Compute relevance scores (logits)
            scores = self.model(**inputs).logits.squeeze(-1)

        # Pair each candidate with its score and sort descending
        results = list(zip(candidates, scores.cpu().tolist()))
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        return results[:self.top_k]

    def rerank_with_metadata(self, prompt: str, retrievals: list, top_k: int = 5):
        """
        Reranks raw Elasticsearch hits using a cross-encoder, preserving
        original hit metadata and adding reranker scores.

        Args:
            prompt (str): The query or question.
            retrievals (list): Raw Elasticsearch hits (dicts).
            top_k (int): Number of top results to return.

        Returns:
            List[dict]: Top-k hits in their original format with an added "rerank_score" field.
        """
        if not retrievals:
            return []

        # Map candidate texts to raw hits
        text_to_hit = {}
        texts = []

        for hit in retrievals:
            src = hit["_source"]
            text = src.get("text", "")
            # Only include hits that have text
            if text:
                texts.append(text)
                text_to_hit[text] = hit

        # Rerank the texts
        reranked = self.rerank(prompt, texts)[:top_k]

        # Map back to original hits and attach reranker scores
        reranked_hits = []
        for text, score in reranked:
            hit = text_to_hit[text].copy()  # Copy so we can safely add score
            hit["rerank_score"] = score
            reranked_hits.append(hit)

        return reranked_hits

