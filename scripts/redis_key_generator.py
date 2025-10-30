import hashlib

# Generate a unique cache key for the prompt and context
def make_cache_key(prompt, context):
    raw = prompt + context
    return "rag_cache:" + hashlib.sha256(raw.encode()).hexdigest()
