import requests

url = "https://ilbeygulmez-mlsum-rag-llm.hf.space"

data = {
    "prompt": "Merhaba, nasılsın?"
}

res = requests.post(url + "/ask", json=data)
print(res.status_code)
print(res.json())
