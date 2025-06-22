import os
import requests
os.environ["TAVILY_API_KEY"] = "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"


response = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
        json={"query": "What are the top places for solo religious pilgrimages in India?", "num_results": 3}
    )
print("STATUS:", response.status_code)
print("JSON:", response.json())
results = response.json().get("results", [])
for item in results:
    print(f"Found URL: {item.get('url')}")
    