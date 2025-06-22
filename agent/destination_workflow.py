# LangGraph Workflow for Travel Recommendations
# This workflow processes a user's travel query, searches for relevant content,
# scrapes and summarizes travel information, and generates a final recommendation using a language model.
#------UPDATED 1--------------
from langgraph.graph import StateGraph, END
import requests
import os
from typing import TypedDict, List
from bs4 import BeautifulSoup

# Set env keys (⚠️ move to .env in prod)
os.environ["TAVILY_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"

# -----------------------
# Define the state schema
# -----------------------
class TravelState(TypedDict):
    query: str  # Human query
    structured_query: str
    search_result: str
    final_output: str

# ---------------------
# Groq LLM Integration
# ---------------------
def groq_generate(prompt: str) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1024
        }
    )
    return resp.json()["choices"][0]["message"]["content"]

# ---------------------------------
# Preprocessing Node: Query Cleaner
# ---------------------------------
def preprocess_query(state: TravelState) -> TravelState:
    prompt = f"""You are a travel planner assistant. Rewrite the following unstructured query into a clear, well-formed question that can be used for searching travel recommendations. Focus on extracting the user's intent and interests.

Unstructured Query: {state['query']}

- Only give the question as the result not extras.
- Include "in India" phrase to the Question.

Examples:
- "I want to go trekking with my friends" → "What are the top places for trekking with friends in India?"
- "We are planning a romantic honeymoon trip" → "What are the top places for a romantic honeymoon in India?"
- "Spiritual vibes and temples" → "What are the top places for spiritual tourism and temples in India?"

Format of the output:" What are the top places for ..(user's interest).. in India?"

"""
    
    state["structured_query"] = groq_generate(prompt).strip().strip('"“”\' ')
    #print(f"Structured Query: {state['structured_query']}")  # Debug log
    return state

# ---------------------------------
# Tavily Search Integration + URLs
# ---------------------------------
def tavily_search(query: str) -> List[str]:
    #print(f"Searching Tavily for query: {query}")  # Debug log
    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
        json={"query": query, "num_results": 3}
    )
    #print("STATUS:", response.status_code)
    #print("JSON:", response.json())
    results = response.json().get("results", [])
    '''for item in results:
        print(f"Found URL: {item.get('url')}")'''
    return [item.get("url") for item in results if item.get("url")]

# -------------------------------------
# Scraping and Summarization Function
# -------------------------------------
def scrape_and_summarize(url: str) -> str:
    try:
        response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        content = "\n".join(
            [p.get_text(strip=True) for p in soup.find_all("p")] +
            [t.get_text(strip=True) for t in soup.find_all("table")]
        )
        if not content.strip():
            return ""
        prompt = f"""Summarize the following travel content into a single sentence listing top 5 Indian travel destinations if present. Ignore if irrelevant or international.

Content:
{content[:3000]}"""
        return groq_generate(prompt).strip()
    except Exception as e:
        return ""

# ------------------
# Node: Search Query
# ------------------
def search_node(state: TravelState) -> TravelState:
    urls = tavily_search(state["structured_query"])
    summaries = []

    for url in urls:
        summary = scrape_and_summarize(url)
        if summary:
            summaries.append(summary)
    if not summaries:
        state["search_result"] = "No relevant content found."
    else:
        state["search_result"] = "\n".join(summaries)
    
    return state

# ----------------
# Node: LLM Output
# ----------------
def llm_node(state: TravelState) -> TravelState:
    context = state["search_result"]
    #print(f"Context for LLM: {context[:500]}...")  # Debug log
    user_query = state["query"]

    prompt = f"""
You are a professional travel consultant with deep expertise in Indian tourism. Your task is to recommend destinations within India based on the user's query and the summarized travel insights below.

-------------------------------
Summarized Information:
{context}

User Query:
"{user_query}"
-------------------------------

OBJECTIVE:
Carefully analyze the user's interest and recommend the top 3 destinations in India that strictly align with their theme. These top 3 should be described in detail. If you detect more than 3 relevant places, list the rest as names only under a separate heading titled "Other Suggested Places".

FILTERING LOGIC:
Only recommend destinations based on the user’s interest. Do not mix unrelated themes. Use the logic below:

- If the query includes religious, pilgrimage, or spiritual → Only suggest spiritual/religious destinations.
- If it includes romantic or honeymoon → Only suggest romantic getaways.
- If it includes heritage, historical, or cultural → Only suggest UNESCO heritage sites, monuments, and cultural cities.
- If it includes adventure, trekking, safari, or rafting → Only suggest adventure spots.
- If no specific theme is detected → Recommend a well-rounded mix of scenic and cultural destinations.

OUTPUT FORMAT (Keep within 400–500 words total):

Top 3 Recommended Destinations:

For each of the top 3, provide:
1. Destination: <Name>  
2. Why Visit: 2–3 sentences explaining what makes it special  
3. Practical Info:
   - Accessibility
   - Best season
   - Local transport
   - Accommodation range
4. Bonus Tips:
   - Must-visit attraction
   - Local dish to try
   - Cultural tip

Other Suggested Places(should be different from top 3):
- <Name 1>
- <Name 2>
- <Name 3>
(... if applicable)

Use clear headings, bullet points, and a friendly, expert tone. Stay focused and relevant to the user’s travel intent.
"""

    state["final_output"] = groq_generate(prompt.strip())
    return state

# ---------------------
# Build the LangGraph
# ---------------------
def build_graph():
    workflow = StateGraph(state_schema=TravelState)
    workflow.add_node("preprocess", preprocess_query)
    workflow.add_node("search", search_node)
    workflow.add_node("llm", llm_node)

    workflow.set_entry_point("preprocess")
    workflow.add_edge("preprocess", "search")
    workflow.add_edge("search", "llm")
    workflow.add_edge("llm", END)

    return workflow.compile()

# Compile graph
destination_graph = build_graph()

#state = destination_graph.invoke({"query": "i want to go trekking with my friends"})






