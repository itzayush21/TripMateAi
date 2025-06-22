import os
import requests
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from groq import Groq

# Set environment keys (replace with your actual keys or load via dotenv)
os.environ["TAVILY_API_KEY"] = "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"
os.environ["GROQ_API_KEY"] = "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"


# ----- Shared State Definition -----
class ChatState(TypedDict):
    messages: List[dict]           # List of messages as dicts with 'role' and 'content'
    context: str                   # Web context from Tavily
    final_response: str            # Final response to user
    structured_query: str          # Cleaned query formed using LLM

# ----- Utility: Groq LLM Call -----
def groq_generate(prompt: str) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert at rephrasing queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# ----- Node 0: Preprocess Query -----
def preprocess_query(state: ChatState) -> ChatState:
    user_input = [m["content"] for m in state["messages"] if m["role"] == "user"][-1]
    
    
    prompt = f"""You are a travel planner assistant. Rewrite the following unstructured query into a clear, well-formed question that can be used for searching travel recommendations. Focus on extracting the user's intent and interests.

Unstructured Query: {user_input}

- Only give the question as the result not extras.
- Include \"in India\" phrase to the Question.

Examples:
- \"I want to go trekking with my friends\" → \"What are the top places for trekking with friends in India?\"
- \"We are planning a romantic honeymoon trip\" → \"What are the top places for a romantic honeymoon in India?\"
- \"Spiritual vibes and temples\" → \"What are the top places for spiritual tourism and temples in India?\"

Format of the output: \"What are the top places for ..(user's interest).. in India?\"
"""
    state["structured_query"] = groq_generate(prompt).strip('"“”')
    return state

# ----- Node 1: Fetch Fresh Context via Tavily -----
def fetch_context(state: ChatState) -> ChatState:
    query = state["structured_query"]
    headers = {"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"}
    payload = {"query": query, "search_depth": "basic", "include_answer": True}

    try:
        resp = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
        if resp.ok:
            state["context"] = resp.json().get("answer", "")
        else:
            state["context"] = "No live data found."
    except Exception:
        state["context"] = "Error during Tavily fetch."
    return state

# ----- Node 2: Generate Final Response using Groq LLM -----
def generate_response(state: ChatState) -> ChatState:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    user_input = [m for m in state["messages"] if m["role"] == "user"][-1]["content"]

    system_prompt = f"""
You are a travel expert for Indian destinations. The user is asking about: \"{user_input}\"
Structured Query: {state['structured_query']}
Context from web: {state['context']}

Strictly suggest 3 destinations relevant to this query.
Use this format:
- **Destination**: Name
- **Why Visit**: 2–3 lines
- **Practical Info**: Access, best time, transport, stays
- **Bonus Tips**: Attraction, dish, etiquette

Keep it around 300–400 words. Respond in a helpful, local-savvy tone.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            *state["messages"]
        ],
        temperature=0.7
    )

    msg = response.choices[0].message.content
    state["messages"].append({"role": "assistant", "content": msg})
    state["final_response"] = msg
    return state

# ----- LangGraph Flow Setup -----
def build_travel_chat_agent():
    graph = StateGraph(ChatState)
    graph.add_node("preprocess_query", preprocess_query)
    graph.add_node("fetch_context", fetch_context)
    graph.add_node("generate_response", generate_response)

    graph.set_entry_point("preprocess_query")
    graph.add_edge("preprocess_query", "fetch_context")
    graph.add_edge("fetch_context", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()

