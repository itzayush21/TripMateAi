from flask import Flask, request, jsonify,render_template

from agent.destination_workflow import destination_graph
from agent.destination_chat_workflow import build_travel_chat_agent

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")  # This will load templates/index.html

@app.route("/destination", methods=["POST"])
def suggest_destination():
    user_input = request.json.get("query", "")
    state = destination_graph.invoke({"query": user_input})
    return jsonify({
        "result": state.get("final_output", "Something went wrong."),
        "search_results": state.get("search_result","Something went wrong.")
    })

destination_graph1 = build_travel_chat_agent()

# In-memory session storage
if not hasattr(app, "memory"):
    app.memory = {}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id", "default")
    user_message = data["message"]

    # Initialize session if not present
    session_memory = app.memory.setdefault(session_id, {
        "messages": [],
        "travel_theme": "",
        "context": "",
        "final_response": "",
        "structured_query": ""
    })

    # Append new user message to chat history
    session_memory["messages"].append({"role": "user", "content": user_message})

    # Run the LangGraph agent
    result = destination_graph1.invoke(session_memory)
    
    
    

    # Update session memory
    app.memory[session_id] = result

    # Return latest assistant message
    latest_msg = next((m["content"] for m in reversed(result["messages"]) if m["role"] == "assistant"), "")
    return jsonify({"response": latest_msg})

if __name__ == "__main__":
    app.run(debug=True)
