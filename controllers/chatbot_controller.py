from flask import Blueprint, request, jsonify
from services.chatbot_service import process_question
from services.chatbot_vector_service import build_or_load_vectorstore, process_vector_question
from flask import render_template

chatbot_bp = Blueprint("chatbot", __name__)

@chatbot_bp.route("/ask", methods=["POST"])
def ask():
    """
    Ask a natural language question about branch_audit table.
    ---
    tags:
      - Chatbot
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            question:
              type: string
              example: "Tell me about the lowest scoring branch in Bahawalpur"
            client_id:
              type: integer
              example: 42
            format_id:
              type: integer
              example: 7
    responses:
      200:
        description: SQL query and chatbot answer
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json(silent=True) or {}
    question = data.get("question")
    client_id = data.get("client_id")
    format_id = data.get("format_id")
    if not question or not isinstance(question, str):
        return jsonify({"error": "Missing or invalid 'question' field"}), 400
    try:
        client_id = int(client_id)
        format_id = int(format_id)
    except (TypeError, ValueError):
        return jsonify({"error": "'client_id' and 'format_id' must be integers"}), 400
    try:
        result = process_question(question=question, client_id=client_id, format_id=format_id)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


@chatbot_bp.route("/embed", methods=["POST"])
def embed():
    """
    Build or update vector embeddings for the given client and format.
    ---
    tags:
      - Chatbot
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            client_id:
              type: integer
              example: 42
            format_id:
              type: integer
              example: 7
    responses:
      200:
        description: Embedding process result
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json(silent=True) or {}
    client_id = data.get("client_id")
    format_id = data.get("format_id")
    try:
        client_id = int(client_id)
        format_id = int(format_id)
    except (TypeError, ValueError):
        return jsonify({"error": "'client_id' and 'format_id' must be integers"}), 400
    try:
        store = build_or_load_vectorstore(client_id=client_id, format_id=format_id)
        total = getattr(getattr(store, "index", None), "ntotal", 0) if store is not None else 0
        return jsonify(
            {
                "message": f"Vectorstore built/updated for client {client_id}, format {format_id}",
                "total_embeddings": int(total),
            }
        ), 200
    except Exception as e:
        return jsonify({"error": "Failed to build embeddings", "detail": str(e)}), 500


@chatbot_bp.route("/ask-vector", methods=["POST"])
def ask_vector():
    """
    Ask a natural language question using the embedded FAISS vectorstore.
    ---
    tags:
      - Chatbot
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            question:
              type: string
              example: "What do customers say about Lahore Main Branch?"
            client_id:
              type: integer
              example: 42
            format_id:
              type: integer
              example: 7
    responses:
      200:
        description: Chatbot answer from vectorstore
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json(silent=True) or {}
    question = data.get("question")
    client_id = data.get("client_id")
    format_id = data.get("format_id")
    if not question or not isinstance(question, str):
        return jsonify({"error": "Missing or invalid 'question' field"}), 400
    try:
        client_id = int(client_id)
        format_id = int(format_id)
    except (TypeError, ValueError):
        return jsonify({"error": "'client_id' and 'format_id' must be integers"}), 400
    try:
        result = process_vector_question(question=question, client_id=client_id, format_id=format_id)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Failed to answer from vectorstore", "detail": str(e)}), 500


@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    """
    Unified chat endpoint that decides whether to use SQL or embedding-based QA.
    ---
    tags:
      - Chatbot
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            question:
              type: string
              example: "What are customers saying about Lahore branch?"
            client_id:
              type: integer
              example: 42
            format_id:
              type: integer
              example: 7
    responses:
      200:
        description: Unified chatbot response
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True) or {}
    question = data.get("question")
    client_id = data.get("client_id")
    format_id = data.get("format_id")

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400
    try:
        client_id = int(client_id)
        format_id = int(format_id)
    except (TypeError, ValueError):
        return jsonify({"error": "'client_id' and 'format_id' must be integers"}), 400

    q_lower = question.lower()
    vector_keywords = ["feedback", "comment", "say", "mention", "experience", "satisfied", "review"]
    sql_keywords = ["average", "count", "top", "lowest", "highest", "branch", "nps", "score"]

    try:
        # Simple heuristic to choose the method
        if any(k in q_lower for k in vector_keywords) and not any(k in q_lower for k in sql_keywords):
            result = process_vector_question(question, client_id, format_id)
            result["source"] = "vector"
        else:
            result = process_question(question, client_id, format_id)
            result["source"] = "sql"

        # Chat UI response structure
        chat_response = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": result.get("answer", "")},
            ],
            "source": result.get("source"),
            "sql": result.get("sql"),
            "sources": result.get("sources"),
            "latency_ms": result.get("latency_ms"),
        }
        return jsonify(chat_response), 200

    except Exception as e:
        return jsonify({"error": "Chat processing failed", "detail": str(e)}), 500
      
  
  
@chatbot_bp.route("/chat-ui")
def chat_ui():
      """Render the Chatbot interface."""
      return render_template("chat_ui.html")
