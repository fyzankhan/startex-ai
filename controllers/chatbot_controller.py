from flask import Blueprint, request, jsonify
from services.chatbot_service import chatbot_service

chatbot_bp = Blueprint("chatbot", __name__)

@chatbot_bp.route("/ask", methods=["POST"])
def ask():

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
        result = chatbot_service.answer_question(question=question, client_id=client_id, format_id=format_id)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500