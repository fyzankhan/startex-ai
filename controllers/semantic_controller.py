import os
import logging
import traceback
from flask import Blueprint, request, jsonify
from services.semantic_service import SemanticService
from flask import render_template

semantic_bp = Blueprint("semantic", __name__)
logger = logging.getLogger("semantic")
logger.setLevel(logging.INFO)


@semantic_bp.route("/sentiment/insights", methods=["POST"])
def sentiment_insights():
    """
    Sentiment Insights API
    ---
    tags:
      - Semantic
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            formatID: { type: number, example: 1 }
            sectionID: { type: array, items: { type: number }, example: [1, 2, 3] }
            Tag: { type: array, items: { type: string }, example: ["Food Quality", "Courtesy"] }
            locationID: { type: array, items: { type: number }, example: [1, 2, 3] }
            dateFrom: { type: string, format: date, example: "2025-09-01" }
            dateTo: { type: string, format: date, example: "2025-09-26" }
    responses:
      200:
        description: Semantic Sentiment Insights
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    try:
        data = request.get_json() or {}
        logger.info(f"[SemanticInsights] filters={data}")

        result = SemanticService.get_insights(data)
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[SemanticInsights] {e}")
        if os.getenv("FLASK_ENV") == "development":
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
        return jsonify({"error": "Internal Server Error"}), 500


@semantic_bp.route("/sentiment-ui")
def chat_ui():
      """Render the semanti interface."""
      return render_template("sentiment_ui.html")