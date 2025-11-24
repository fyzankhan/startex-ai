from flask import Blueprint, jsonify
from services.sync_service import SyncService

sync_bp = Blueprint("sync_bp", __name__)

@sync_bp.post("/vectors")
def sync_vectors():
    result = SyncService.sync()
    return jsonify(result)
