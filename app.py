from flask import Flask, jsonify
from flasgger import Swagger
from controllers.chatbot_controller import chatbot_bp
from controllers.semantic_controller import semantic_bp

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Service is healthy"}), 200

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "ok", "message": "Startex AI"}), 200

app.register_blueprint(chatbot_bp, url_prefix="/chatbot")
app.register_blueprint(semantic_bp, url_prefix="/semantic")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
