from flask import Flask
from flasgger import Swagger
from controllers.chatbot_controller import chatbot_bp
from controllers.semantic_controller import semantic_bp

app = Flask(__name__)
swagger = Swagger(app)


app.register_blueprint(chatbot_bp, url_prefix="/chatbot")
app.register_blueprint(semantic_bp, url_prefix="/semantic")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
