import json
from flask import Flask, request, jsonify
from model import LanguageModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

model = LanguageModel()

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "没有提供问题"}), 400

    response = model.generate_answer(question)

    return jsonify({"answer": response})

@app.route('/')
def home():
    return "LGCAI-API服务已启动"

if __name__ == "__main__":
    print("服务已启动，正在监听端口 5000...")
    app.run(debug=True, host="0.0.0.0", port=5000)
