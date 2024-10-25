from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get("message")
    # Here, you would add your NLP processing logic.
    response = f"This is a response to '{user_input}'."
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
