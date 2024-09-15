from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify(message="Welcome to the Student Exam Performance Predictor!")

# Ensure Vercel recognizes the app
from vercel import make_response
app = make_response(app)

if __name__ == "__main__":
    app.run(debug=True)
