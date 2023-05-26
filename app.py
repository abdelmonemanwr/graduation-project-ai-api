import json
import torch
from flask_cors import CORS
from collections import Counter
from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)

def count_words(string):
    freq = Counter(string)
    return (freq[' '] + 1)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/summarize/", methods=["POST"])
def summarize() -> str:
    global summarizer
    text = request.json.get("string")
    x = count_words(text)
    Max_Length = int(x / 2)
    Min_Length = int(x / 4)
    summerized = summarizer(  text,
                              max_length=Max_Length,
                              min_length=Min_Length,
                              do_sample=False
                            )
    return jsonify({"message": summerized})

if __name__ == '__main__':
    app.run(debug=True)
