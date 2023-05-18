import os
import json
import pickle
from flask_cors import CORS
from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


app = Flask(__name__)
CORS(app)

def sentimentVectorizer():
    # Load the saved vectorizer
    with open('seniment_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

def sentimentModel():
    # Load the model from disk
    with open('svc_model.pkl', 'rb') as file:
        svm = pickle.load(file)
    return svm

@app.route("/sentiment_analyzer/", methods=["POST"])
def sentiment_analyzer() -> str:
    try:
        # Check if request body is valid JSON
        if not request.is_json:
            return {"error": "Request body must be valid JSON"}, 400

        # Get input text from request body
        input_text = request.json.get("string")

        vec = sentimentVectorizer()
        mod = sentimentModel()
        result = {}
        if (mod ==""):
            result = {
                "message": "Failure Occurs",
                "info": "please enter a valid model number from 0 to 4 inclusive."
            }
        else :
            # do process here
            sid = SentimentIntensityAnalyzer()
            deep_output = sid.polarity_scores(input_text)
            #deep_output = mod.polarity_scores(deep_output)['compound']
            result = {
                "message": "Done Successfully",
                "result": deep_output
            }
        # return output
        return result
    except Exception as e:
        return {"error": str(e)}, 500





def toxicVec():
    # Load the saved vectorizer
    with open('toxic_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

def toxicModel(text):
    # Load the saved vectorizer
    with open('rf.pkl', 'rb') as file:
        rf = pickle.load(file)
    return rf


@app.route("/toxic_comment/", methods=["POST"])
def toxic_comment_classifier() -> str:
    try:
        # Check if request body is valid JSON
        if not request.is_json:
            return {"error": "Request body must be valid JSON"}, 400

        # Get input text from request body
        input_text = request.json.get("string")

        vec = toxicVec()
        mod = toxicModel(input_text)
        result = {}
        if (mod == ""):
            result = {
                "message": "Failure Occurs",
                "info": "please enter a valid model number from 0 to 5"
            }
        else :
            # do process here
            deep_output = vec.transform([input_text])
            deep_output = mod.predict_proba(deep_output)[:,1]
            result = {
                "message": "Done Successfully",
                "result": str(deep_output[0])
            }
        # return output
        return result
    except Exception as e:
        return {"error": str(e)}, 500




if __name__ == '__main__':
    app.run(debug=True)
