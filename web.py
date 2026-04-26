import os
import pickle
import re

import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

load_dotenv()

VECTORIZER_FILE = os.getenv('VECTORIZER_FILE', 'models/vectorizer/tf_idf_vectorizer.pkl')
BEST_MODEL_FILE = os.getenv('BEST_MODEL_FILE', 'models/best_model.pkl')

# Load the trained model and vectorizer
def load_model():
    try:
        # Load the best performing model (logistic regression with TF-IDF)
        with open(BEST_MODEL_FILE, 'rb') as f:
            model = pickle.load(f)

        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def clean_text(text: str):
    """Text preprocessing function from the notebook"""
    if not isinstance(text, str):
        text = str(text)
    
    text = (text.lower()
                .replace('\n', '')
                .replace('\t', '')
                .replace('.', '')
                .replace('?', '')
                .replace('!', '')
                .replace(',', '')
                .replace('-', '')
            )

    text = re.sub(r'\d', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)

    return text

def predict_complaint(text, model, vectorizer):
    """Make prediction on complaint text"""
    try:
        # Clean the input text
        cleaned_text = clean_text(text)

        text_vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(text_vector)[0]

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vector)[0]
            classes = model.classes_
            confidence = max(probabilities)
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(text_vector)[0]
            # Apply softmax
            exp_scores = np.exp(decision_scores)
            confidence = float(max(exp_scores / np.sum(exp_scores)))
        else:
            confidence = None
        
        return prediction, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

model, vectorizer = load_model()

@app.route('/')
def home():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get complaint text from form
        complaint_text = request.form.get('complaint_text')
        
        if not complaint_text or complaint_text.strip() == '':
            return render_template('index.html', 
                                 error='Please enter a complaint description',
                                 prediction=None)
        
        # Make prediction
        prediction, confidence = predict_complaint(complaint_text, model, vectorizer)
        
        if prediction is None:
            return render_template('index.html', 
                                 error='Error making prediction. Please try again.',
                                 prediction=None)
        
        # Format the prediction
        formatted_prediction = prediction.replace('_', ' ').title()
        
        return render_template('index.html',
                             prediction=formatted_prediction,
                             confidence=confidence,
                             original_text=complaint_text,
                             error=None)
    
    except Exception as e:
        return render_template('index.html', 
                             error=f'An error occurred: {str(e)}',
                             prediction=None)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON predictions"""
    try:
        data = request.get_json()
        
        if not data or 'complaint_text' not in data:
            return jsonify({'error': 'complaint_text is required'}), 400
        
        complaint_text = data['complaint_text']
        
        if not complaint_text or complaint_text.strip() == '':
            return jsonify({'error': 'complaint_text cannot be empty'}), 400
        
        # Make prediction
        prediction, confidence = predict_complaint(complaint_text, model, vectorizer)
        
        if prediction is None:
            return jsonify({'error': 'Error making prediction'}), 500
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence) if confidence is not None else None,
            'original_text': complaint_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if model is None or vectorizer is None:
        print("Error: Could not load model or vectorizer. Please check if model files exist.")
    else:
        print("Model loaded successfully. Starting web server...")
        app.run(debug=True, host='0.0.0.0', port=8080)