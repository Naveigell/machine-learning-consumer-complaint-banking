# Banking Complaint Classifier Web Application

A Flask-based web application that classifies banking complaints using machine learning models trained on consumer complaint data.

## Features

- **Web Interface**: User-friendly form for entering complaint descriptions
- **Real-time Classification**: Instant prediction using the best performing model (Logistic Regression with TF-IDF)
- **API Endpoint**: RESTful API for programmatic access
- **Confidence Scores**: Shows prediction confidence when available
- **Responsive Design**: Works on desktop and mobile devices

## Model Performance

The application uses the Logistic Regression model with TF-IDF vectorization, which achieved:
- **Overall Accuracy**: 87%
- **Categories**: Credit Card, Credit Reporting, Debt Collection, Mortgages & Loans, Retail Banking

## Required Packages

The project requires the following Python packages:

```bash
flask==3.1.3          # Web framework
scikit-learn==1.6.1   # Machine learning library
pandas==2.3.3          # Data manipulation
numpy==1.26.4          # Numerical computing
nltk==3.9.2            # Natural language processing
gensim==4.4.0          # Word embeddings (for Word2Vec)
tabulate==0.10.0       # Table formatting
dotenv==0.9.9          # Environment variable management
matplotlib             # Data visualization (optional for training)
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the following files exist:
- `data/complaints.csv` (training data)
- Models will be created during training process

## Training the Model

To train the machine learning models from scratch:

1. **Prepare the training data:**
   - Ensure `data/complaints.csv` contains the complaint data with 'product' and 'narrative' columns
   - The script will automatically clean the text data

2. **Run the training script:**
```bash
python train.py
```

3. **Training process includes:**
   - Text preprocessing (lowercase, remove punctuation, digits, stopwords)
   - Stratified K-Fold cross-validation (5 folds) for model evaluation
   - Comparison of multiple models:
     - Logistic Regression
     - SGD Classifier
     - Multinomial Naive Bayes
   - Training the best performing model (SGD Classifier)
   - Saving the trained model and TF-IDF vectorizer

4. **Output files created:**
   - `models/vectorizer/tfidf_vectorizer.pkl` - TF-IDF vectorizer
   - `models/best_model.pkl` - Trained SGD model
   - `analysis/stratified_k_fold_for_analysis.csv` - Cross-validation results

5. **Training details:**
   - Uses TF-IDF vectorization with 10,000 max features
   - N-gram range: (1, 2) for unigrams and bigrams
   - Min document frequency: 3
   - Max document frequency: 0.9
   - Random state: 42 for reproducibility

## Usage

### Web Interface

1. Start the web server:
```bash
python web.py
```

2. Open your browser and go to: `http://localhost:5000`

3. Enter your banking complaint description in the text area

4. Click "Classify Complaint" to see the prediction

### API Usage

Send POST requests to `/api/predict` with JSON payload:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"complaint_text": "I was charged an unexpected fee on my credit card"}'
```

Response format:
```json
{
  "prediction": "credit_card",
  "confidence": 0.85,
  "original_text": "I was charged an unexpected fee on my credit card"
}
```

## Project Structure

```
BankingComplaint/
├── web.py                 # Flask web application
├── train.py              # Model training script
├── templates/
│   └── index.html        # HTML template
├── models/
│   ├── vectorizer/
│   │   └── tfidf_vectorizer.pkl  # TF-IDF vectorizer
│   └── best_model.pkl    # Trained SGD model
├── data/
│   └── complaints.csv    # Training data
├── analysis/
│   └── stratified_k_fold_for_analysis.csv  # Cross-validation results
├── main.ipynb           # Jupyter notebook with model training
└── requirements.txt     # Python dependencies
```

## Model Details

The application implements the same text preprocessing and classification pipeline as in `main.ipynb`:

1. **Text Preprocessing**: Lowercase, remove punctuation, digits, and stopwords
2. **Feature Extraction**: TF-IDF vectorization
3. **Classification**: Logistic Regression
4. **Post-processing**: Format predictions for display

## Development

The web application loads the model and vectorizer at startup to ensure fast predictions. The vectorizer is recreated using the training data to maintain consistency with the original model training process.

## Troubleshooting

- **Model Loading Errors**: Ensure `models/best_model.pkl` and `models/vectorizer/tfidf_vectorizer.pkl` exist and are accessible
- **Data Loading Errors**: Verify `data/complaints.csv` exists in the correct location with 'product' and 'narrative' columns
- **Import Errors**: Install all required packages using `pip install -r requirements.txt`
- **NLTK Download Issues**: The training script automatically downloads stopwords. If issues occur, manually run: `python -c "import nltk; nltk.download('stopwords')"`
- **Port Conflicts**: The application runs on port 5000 by default. Modify `web.py` to change the port if needed
- **Training Issues**: 
  - Ensure sufficient disk space for model files
  - Check that the CSV file is not corrupted
  - Verify Python version compatibility (3.7+ recommended)
- **Memory Issues**: For large datasets, consider reducing `max_features` in the TF-IDF vectorizer or using a smaller sample of the data