import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

RANDOM_STATE = os.getenv('RANDOM_STATE', 42)
STRATIFIED_K_FOLD_SPLIT = os.getenv('STRATIFIED_K_FOLD_SPLIT', 5)
STRATIFIED_K_FOLD_FILE = os.getenv('STRATIFIED_K_FOLD_FILE', 'analysis/stratified_k_fold_for_analysis.csv')
VECTORIZER_FILE = os.getenv('VECTORIZER_FILE', 'models/vectorizer/tf_idf_vectorizer.pkl')
BEST_MODEL_FILE = os.getenv('BEST_MODEL_FILE', 'models/best_model.pkl')

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load and preprocess data
df = pd.read_csv('data/complaints.csv')
df = df[['product', 'narrative']]
df.dropna(inplace=True)

def clean_text(text: str):
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

print("Cleaning text...")

df['product'] = df.apply(lambda row: clean_text(row['product']), axis=1)
df['narrative'] = df.apply(lambda row: clean_text(row['narrative']), axis=1)

# Visualize product distribution
product_counts = df['product'].value_counts()

print("Visualizing product distribution...")
def absolute_value(pct):
    total = sum(product_counts)
    val = int(pct * total / 100)
    return f'{val} ({pct:.1f}%)'

plt.pie(product_counts, labels=product_counts.index, autopct='%1.1f%%')
plt.xlabel('Product')
plt.ylabel('Distribution')
plt.title('Product Distribution')
# plt.show()
print("Skipped..")

tabulate_body = []
tabulate_headers = ["Model", "Mean F1", "Std F1", "Max F1 by Std", "Min F1 by Std"]

def stratified_k_fold(model, name):
    stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )

    X = df['narrative']
    Y = df['product']

    f1_scores = []

    print(f"Training Stratified K Fold TF-IDF for {name} ...")
    for train_index, test_index in stf.split(X, Y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)

        print(f"Got {f1} F1 score for {len(f1_scores)} fold(s)")

    print(f"F1 per fold for {name}:", f1_scores)
    print(f"Mean F1 for {name}:", np.mean(f1_scores))
    print(f"Std F1 for {name}:", np.std(f1_scores))
    print(f"Max F1 by Std for {name}:", float(np.mean(f1_scores)) + float(np.std(f1_scores)))
    print(f"Min F1 by Std for {name}:", float(np.mean(f1_scores)) - float(np.std(f1_scores)))

    tabulate_body.append([name,
                          float(np.mean(f1_scores)),
                          float(np.std(f1_scores)),
                          float(np.mean(f1_scores)) + float(np.std(f1_scores)),
                          float(np.mean(f1_scores)) - float(np.std(f1_scores))])


print("Running stratified k fold...")
if not os.path.isfile(STRATIFIED_K_FOLD_FILE):
    stratified_k_fold(LogisticRegression(max_iter=1000, class_weight='balanced'), 'Logistic Regression')
    stratified_k_fold(SGDClassifier(loss='hinge', random_state=RANDOM_STATE), 'SGD')
    stratified_k_fold(MultinomialNB(), 'MultinomialNB')

    # uncomment this table if you want to show it
    # print(tabulate(tabulate_body, headers=tabulate_headers, tablefmt="fancy_grid"))

    print("Ensuring analysis/* directory exists...")
    os.makedirs('analysis', exist_ok=True)
    print("Saving stratified k fold for analysis...")
    saved_tabulate = pd.DataFrame(data=tabulate_body, columns=tabulate_headers)
    saved_tabulate.to_csv(STRATIFIED_K_FOLD_FILE, index=False)
    print("Analysis csv saved.")
else:
    print("Stratified k fold analysis file already exists.")
    print("Please remove the file if you want to re-run the analysis.")
    print("Skipped.")

# because SGD is good for this case, we will use it, now we will train it
print("Training SGD started...")
print("Splitting data...")
# Split data
x_train, x_test, y_train, y_test = train_test_split(df['narrative'], df['product'], test_size=0.2, random_state=42)

print("Vectorizing data...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9
)

print(f"Save vectorizer to {VECTORIZER_FILE}...")

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

print("Ensuring vectorizer directory is exists...")
os.makedirs('models/vectorizer', exist_ok=True)
with open(VECTORIZER_FILE, 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training SGD...")
sgd_model = SGDClassifier(loss='hinge')
sgd_model.fit(x_train, y_train)

print("Training SGD finished.")
print("Saving SGD model...")
with open(BEST_MODEL_FILE, 'wb') as f:
    pickle.dump(sgd_model, f)
print("SGD model saved.")

